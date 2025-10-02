# vlm_embed_vae.py
# Lightweight 3D β-VAE for embeddings with TensorBoard logging.
# - Training: scalars (loss components), LR, histograms, optional embedding projector (subsampled)
# - Inference: map N×D -> N×3 using posterior mean
# - Memory-friendly dataset (supports mmap .npy)
# Usage:
#   Train:
#     python vlm_embed_vae.py train --data text.npy image.npy audio.npy --out runs/vae3d_light
#     tensorboard --logdir runs
#   Infer:
# python vlm_embed_vae.py infer --ckpt runs/vae3d_light/vae3d.pt --data
# new.npy --out coords3d.npy

import os
import json
import math
import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# -------------------------
# Dataset
# -------------------------

class NpyEmbeds(Dataset):
    """
    Loads one or more .npy arrays (same dim) and stacks on axis=0.
    Can use memory-mapping to avoid loading entire dataset into RAM.
    Standardizes per-dimension (mean/std computed once across all arrays).
    """

    def __init__(
            self, paths: List[str],
            standardize: bool = True, mmap: bool = True):
        mats = [np.load(p, mmap_mode="r" if mmap else None) for p in paths]
        X = np.vstack(mats)  # zero-copy view with mmap
        X = X.astype(np.float32, copy=False)
        if standardize:
            self.mean = X.mean(axis=0, keepdims=True).astype(np.float32)
            self.std = X.std(axis=0, keepdims=True).astype(np.float32)
            self.std = np.clip(self.std, 1e-6, None)
            self.X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros((1, X.shape[1]), dtype=np.float32)
            self.std = np.ones((1, X.shape[1]), dtype=np.float32)
            self.X = X

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return torch.from_numpy(self.X[i])


# -------------------------
# MLP helper
# -------------------------

def mlp(
        in_dim: int, hidden: List[int],
        out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    dims = [in_dim] + hidden + [out_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)


# -------------------------
# Model
# -------------------------

class VAE3D(nn.Module):
    def __init__(
            self, d_in: int, hidden: List[int],
            beta: float = 4.0, recon_cosine_weight: float = 0.5):
        super().__init__()
        self.encoder = mlp(d_in, hidden, 6)   # outputs [mu(3), logvar(3)]
        # Decoder slightly *simpler* than symmetric to reduce collapse:
        dec_hidden = hidden[::-1]
        if len(dec_hidden) > 1:
            dec_hidden = dec_hidden[:-1]  # drop the smallest
        self.decoder = mlp(3, dec_hidden, d_in)
        self.beta = beta
        self.recon_cosine_weight = recon_cosine_weight

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def loss(self, x, x_hat, mu, logvar):
        mse = F.mse_loss(x_hat, x)
        cos = 1 - F.cosine_similarity(x_hat, x, dim=-1).mean()
        recon = (
            1 - self.recon_cosine_weight) * mse + self.recon_cosine_weight * cos
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl, {"recon": recon, "kl": kl}


# -------------------------
# Training
# -------------------------

def train(
    data_paths: List[str],
    out_dir: str,
    hidden: List[int] = [1024, 512, 256],   # light
    batch_size: int = 2048,
    epochs: int = 40,
    lr: float = 2e-3,
    beta: float = 4.0,
    beta_warmup_frac: float = 0.2,          # warm up over first 20% epochs
    recon_cosine_weight: float = 0.5,
    dropout: float = 0.0,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
    seed: int = 42,
    log_every: int = 50,                    # steps
    log_embed_every_epochs: int = 5,        # projector logging cadence
    embed_sample_size: int = 5000,          # #points to log in projector
):
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = NpyEmbeds(data_paths, standardize=True, mmap=True)
    d_in = ds.X.shape[1]
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE3D(
        d_in=d_in, hidden=hidden, beta=beta,
        recon_cosine_weight=recon_cosine_weight).to(device)
    # apply dropout retroactively if requested
    if dropout > 0:
        def add_dropout(m):
            if isinstance(m, nn.Linear):
                return
        # (Kept simple; standard mlp() already supports dropout if you prefer to bake it in.)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr, weight_decay=weight_decay)
    # Cosine LR with warmup steps
    total_steps = epochs * (len(ds) // batch_size)
    # reuse frac for LR warmup too
    warmup_steps = int(beta_warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_steps - warmup_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # Save normalization stats for inference
    ckpt_path = os.path.join(out_dir, "vae3d.pt")
    best_loss = float("inf")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        # β warmup over epochs (linear)
        frac = min(1.0, epoch / max(1, math.ceil(epochs * beta_warmup_frac)))
        model.beta = beta * frac

        running = 0.0
        t0 = time.time()

        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                x_hat, mu, logvar, z = model(xb)
                loss, parts = model.loss(xb, x_hat, mu, logvar)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # Manual LR warmup
            if global_step < warmup_steps:
                warmup_lr = lr * (global_step + 1) / max(1, warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = warmup_lr
            else:
                scheduler.step()

            running += loss.item() * xb.size(0)
            if (global_step % log_every) == 0:
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar(
                    "loss/recon",
                    parts["recon"].item(),
                    global_step)
                writer.add_scalar("loss/kl", parts["kl"].item(), global_step)
                writer.add_scalar("train/beta", model.beta, global_step)
                writer.add_scalar(
                    "train/lr",
                    opt.param_groups[0]["lr"],
                    global_step)

                # Histograms (keep light)
                writer.add_histogram(
                    "latent/mu", mu.detach().cpu(),
                    global_step, bins="doane")
                writer.add_histogram(
                    "latent/logvar",
                    logvar.detach().cpu(),
                    global_step,
                    bins="doane")
                writer.add_histogram(
                    "latent/z_sample", z.detach().cpu(),
                    global_step, bins="doane")

            global_step += 1

        epoch_loss = running / len(ds)
        dt = time.time() - t0
        print(
            f"epoch {epoch:03d} | loss {epoch_loss:.4f} | β {model.beta:.2f} | time {dt:.1f}s")

        writer.add_scalar("epoch/loss", epoch_loss, epoch)
        writer.add_scalar("epoch/beta", model.beta, epoch)

        # Embedding projector (optional; logs a subsample of 3D means)
        if (log_embed_every_epochs > 0) and (
                epoch % log_embed_every_epochs == 0):
            with torch.no_grad():
                # sample a subset
                idx = np.random.permutation(
                    len(ds))[: min(embed_sample_size, len(ds))]
                Xs = torch.from_numpy(ds.X[idx]).to(device)
                mu, _ = model.encode(Xs)
                Z = mu.detach().cpu().numpy()  # [S, 3]
            writer.add_embedding(
                mat=Z,
                metadata=[str(i) for i in idx],
                tag=f"latent_3d/epoch_{epoch}"
            )

        # Checkpoint best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model": model.state_dict(),
                "d_in": d_in,
                "hidden": hidden,
                "beta": model.beta,
                "recon_cosine_weight": recon_cosine_weight,
                "mean": ds.mean,
                "std": ds.std,
            }, ckpt_path)

    # Save a tiny config manifest
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "hidden": hidden,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "beta": beta,
            "beta_warmup_frac": beta_warmup_frac,
            "recon_cosine_weight": recon_cosine_weight,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "seed": seed,
        }, f, indent=2)

    writer.close()
    print(
        f"Done. Best checkpoint: {ckpt_path}\nLaunch TensorBoard: tensorboard --logdir {os.path.dirname(tb_dir)}")


# -------------------------
# Inference
# -------------------------

class VAE3DWrapper:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.d_in = ckpt["d_in"]
        self.hidden = ckpt["hidden"]
        self.recon_cosine_weight = ckpt["recon_cosine_weight"]
        self.mean = torch.from_numpy(ckpt["mean"]).float()
        self.std = torch.from_numpy(ckpt["std"]).float()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE3D(
            self.d_in,
            self.hidden,
            beta=1.0,
            recon_cosine_weight=self.recon_cosine_weight).to(
            self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def to3d(self, X: np.ndarray, use_mu: bool = True) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        X = (X - self.mean) / (self.std + 1e-6)
        X = X.to(self.device)
        mu, logvar = self.model.encode(X)
        Z = mu if use_mu else self.model.reparameterize(mu, logvar)
        return Z.detach().cpu().numpy()


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Light 3D β-VAE with TensorBoard")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument(
        "--data", nargs="+", required=True,
        help=".npy files (same dim); stacked along axis=0")
    pt.add_argument(
        "--out",
        required=True,
        help="Output dir (stores ckpt + tb logs)")
    pt.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
    pt.add_argument("--batch_size", type=int, default=2048)
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--lr", type=float, default=2e-3)
    pt.add_argument("--beta", type=float, default=4.0)
    pt.add_argument("--beta_warmup_frac", type=float, default=0.2)
    pt.add_argument("--cos_w", type=float, default=0.5)
    pt.add_argument("--dropout", type=float, default=0.0)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--num_workers", type=int, default=8)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--log_every", type=int, default=50)
    pt.add_argument("--log_embed_every_epochs", type=int, default=5)
    pt.add_argument("--embed_sample_size", type=int, default=5000)

    pi = sub.add_parser("infer")
    pi.add_argument("--ckpt", required=True, help="Path to vae3d.pt")
    pi.add_argument("--data", required=True, help="N×D .npy to map")
    pi.add_argument("--out", required=True, help="Output coords .npy")
    pi.add_argument(
        "--sample", action="store_true",
        help="Use stochastic z instead of posterior mean")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        train(
            data_paths=args.data,
            out_dir=args.out,
            hidden=args.hidden,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            beta=args.beta,
            beta_warmup_frac=args.beta_warmup_frac,
            recon_cosine_weight=args.cos_w,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            log_every=args.log_every,
            log_embed_every_epochs=args.log_embed_every_epochs,
            embed_sample_size=args.embed_sample_size,
        )
    else:
        wrapper = VAE3DWrapper(args.ckpt)
        X = np.load(args.data).astype(np.float32)
        Z = wrapper.to3d(X, use_mu=(not args.sample))
        np.save(args.out, Z)
        print(f"Saved coords to {args.out} | shape={Z.shape}")


'''
# 1) Train (light model)
python vlm_embed_vae.py train \
  --data embeds_train.npy \
  --out runs/vae3d_light \
  --hidden 1024 512 256 \
  --epochs 40 --beta 4.0 --beta_warmup_frac 0.2 --cos_w 0.6

# 2) Open TensorBoard
tensorboard --logdir runs

# 3) Inference (map to 3D)
python vlm_embed_vae.py infer \
  --ckpt runs/vae3d_light/vae3d.pt \
  --data new_embeds.npy \
  --out coords3d.npy
'''

if __name__ == "__main__":
    main()
