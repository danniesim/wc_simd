"""Lightweight autoencoder training script for VLM embeddings.

Replaces the previous β-VAE implementation with a cosine-regularised
deterministic autoencoder.  The training loop matches the reference snippet
provided by the user while keeping the original CLI ergonomics minimal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

D_IN = 1536
D_LATENT = 3
DEFAULT_HIDDEN: Tuple[int, ...] = (512, 128)
SIM_WEIGHT = 0.5
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NpyEmbeds(Dataset):
    """Streaming dataset over one or more .npy arrays."""

    def __init__(self, paths: Sequence[str], mmap: bool = True) -> None:
        if not paths:
            raise ValueError("Expected at least one data path.")
        self.paths = list(paths)
        self._arrays: List[np.ndarray] = []
        for path in self.paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing data file: {path}")
            arr = np.load(path, mmap_mode="r" if mmap else None)
            if arr.ndim != 2:
                raise ValueError(
                    f"{path}: expected 2D array, got {arr.ndim}D.")
            self._arrays.append(arr)
        dims = {arr.shape[1] for arr in self._arrays}
        if len(dims) != 1:
            raise ValueError(
                "All input arrays must share the same feature dimension.")
        self.feature_dim = dims.pop()
        lengths = [arr.shape[0] for arr in self._arrays]
        self._cum = np.cumsum([0] + lengths)
        logger.info(
            "Loaded %s rows across %d files (dim=%d)",
            f"{self._cum[-1]:,}",
            len(self._arrays),
            self.feature_dim,
        )

    def __len__(self) -> int:
        return int(self._cum[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}.")
        if len(self._arrays) == 1:
            return 0, idx
        file_idx = int(np.searchsorted(self._cum, idx, side="right") - 1)
        local = idx - self._cum[file_idx]
        return file_idx, local

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, local = self._locate(idx)
        row_np = self._arrays[file_idx][local]
        if not row_np.flags.writeable:
            row_np = np.array(row_np, copy=True)
        return torch.from_numpy(row_np).float()


def build_dataloader(
    paths: Sequence[str],
    batch_size: int,
    num_workers: int,
    seed: Optional[int],
    device: torch.device,
    mmap: bool,
) -> Tuple[NpyEmbeds, DataLoader]:
    dataset = NpyEmbeds(paths, mmap=mmap)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=max(num_workers, 0),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        generator=generator if seed is not None else None,
    )
    return dataset, loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AE(nn.Module):
    def __init__(
        self,
        d_in: int = D_IN,
        d_latent: int = D_LATENT,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_latent <= 0:
            raise ValueError("d_in and d_latent must be positive integers.")
        hidden = list(hidden) if hidden is not None else []
        for width in hidden:
            if width <= 0:
                raise ValueError("Hidden layer widths must be positive.")

        enc_layers: List[nn.Module] = []
        last = d_in
        for width in hidden:
            enc_layers.extend(
                [nn.Linear(last, width),
                 nn.LayerNorm(width),
                 nn.GELU()])
            last = width
        enc_layers.append(nn.Linear(last, d_latent))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        last = d_latent
        for width in reversed(hidden):
            dec_layers.extend(
                [nn.Linear(last, width),
                 nn.LayerNorm(width),
                 nn.GELU()])
            last = width
        dec_layers.append(nn.Linear(last, d_in))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def pairwise_cosine(a: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    return a @ a.T


def loss_fn(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    sim_weight: float,
) -> Tuple[torch.Tensor, dict]:
    if sim_weight < 0:
        raise ValueError("sim_weight must be non-negative.")
    x_n = F.normalize(x, dim=-1)
    xh_n = F.normalize(x_hat, dim=-1)
    recon = F.mse_loss(xh_n, x_n)
    with torch.no_grad():
        sim_x = pairwise_cosine(x_n)
    sim_z = pairwise_cosine(z)
    sim_loss = F.mse_loss(sim_z, sim_x)
    total = recon + sim_weight * sim_loss
    return total, {
        "recon": float(recon.detach()),
        "sim": float(sim_loss.detach())}


# ---------------------------------------------------------------------------
# Training / Inference helpers
# ---------------------------------------------------------------------------


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return DEVICE
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    return dev


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast()  # type: ignore[attr-defined]


def train_autoencoder(
    data_paths: Sequence[str],
    out_dir: str,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    hidden: Sequence[int],
    sim_weight: float,
    num_workers: int,
    seed: Optional[int],
    device: torch.device,
    weight_decay: float,
    log_every: int,
    mmap: bool,
) -> AE:
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if sim_weight < 0:
        raise ValueError("sim_weight must be non-negative.")

    os.makedirs(out_dir, exist_ok=True)
    dataset, loader = build_dataloader(
        data_paths, batch_size, num_workers, seed, device, mmap
    )
    model = AE(
        d_in=dataset.feature_dim,
        d_latent=latent_dim,
        hidden=hidden).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history: List[dict] = []
    best_loss = float("inf")
    ckpt_path = os.path.join(out_dir, "ae.pt")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_sim = 0.0
        total_samples = 0

        for batch_idx, xb in enumerate(loader):
            if not torch.is_tensor(xb):
                raise TypeError("Dataloader must return a tensor batch.")
            xb = xb.to(device, non_blocking=True)
            xb = F.normalize(xb, dim=-1)
            opt.zero_grad(set_to_none=True)

            with autocast_context(device):
                z, x_hat = model(xb)
                loss, parts = loss_fn(xb, x_hat, z, sim_weight=sim_weight)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            batch_size_actual = xb.shape[0]
            total_samples += batch_size_actual
            total_loss += float(loss.detach()) * batch_size_actual
            total_recon += parts["recon"] * batch_size_actual
            total_sim += parts["sim"] * batch_size_actual

            global_step += 1
            if log_every > 0 and (global_step % log_every) == 0:
                logger.info(
                    "epoch %02d step %06d | loss %.4f | recon %.4f | sim %.4f",
                    epoch,
                    global_step,
                    parts["recon"] + sim_weight * parts["sim"],
                    parts["recon"],
                    parts["sim"],
                )

        if total_samples == 0:
            raise RuntimeError("No samples were processed during training.")

        mean_loss = total_loss / total_samples
        mean_recon = total_recon / total_samples
        mean_sim = total_sim / total_samples
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "recon": mean_recon,
                "sim": mean_sim,
            }
        )

        logger.info(
            "epoch %02d complete | loss %.4f | recon %.4f | sim %.4f",
            epoch,
            mean_loss,
            mean_recon,
            mean_sim,
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            tmp_path = ckpt_path + ".tmp"
            torch.save(
                {
                    "model": model.state_dict(),
                    "d_in": dataset.feature_dim,
                    "latent_dim": latent_dim,
                    "hidden": list(hidden),
                    "sim_weight": sim_weight,
                },
                tmp_path,
            )
            os.replace(tmp_path, ckpt_path)
            logger.info(
                "Saved checkpoint to %s (loss=%.4f).",
                ckpt_path,
                mean_loss)

    history_path = os.path.join(out_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Wrote training history to %s.", history_path)

    return model


def load_model(ckpt_path: str, device: torch.device) -> Tuple[AE, dict]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    for key in ("model", "d_in", "latent_dim", "hidden"):
        if key not in payload:
            raise KeyError(f"Checkpoint missing required key: {key}")
    model = AE(
        d_in=int(payload["d_in"]),
        d_latent=int(payload["latent_dim"]),
        hidden=payload.get("hidden", DEFAULT_HIDDEN),
    )
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc
    return model.to(device), payload


def infer_embeddings(
    ckpt_path: str,
    data_path: str,
    out_path: str,
    *,
    batch_size: int,
    device: torch.device,
    normalize_latent: bool,
) -> None:
    model, payload = load_model(ckpt_path, device=device)
    model.eval()
    input_dim = int(payload["d_in"])
    latent_dim = int(payload["latent_dim"])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    mat = np.load(data_path, mmap_mode="r")
    if mat.ndim != 2:
        raise ValueError(f"{data_path}: expected 2D array, got {mat.ndim}D.")
    if mat.shape[1] != input_dim:
        raise ValueError(
            f"Data dim {mat.shape[1]} does not match model input {input_dim}.")

    total = mat.shape[0]
    coords = np.empty((total, latent_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = mat[start:end].astype(np.float32, copy=False)
            xb = torch.from_numpy(batch).to(device)
            xb = F.normalize(xb, dim=-1)
            z = model.encoder(xb)
            if normalize_latent:
                z = F.normalize(z, dim=-1)
            coords[start:end] = z.detach().cpu().numpy()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, coords)
    logger.info("Saved %s embeddings to %s.", coords.shape[0], out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run the cosine-regularised autoencoder."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train the autoencoder.")
    pt.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more .npy files.")
    pt.add_argument("--out", required=True, help="Output directory.")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch_size", type=int, default=4096)
    pt.add_argument("--lr", type=float, default=LR)
    pt.add_argument("--latent_dim", type=int, default=D_LATENT)
    pt.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=None,
        help="Hidden layer widths (defaults to 512 128).",
    )
    pt.add_argument("--sim_weight", type=float, default=SIM_WEIGHT)
    pt.add_argument("--num_workers", type=int, default=4)
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("--device", type=str, default="auto")
    pt.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    pt.add_argument("--log_every", type=int, default=0)
    pt.add_argument(
        "--no_mmap",
        action="store_true",
        help="Disable numpy memmap.")

    pi = sub.add_parser("infer", help="Encode data with a trained model.")
    pi.add_argument("--ckpt", required=True, help="Path to ae.pt checkpoint.")
    pi.add_argument("--data", required=True, help="Input .npy file.")
    pi.add_argument(
        "--out",
        required=True,
        help="Output .npy file for latents.")
    pi.add_argument("--batch_size", type=int, default=8192)
    pi.add_argument("--device", type=str, default="auto")
    pi.add_argument(
        "--normalize_latent",
        action="store_true",
        help="L2-normalize latent vectors before saving.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.cmd == "train":
            hidden = tuple(args.hidden) if args.hidden else DEFAULT_HIDDEN
            device = resolve_device(args.device)
            train_autoencoder(
                data_paths=args.data,
                out_dir=args.out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                latent_dim=args.latent_dim,
                hidden=hidden,
                sim_weight=args.sim_weight,
                num_workers=args.num_workers,
                seed=args.seed,
                device=device,
                weight_decay=args.weight_decay,
                log_every=args.log_every,
                mmap=not args.no_mmap,
            )
        elif args.cmd == "infer":
            device = resolve_device(args.device)
            infer_embeddings(
                args.ckpt,
                args.data,
                args.out,
                batch_size=args.batch_size,
                device=device,
                normalize_latent=args.normalize_latent,
            )
        else:
            raise ValueError(f"Unknown command {args.cmd!r}.")
    except Exception as exc:
        logger.exception("Command failed.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
