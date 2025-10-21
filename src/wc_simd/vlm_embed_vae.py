# vlm_embed_vae.py
# Lightweight 3D β-VAE for embeddings with TensorBoard logging.
# - Training: scalars (loss components), LR, histograms, optional embedding projector (subsampled)
# - Inference: map N×D -> N×3 using posterior mean
# - Memory-friendly lazy dataset (supports mmap .npy, assumes normalized data)
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
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger(__name__)


# -------------------------
# Dataset
# -------------------------

class NpyEmbeds(Dataset):
    """Streaming dataset over one or more .npy arrays.

    Simplified version that assumes data is already normalized and always uses
    lazy/streaming access with memory-mapped files.

    Features:
    - Lazy / streaming access (no full concatenation)
    - Memory-mapped file access for efficiency
    - Assumes input data is already normalized
    """

    def __init__(
        self,
        paths: List[str],
        mmap: bool = True,
    ):
        import time
        print(f"Initializing dataset with {len(paths)} files...")
        t0 = time.time()

        self.paths = paths

        print(f"Loading file metadata...")
        mats = [np.load(p, mmap_mode="r" if mmap else None) for p in paths]
        print(f"File loading took: {time.time() - t0:.2f}s")

        # Always use lazy mode setup
        self.arrays = mats
        lengths = [m.shape[0] for m in mats]
        self.cum_lengths = np.cumsum([0] + lengths)  # len = n_files + 1
        self.feature_dim = mats[0].shape[1]

        print(
            f"Dataset size: {self.cum_lengths[-1]:,} samples, {self.feature_dim} dims")

        # No standardization - assume data is already normalized
        self.mean = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.std = np.ones((1, self.feature_dim), dtype=np.float32)
        self.mean_t = torch.from_numpy(self.mean[0])  # 1D
        self.std_t = torch.from_numpy(self.std[0])

        print(f"Dataset initialization complete: {time.time() - t0:.2f}s")

    def __len__(self):
        return int(self.cum_lengths[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        # Binary search over cum_lengths (optimized for single file case)
        if len(self.cum_lengths) == 2:  # Single file optimization
            return 0, idx
        file_idx = int(
            np.searchsorted(
                self.cum_lengths,
                idx,
                side="right") - 1)
        local = idx - self.cum_lengths[file_idx]
        return file_idx, local

    def __getitem__(self, i):
        file_idx, local = self._locate(i)
        row_np = self.arrays[file_idx][local]
        # Convert to torch. For read-only memmap slices
        # torch.from_numpy emits a warning (non-writable). We only read, but to
        # avoid the warning (and any accidental in-place ops) create a writable
        # view only when necessary.
        if not row_np.flags.writeable:
            # Minimal overhead: single row copy (cheap vs batch size stacking)
            row_np = np.array(row_np, copy=True)
        return torch.from_numpy(row_np).float()

    # Helper for projector logging (returns numpy batch)
    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        out = np.empty((len(indices), self.feature_dim), dtype=np.float32)
        # Group indices per file for efficiency
        indices = np.asarray(indices)
        order = np.argsort(indices)
        sorted_idx = indices[order]
        ptr = 0
        for file_idx in range(len(self.arrays)):
            start = self.cum_lengths[file_idx]
            end = self.cum_lengths[file_idx + 1]
            mask = (sorted_idx >= start) & (sorted_idx < end)
            if not np.any(mask):
                continue
            rel = sorted_idx[mask] - start
            block = self.arrays[file_idx][rel].astype(np.float32, copy=False)
            count = block.shape[0]
            out[order[ptr:ptr + count]] = block
            ptr += count
        return out


class ChunkShuffleSampler(Sampler[int]):
    """Approximate global shuffle without allocating len(dataset) upfront.

    Torch's default RandomSampler+shuffle=True materializes a full randperm
    tensor every epoch which is prohibitively slow for tens of millions of
    rows.  This sampler only shuffles within fixed-size chunks so we keep the
    startup cost bounded while still decorrelating batches sufficiently for
    SGD.  Chunk order varies every epoch via the shared RNG state.
    """

    def __init__(
            self,
            data_source: Dataset,
            chunk_size: int,
            seed: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.data_source = data_source
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        import torch

        n = len(self.data_source)
        if n == 0:
            return iter(())

        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        chunk_size = min(self.chunk_size, n)

        def _iter():
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                block_len = end - start
                perm = torch.randperm(block_len, generator=g).tolist()
                for offset in perm:
                    yield start + offset

        return _iter()


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
            beta: float, recon_cosine_weight: float,
            latent_dim: int,
            dropout: float = 0.0,
            recon_mse_mode: str = "element_mean",
        # element_mean | scale_by_dim | sum_per_sample | disabled
            recon_mse_manual_scale: Optional[float] = None,
            recon_log_extra: bool = True,
            kl_uniform_weight: float = 0.0,
            ae_mode: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        # outputs [mu(D), logvar(D)]
        self.encoder = mlp(d_in, hidden, latent_dim * 2, dropout=dropout)
        # Decoder slightly *simpler* than symmetric to reduce collapse:
        dec_hidden = hidden[::-1]
        if len(dec_hidden) > 1:
            dec_hidden = dec_hidden[:-1]  # drop the smallest
        self.decoder = mlp(latent_dim, dec_hidden, d_in, dropout=dropout)
        self.beta = beta
        self.recon_cosine_weight = recon_cosine_weight
        self.recon_mse_mode = recon_mse_mode
        self.recon_mse_manual_scale = recon_mse_manual_scale
        self.recon_log_extra = recon_log_extra
        # Weight for variance penalty across per-dimension KL to avoid
        # single-dim collapse
        self.kl_uniform_weight = kl_uniform_weight
        # If ae_mode=True we behave as a deterministic autoencoder (no KL, no
        # sampling)
        self.ae_mode = ae_mode

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.ae_mode:
            # Deterministic pass-through (no noise)
            return mu
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

    def loss(self, x, x_hat, mu, logvar,
             beta: float,
             free_bits: float = 0.0,
             capacity_C: Optional[float] = None,
             beta_capacity: Optional[float] = None):
        """Compute VAE objective variants.

        Modes supported (chosen implicitly by args):
        1. Standard β-VAE (capacity_C is None):
              loss = recon + beta * KL_used
           where free bits (if >0) clamp per-dim KL minima.
        2. Capacity (Burgess et al.) if capacity_C provided:
              loss = recon + beta_capacity * |KL_raw - C(t)|
           'beta_capacity' acts like γ in the paper; 'beta' is ignored.

        Returns
        -------
        loss : torch.Tensor
        parts : dict with recon, kl_raw, kl_used, capacity_C, active_dims
        """
        # Raw element-wise mean MSE (baseline, invariant to dim & batch)
        mse_element_mean = F.mse_loss(x_hat, x)
        # Cosine term (already mean over batch)
        cos_term = 1 - F.cosine_similarity(x_hat, x, dim=-1).mean()

        # Choose MSE variant
        if self.recon_mse_mode == "element_mean":
            mse_used = mse_element_mean
        elif self.recon_mse_mode == "scale_by_dim":
            mse_used = mse_element_mean * x.shape[-1]
        elif self.recon_mse_mode == "sum_per_sample":
            # Sum over features per sample, average over batch
            mse_used = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        elif self.recon_mse_mode == "disabled":
            mse_used = mse_element_mean * 0.0  # keep graph
        else:
            raise ValueError(f"Unknown recon_mse_mode={self.recon_mse_mode}")

        if self.recon_mse_manual_scale is not None:
            mse_used = mse_used * float(self.recon_mse_manual_scale)

        recon = (
            1 - self.recon_cosine_weight) * mse_used + self.recon_cosine_weight * cos_term

        if self.ae_mode:
            # Pure autoencoder: no KL term, return zeros for diagnostics
            kl_raw = torch.tensor(0.0, device=x.device)
            kl_used = torch.tensor(0.0, device=x.device)
            active_dims = torch.tensor(0, device=x.device)
            capacity_C = None  # ensure logged as 0 below
            kl_var = torch.tensor(0.0, device=x.device)
            kl_uniform_penalty = torch.tensor(0.0, device=x.device)
            total = recon  # reconstruction-only objective
        else:
            # KL per sample per dim (batch, latent_dim)
            kl_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
            kl_per_sample = kl_elem.sum(dim=-1)  # (batch,)
            kl_raw = kl_per_sample.mean()        # scalar
            kl_per_dim = kl_elem.mean(0)         # (latent_dim,)

            active_dims = (
                kl_per_dim > max(
                    1e-6,
                    free_bits if free_bits > 0 else 0.01)).sum()

            if capacity_C is not None:  # capacity variant
                # Use raw KL for distance to target capacity
                gamma = beta_capacity if beta_capacity is not None else beta
                kl_obj = torch.abs(kl_raw - capacity_C)
                total = recon + gamma * kl_obj
                kl_used = kl_raw  # for logging
            else:
                # Free bits (per dim minimum)
                if free_bits > 0.0:
                    kl_per_dim_fb = torch.clamp(kl_per_dim, min=free_bits)
                    kl_used = kl_per_dim_fb.sum()
                else:
                    kl_used = kl_raw
                total = recon + beta * kl_used

            kl_uniform_penalty = torch.tensor(0.0, device=x.device)
            if self.kl_uniform_weight > 0 and self.latent_dim > 1:
                mask = kl_per_dim > 1e-6
                if mask.sum() > 1:
                    kl_var = kl_per_dim[mask].var(unbiased=False)
                    kl_uniform_penalty = kl_var
                    total = total + self.kl_uniform_weight * kl_uniform_penalty
                else:
                    kl_var = torch.tensor(0.0, device=x.device)
            else:
                kl_var = torch.tensor(0.0, device=x.device)

        parts = {
            "recon": recon.detach(),
            "kl_raw": kl_raw.detach(),
            "kl_used": kl_used.detach(),
            "active_dims": active_dims.detach(),
            "capacity_C": torch.tensor(
                0.0 if capacity_C is None else capacity_C), }
        if self.recon_log_extra:
            parts.update({
                "mse_element_mean": mse_element_mean.detach(),
                "mse_used": mse_used.detach(),
                "cos_term": cos_term.detach(),
                "kl_var": kl_var.detach(),
                "kl_uniform_penalty": kl_uniform_penalty.detach(),
            })
        return total, parts


# -------------------------
# Training
# -------------------------

def train(
    data_paths: List[str],
    out_dir: str,
    hidden: List[int],
    batch_size: int,
    epochs: int,
    lr: float,
    beta: float,
    # warm up over first 20% epochs (epochs-based for beta)
    beta_warmup_frac: float,
    lr_warmup_frac: float,         # fraction of total steps for LR warmup
    recon_cosine_weight: float,
    recon_mse_mode: str,
    recon_mse_manual_scale: Optional[float],
    dropout: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
    log_every: int,                  # steps
    log_embed_every_epochs: int,        # projector logging cadence
    embed_sample_size: int,        # #points to log in projector
    # Regularisation variants
    free_bits: float,
    capacity_max: float,
    capacity_warmup_frac: float,
    capacity_gamma: float,
    latent_dim: int,
    kl_uniform_weight: float,
    ae_mode: bool,
    mmap: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # ------------------------------------------------------------------
    # Write config manifest *before* training so it is preserved even if
    # the run is interrupted (e.g. Ctrl+C, preemption, OOM, etc.).
    # This mirrors the post-training manifest for reproducibility.
    # ------------------------------------------------------------------
    early_config_path = os.path.join(out_dir, "config.json")
    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _write_config(data: dict, merge: bool = False):
        """Write (or merge-update) the config manifest safely."""
        try:
            if merge and os.path.exists(early_config_path):
                try:
                    with open(early_config_path, "r") as rf:
                        existing = json.load(rf)
                except Exception:
                    existing = {}
                existing.update(data)
                payload = existing
            else:
                payload = data
            tmp_path = early_config_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, early_config_path)
        except Exception:
            logger.exception("Failed to write config manifest")

    # Initial manifest (superset of hyperparameters + bookkeeping)
    _write_config({
        "data_paths": data_paths,
        "out_dir": out_dir,
        "hidden": hidden,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "beta": beta,
        "beta_warmup_frac": beta_warmup_frac,
        "lr_warmup_frac": lr_warmup_frac,
        "recon_cosine_weight": recon_cosine_weight,
        "dropout": dropout,
        "recon_mse_mode": recon_mse_mode,
        "recon_mse_manual_scale": recon_mse_manual_scale,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "seed": seed,
        "latent_dim": latent_dim,
        "free_bits": free_bits,
        "capacity_max": capacity_max,
        "capacity_warmup_frac": capacity_warmup_frac,
        "capacity_gamma": capacity_gamma,
        "kl_uniform_weight": kl_uniform_weight,
        "ae_mode": ae_mode,
        "log_every": log_every,
        "log_embed_every_epochs": log_embed_every_epochs,
        "embed_sample_size": embed_sample_size,
        "mmap": mmap,
        "start_time": start_time,
        "status": "started",
        "pid": os.getpid()
    })
    logger.info(f"Wrote initial config manifest to {early_config_path}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Creating dataset...")
    t_ds = time.time()
    print(f"Memory map loading: {'enabled' if mmap else 'disabled'}")
    ds = NpyEmbeds(
        data_paths,
        mmap=mmap,
    )
    print(f"Dataset creation took: {time.time() - t_ds:.2f}s")

    d_in = ds.feature_dim

    print("Creating DataLoader...")
    t_dl = time.time()

    shuffle_chunk = max(batch_size * 8, 8192)
    sampler = ChunkShuffleSampler(
        ds, chunk_size=shuffle_chunk, seed=seed)
    print(
        f"Using chunked shuffle sampler with chunk size {shuffle_chunk:,}")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=2,  # Reduced from 4
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    print(f"DataLoader creation took: {time.time() - t_dl:.2f}s")
    print(
        f"Using {num_workers} workers for dataset of {len(ds):,} samples")
    _write_config({"shuffle_chunk_size": shuffle_chunk}, merge=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Creating model...")
    t_model = time.time()
    model = VAE3D(
        d_in=d_in, hidden=hidden, beta=beta,
        recon_cosine_weight=recon_cosine_weight,
        latent_dim=latent_dim,
        dropout=dropout,
        recon_mse_mode=recon_mse_mode,
        recon_mse_manual_scale=recon_mse_manual_scale,
        kl_uniform_weight=kl_uniform_weight,
        ae_mode=ae_mode).to(device)
    print(f"Model creation took: {time.time() - t_model:.2f}s")
    # Dropout integrated directly in encoder/decoder.

    print("Creating optimizer...")
    t_opt = time.time()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr, weight_decay=weight_decay)
    print(f"Optimizer creation took: {time.time() - t_opt:.2f}s")
    # Cosine LR with independent LR warmup fraction
    total_steps = epochs * (len(ds) // batch_size)
    warmup_steps = int(lr_warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_steps - warmup_steps))
    # Mixed precision scaler (prefer new torch.amp API; only create on CUDA)
    if device == "cuda":
        scaler = None
        # Try newest positional API first (torch>=2.4 style)
        try:
            scaler = torch.amp.GradScaler("cuda")  # positional device_type
        except Exception:
            pass
        if scaler is None:
            # Try keyword form (some intermediary versions)
            try:
                scaler = torch.amp.GradScaler(device_type="cuda")
            except Exception:
                pass
        if scaler is None:
            # Last resort: deprecated CUDA namespace (will warn on newer torch)
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        class _NoScaler:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
        scaler = _NoScaler()

    # Save normalization stats for inference
    ckpt_path = os.path.join(out_dir, "vae3d.pt")
    best_loss = float("inf")

    global_step = 0
    total_batches_per_epoch = len(ds) // batch_size
    print(
        f"Starting training: {epochs} epochs, {total_batches_per_epoch:,} batches per epoch")

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            if hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch - 1)
            # β warmup over epochs (linear)
            if not ae_mode:
                frac = min(
                    1.0, epoch / max(1, math.ceil(epochs * beta_warmup_frac)))
                model.beta = beta * frac
            else:
                model.beta = 0.0  # not used, for logging consistency

            running = 0.0
            t0 = time.time()
            print(f"Starting epoch {epoch}/{epochs} (β={model.beta:.3f})...")

            batch_count = 0
            for xb in dl:
                if batch_count == 0:
                    print(f"Processing first batch of epoch {epoch}...")
                xb = xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                batch_count += 1
                autocast_enabled = (device == "cuda")
                # Use new torch.amp.autocast if available
                if autocast_enabled:
                    try:
                        autocast_ctx = torch.amp.autocast(device_type="cuda")
                    except TypeError:  # older torch
                        autocast_ctx = torch.cuda.amp.autocast()
                else:
                    from contextlib import nullcontext
                    autocast_ctx = nullcontext()
                with autocast_ctx:
                    x_hat, mu, logvar, z = model(xb)
                    # Capacity schedule (step based) if enabled
                    if (not ae_mode) and capacity_max > 0:
                        capacity_steps = int(
                            capacity_warmup_frac * total_steps)
                        c_frac = 1.0 if capacity_steps <= 0 else min(
                            1.0, global_step / max(1, capacity_steps))
                        C_t = capacity_max * c_frac
                        loss, parts = model.loss(
                            xb, x_hat, mu, logvar,
                            beta=model.beta,  # still log beta, but gamma used for capacity
                            free_bits=0.0,
                            capacity_C=C_t,
                            beta_capacity=capacity_gamma,
                        )
                    else:  # standard β-VAE loss or AE if ae_mode
                        loss, parts = model.loss(
                            xb, x_hat, mu, logvar,
                            beta=model.beta,
                            free_bits=(0.0 if ae_mode else free_bits),
                            capacity_C=None,
                        )
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
                # Lightweight scalar logging; defer heavier histogram logging
                # until after epoch 1
                if (global_step % log_every) == 0:
                    writer.add_scalar("loss/total", loss.item(), global_step)
                    writer.add_scalar(
                        "loss/recon",
                        parts["recon"].item(),
                        global_step)
                    writer.add_scalar(
                        "loss/kl_raw",
                        parts["kl_raw"].item(),
                        global_step)
                    writer.add_scalar(
                        "loss/kl_used",
                        parts["kl_used"].item(),
                        global_step)
                    # Extra reconstruction diagnostics if present
                    if "mse_element_mean" in parts:
                        writer.add_scalar(
                            "recon/mse_element_mean",
                            parts["mse_element_mean"].item(),
                            global_step)
                    if "mse_used" in parts:
                        writer.add_scalar(
                            "recon/mse_used",
                            parts["mse_used"].item(),
                            global_step)
                    if "cos_term" in parts:
                        writer.add_scalar(
                            "recon/cos_term",
                            parts["cos_term"].item(),
                            global_step)
                    if "kl_var" in parts:
                        writer.add_scalar(
                            "latent/kl_var",
                            parts["kl_var"].item(),
                            global_step)
                    if "kl_uniform_penalty" in parts:
                        writer.add_scalar(
                            "latent/kl_uniform_penalty",
                            parts["kl_uniform_penalty"].item(),
                            global_step)
                    # Ratio diagnostics (avoid div by zero)
                    denom = parts["recon"].item()
                    if denom != 0.0:
                        writer.add_scalar(
                            "diagnostic/kl_over_recon", parts["kl_used"].item() /
                            max(1e-12, denom),
                            global_step)
                    writer.add_scalar(
                        "latent/active_dims",
                        parts["active_dims"].item(),
                        global_step)
                    if capacity_max > 0:
                        writer.add_scalar(
                            "capacity/C_t",
                            parts["capacity_C"].item(),
                            global_step)
                        writer.add_scalar(
                            "capacity/gamma", capacity_gamma, global_step)
                    if not ae_mode:
                        writer.add_scalar(
                            "train/beta", model.beta, global_step)
                    writer.add_scalar(
                        "train/lr",
                        opt.param_groups[0]["lr"],
                        global_step)

                    # Histograms: skip during epoch 1 to reduce startup
                    # overhead
                    if epoch > 1:
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
            logger.info(
                f"epoch {epoch:03d} | loss {epoch_loss:.4f} | β {model.beta:.2f} | time {dt:.1f}s")

            writer.add_scalar("epoch/loss", epoch_loss, epoch)
            writer.add_scalar("epoch/beta", model.beta, epoch)

            # Periodically update manifest with progress (lightweight)
            if epoch == 1 or epoch % max(1, (epochs // 5)) == 0:
                _write_config({
                    "last_epoch": epoch,
                    "last_epoch_loss": epoch_loss,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "status": "running"
                }, merge=True)

            # Embedding projector (optional; logs a subsample of latent means)
            # Projector logging only after first epoch (user request) and then
            # every N epochs
            if (log_embed_every_epochs > 0) and (
                    epoch > 1 and epoch % log_embed_every_epochs == 0):
                with torch.no_grad():
                    idx = np.random.permutation(
                        len(ds))[: min(embed_sample_size, len(ds))]
                    X_block = ds.get_rows(idx)
                    Xs = torch.from_numpy(X_block).to(device)
                    mu, _ = model.encode(Xs)
                    # TensorBoard supports arbitrary dimensional embeddings (it
                    # will project)
                    Z = mu.detach().cpu().numpy()  # [S, latent_dim]
                    writer.add_embedding(
                        mat=Z,
                        metadata=[str(i) for i in idx],
                        tag=f"latent_d{model.latent_dim}",
                        global_step=epoch,
                    )
                    # Flush to make the projector config visible immediately.
                    writer.flush()

            # Checkpoint best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    "model": model.state_dict(),
                    "d_in": d_in,
                    "hidden": hidden,
                    "beta": model.beta,
                    "recon_cosine_weight": recon_cosine_weight,
                    "latent_dim": model.latent_dim,
                    "dropout": dropout,
                    "recon_mse_mode": recon_mse_mode,
                    "recon_mse_manual_scale": recon_mse_manual_scale,
                    "free_bits": free_bits,
                    "capacity_max": capacity_max,
                    "capacity_warmup_frac": capacity_warmup_frac,
                    "capacity_gamma": capacity_gamma,
                    "kl_uniform_weight": kl_uniform_weight,
                    "ae_mode": ae_mode,
                    "mean": ds.mean,
                    "std": ds.std,
                }, ckpt_path)
                _write_config(
                    {"best_loss": best_loss, "best_epoch": epoch},
                    merge=True)
    except KeyboardInterrupt:
        # Mark aborted and re-raise to allow outer handling if desired
        _write_config({
            "status": "aborted",
            "aborted_epoch": epoch if 'epoch' in locals() else None,
            "global_step": global_step,
            "best_loss": best_loss
        }, merge=True)
        logger.warning("Training aborted by user (KeyboardInterrupt)")
        raise

    # Finalize manifest (merge) if not aborted
    end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_config({
        "end_time": end_time,
        "status": "finished",
        "final_global_step": global_step,
        "best_loss": best_loss,
        "epochs_completed": epochs,
        "device": device
    }, merge=True)

    writer.close()
    logger.info(
        f"Done. Best checkpoint: {ckpt_path}\nLaunch TensorBoard: tensorboard --logdir {os.path.dirname(tb_dir)}")


# -------------------------
# Inference
# -------------------------

class VAE3DWrapper:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.d_in = ckpt["d_in"]
        self.hidden = ckpt["hidden"]
        self.recon_cosine_weight = ckpt["recon_cosine_weight"]
        self.latent_dim = ckpt.get("latent_dim", 3)
        self.dropout = ckpt.get("dropout", 0.0)
        recon_mse_mode = ckpt.get("recon_mse_mode", "element_mean")
        recon_mse_manual_scale = ckpt.get("recon_mse_manual_scale", None)
        self.mean = torch.from_numpy(ckpt["mean"]).float()
        self.std = torch.from_numpy(ckpt["std"]).float()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        ae_mode = ckpt.get("ae_mode", False)
        self.model = VAE3D(
            self.d_in,
            self.hidden,
            beta=1.0,
            recon_cosine_weight=self.recon_cosine_weight,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            recon_mse_mode=recon_mse_mode,
            recon_mse_manual_scale=recon_mse_manual_scale,
            recon_log_extra=False,
            kl_uniform_weight=ckpt.get("kl_uniform_weight", 0.0),
            ae_mode=ae_mode).to(
            self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def to3d(self, X: np.ndarray, use_mu: bool = True,
             batch_size: int = 4096) -> np.ndarray:
        """Convert embeddings to 3D coordinates in batches to avoid memory issues."""
        n_samples = X.shape[0]
        Z_list = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = torch.tensor(
                X[i: end_idx],
                dtype=torch.float32).to(
                self.device)
            mu, logvar = self.model.encode(X_batch)
            Z_batch = mu if use_mu else self.model.reparameterize(mu, logvar)
            Z_list.append(Z_batch.detach().cpu().numpy())

        return np.concatenate(Z_list, axis=0)


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
    pt.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[
            768,
            384,
            128,
            64],
        help="Hidden layer sizes, e.g. --hidden 1024 512 256")
    pt.add_argument("--batch_size", type=int, default=1024 * 2)
    pt.add_argument("--epochs", type=int, default=25)
    pt.add_argument("--lr", type=float, default=1e-5)
    pt.add_argument("--beta", type=float, default=0.5)
    pt.add_argument("--beta_warmup_frac", type=float, default=0.25)
    pt.add_argument(
        "--lr_warmup_frac",
        type=float,
        default=0.5,
        help="Fraction of total optimizer steps to linearly warm up LR (defaults to beta_warmup_frac if not set).")
    pt.add_argument("--cos_w", type=float, default=0)
    pt.add_argument(
        "--recon_mse_mode",
        type=str,
        default="scale_by_dim",
        choices=[
            "element_mean",
            "scale_by_dim",
            "sum_per_sample",
            "disabled"],
        help="How to aggregate MSE: element_mean (default), scale_by_dim (multiply by feature dim), sum_per_sample (sum over features / batch), disabled (ignore MSE).")
    pt.add_argument(
        "--recon_mse_manual_scale",
        type=float,
        default=1.0,
        help="Optional extra multiplier applied after mode scaling (e.g. 2.0).")
    pt.add_argument("--dropout", type=float, default=0.2)
    pt.add_argument("--weight_decay", type=float, default=2e-5)
    pt.add_argument("--num_workers", type=int, default=4)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--log_every", type=int, default=50)
    pt.add_argument("--log_embed_every_epochs", type=int, default=3)
    pt.add_argument("--embed_sample_size", type=int, default=5000)
    pt.add_argument(
        "--no_mmap",
        action="store_true",
        help="Disable numpy memmap and load arrays eagerly. Useful if the dataset fits in RAM and random memmap access is slow on your storage.")
    pt.add_argument(
        "--latent_dim",
        type=int,
        default=3,
        help="Latent dimensionality (default 3). Embedding projector logs this latent space regardless of dimension.")
    # Regularisation (pick one strategy)
    pt.add_argument(
        "--free_bits",
        type=float,
        default=0.3,
        help="Per-dim minimum KL (nats). Set to 0.0 if using capacity objective. Typical 0.1–0.5.")
    pt.add_argument(
        "--capacity_max",
        type=float,
        default=0,
        help="Total KL capacity C_max in nats. Use with |KL - C(t)|; set to 0.0 if using free-bits. Typical 20–50.")
    pt.add_argument(
        "--capacity_warmup_frac",
        type=float,
        default=0.4,
        help="Fraction of total steps to linearly increase C(t) to capacity_max (0.4–0.6 common).")
    pt.add_argument(
        "--capacity_gamma",
        type=float,
        default=25.0,
        help="Gamma weight for capacity objective (|KL - C(t)|). 10–50 is typical; 25 is a good start.")
    pt.add_argument(
        "--kl_uniform_weight",
        type=float,
        default=0.01,
        help="Variance penalty over per-dimension KL (var of KL dims) to encourage multi-dim latent usage; try 0.01–0.1. 0 disables.")
    pt.add_argument(
        "--ae",
        action="store_true",
        help="Enable plain autoencoder mode (no KL term, deterministic latent). Overrides beta, free_bits, capacity_* and kl_uniform_weight to effectively 0 during optimization.")
    # No performance options needed - always use lazy mode without
    # standardization

    pi = sub.add_parser("infer")
    pi.add_argument("--batch_size", type=int, default=1024 * 8)
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
            data_paths=args.data, out_dir=args.out, hidden=args.hidden,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            beta=args.beta, beta_warmup_frac=args.beta_warmup_frac,
            lr_warmup_frac=(args.lr_warmup_frac
                            if args.lr_warmup_frac is
                            not None else args.beta_warmup_frac),
            recon_cosine_weight=args.cos_w, recon_mse_mode=args.recon_mse_mode,
            recon_mse_manual_scale=args.recon_mse_manual_scale,
            dropout=args.dropout, weight_decay=args.weight_decay,
            num_workers=args.num_workers, seed=args.seed,
            log_every=args.log_every,
            log_embed_every_epochs=args.log_embed_every_epochs,
            embed_sample_size=args.embed_sample_size, free_bits=args.free_bits,
            capacity_max=args.capacity_max,
            capacity_warmup_frac=args.capacity_warmup_frac,
            capacity_gamma=args.capacity_gamma, latent_dim=args.latent_dim,
            kl_uniform_weight=(0.0 if args.ae else args.kl_uniform_weight),
            ae_mode=args.ae,
            mmap=(not args.no_mmap))
    else:
        wrapper = VAE3DWrapper(args.ckpt)
        X = np.load(args.data).astype(np.float32)
        logger.info(
            f"Processing {X.shape[0]} samples in batches of {args.batch_size}")
        Z = wrapper.to3d(
            X, use_mu=(not args.sample),
            batch_size=args.batch_size)
        np.save(args.out, Z)
        logger.info(f"Saved coords to {args.out} | shape={Z.shape}")


'''
# 1) Train (light model)
python src/wc_simd/vlm_embed_vae.py train \
  --data data/vlm_embed/iiif_no_text_embedding_matrix.npy \
  --out runs/vlm_embed_vae3d_light_8 \
  --hidden 1024 512 256 \
  --epochs 40 --beta 4.0 --beta_warmup_frac 0.2 --cos_w 0.6

# 2) Open TensorBoard
tensorboard --logdir runs

# 3) Inference (map to 3D)
python src/wc_simd/vlm_embed_vae.py infer \
  --ckpt runs/vlm_embed_vae3d_light_3/vae3d.pt \
  --data data/vlm_embed/iiif_no_text_embedding_matrix.npy \
  --out data/vlm_embed/coords3d.npy
'''

if __name__ == "__main__":
    # Basic configuration only if root handlers not already set (so library
    # use is clean)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
