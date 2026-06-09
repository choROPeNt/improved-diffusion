"""Train a single AbstractVAE configuration — HPC / H100 batch job.

Standalone port of notebooks/VAE_beta_experiments.py. Trains ONE (beta,
latent_channels) configuration and writes checkpoints + history.json to an
output directory. Run many configs as a SLURM job array (see
slurm/train_vae.sbatch) to reproduce the beta / latent-dim sweeps.

Held-out reconstruction (val_recon) is the metric to compare across runs; the
train/val split is stratified by phi (fiber volume fraction) so both shares the
same distribution.

Example
-------
    python scripts/vae_train.py \
        --data_path /data/.../patches_vae.h5 \
        --out_dir   /data/.../experiments/vae/lc8_b0.05 \
        --latent_channels 8 --beta 5e-2 \
        --total_steps 30000 --batch_size 64 --lr 2e-3
"""

import argparse
import functools
import json
import os
import time
from typing import cast

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from improved_diffusion.models.vae import AbstractVAE


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (h5-backed, lazy per-worker handle)
# ─────────────────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    """HDF5-backed binary/gray patch dataset. Returns (binary, gray, phi, src)."""

    def __init__(self, h5_path, class_filter=None, source_filter=None, transform=None):
        self.h5_path = str(h5_path)
        self.transform = transform
        self._file = None  # opened lazily per worker

        with h5py.File(self.h5_path, "r") as f:
            self.class_names = json.loads(cast(str, f.attrs["class_names"]))
            self.source_files = json.loads(cast(str, f.attrs["source_files"]))
            class_ids = np.asarray(f["class_id"])
            sources = np.asarray(f["source"]).astype(str)

        mask = np.ones(len(class_ids), dtype=bool)
        if class_filter is not None:
            keep = {self.class_names.index(c) for c in class_filter}
            mask &= np.isin(class_ids, list(keep))
        if source_filter is not None:
            mask &= np.isin(sources, source_filter)
        self._indices = np.where(mask)[0]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        idx = int(self._indices[i])

        f = cast(h5py.File, self._file)
        binary = torch.from_numpy(cast(h5py.Dataset, f["patches"])[idx].astype(np.float32)).unsqueeze(0)
        gray = torch.from_numpy(cast(h5py.Dataset, f["images"])[idx].astype(np.float32) / 255.0).unsqueeze(0)
        phi = torch.tensor(float(cast(h5py.Dataset, f["phi"])[idx]))
        s = cast(h5py.Dataset, f["source"])[idx]
        source = s.decode() if isinstance(s, bytes) else str(s)

        if self.transform is not None:
            binary = self.transform(binary)
            gray = self.transform(gray)
        return binary, gray, phi, source


def collate_downscale(batch, recon="mse"):
    """Stack binary patches, avg-pool 512->256, re-threshold.

    Target scaling matches the reconstruction loss:
      mse -> {-1, +1}  (Gaussian likelihood on a tanh-style output)
      bce -> { 0,  1}  (Bernoulli targets for binary_cross_entropy_with_logits)
    """
    binary_list, gray_list, phi_list, _ = zip(*batch)
    x = torch.stack(binary_list)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    x = (x > 0.5).float()                       # {0, 1}
    if recon == "mse":
        x = x * 2.0 - 1.0                       # -> {-1, 1}
    phi = torch.stack(phi_list)
    return x, phi


def stratified_split(ds, data_path, val_frac, n_bins, seed):
    """Disjoint train/val indices stratified by phi quantile bins."""
    with h5py.File(data_path, "r") as f:
        phi_all = np.asarray(f["phi"])[ds._indices].astype(np.float64)

    edges = np.quantile(phi_all, np.linspace(0, 1, n_bins + 1))
    bin_id = np.clip(np.digitize(phi_all, edges[1:-1]), 0, n_bins - 1)

    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(phi_all), dtype=bool)
    for b in range(n_bins):
        idx_b = np.where(bin_id == b)[0]
        rng.shuffle(idx_b)
        n_val_b = int(round(len(idx_b) * val_frac))
        val_mask[idx_b[:n_val_b]] = True

    val_idx = np.where(val_mask)[0].tolist()
    train_idx = np.where(~val_mask)[0].tolist()
    return train_idx, val_idx, phi_all


# ─────────────────────────────────────────────────────────────────────────────
# Model / loss
# ─────────────────────────────────────────────────────────────────────────────
def build_vae(args, device):
    return AbstractVAE(
        in_channels=args.in_channels if hasattr(args, "in_channels") else 1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        channel_mult=tuple(args.channel_mult),
        dims=args.dims if hasattr(args, "dims") else 2,
        attn_ds=tuple(args.attn_ds),
        attn_heads=args.attn_heads,
        spatial_latent=args.spatial_latent if hasattr(args, "spatial_latent") else True,
    ).to(device)


def recon_term(out, x, recon):
    """Reconstruction loss. `out` is logits (bce) or values in ~[-1,1] (mse)."""
    if recon == "bce":
        return F.binary_cross_entropy_with_logits(out, x, reduction="mean")
    return F.mse_loss(out, x, reduction="mean")


def vae_loss(x_hat, x, mu, logvar, beta, recon="mse"):
    rec = recon_term(x_hat, x, recon)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec, kl


def _binarize_target(x, recon):
    # threshold at the midpoint of each scale: bce {0,1}->0.5, mse {-1,1}->0
    return x > (0.5 if recon == "bce" else 0.0)


@torch.no_grad()
def eval_metrics(model, loader, device, recon):
    """Held-out metrics, decoding the deterministic latent mean mu (no sampling
    noise). Returns (recon_loss, iou, dice) — IoU/Dice on the thresholded
    reconstruction are scale-independent, so they compare across mse/bce.
    For both heads the prediction threshold is 0 (mse value>0; bce logit>0 <=>
    sigmoid>0.5)."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    inter = union = pred_sum = tgt_sum = 0
    for x, _phi in loader:
        x = x.to(device, non_blocking=True)
        mu, logvar = model.encode(x)
        out = model.decode(mu)
        total += recon_term(out, x, recon).item() * x.shape[0]
        n += x.shape[0]
        pred = out > 0.0
        tgt = _binarize_target(x, recon)
        i = (pred & tgt).sum().item()
        inter += i
        union += (pred | tgt).sum().item()
        pred_sum += pred.sum().item()
        tgt_sum += tgt.sum().item()
    if was_training:
        model.train()
    iou = inter / max(union, 1)
    dice = (2 * inter) / max(pred_sum + tgt_sum, 1)
    return total / max(n, 1), iou, dice


def cycle(dl):
    while True:
        yield from dl


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)

    # ── AMP: bf16 on H100 (no scaler), fp16 elsewhere on cuda, off on cpu ────
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    )
    scaler = GradScaler(
        device.type, enabled=use_amp and amp_dtype == torch.float16
    )
    if use_amp:
        print(f"[amp] enabled, dtype={amp_dtype}", flush=True)

    # ── data ────────────────────────────────────────────────────────────────
    ds = PatchDataset(args.data_path)
    train_idx, val_idx, phi_all = stratified_split(
        ds, args.data_path, args.val_frac, args.n_bins, args.seed
    )
    train_sub, val_sub = Subset(ds, train_idx), Subset(ds, val_idx)
    phi_tr, phi_va = phi_all[train_idx], phi_all[val_idx]
    print(
        f"[data] {len(ds)} patches | train {len(train_sub)} | val {len(val_sub)} "
        f"({len(train_sub)/len(ds):.0%}/{len(val_sub)/len(ds):.0%})",
        flush=True,
    )
    print(
        f"[phi]  train mean={phi_tr.mean():.4f} std={phi_tr.std():.4f} | "
        f"val mean={phi_va.mean():.4f} std={phi_va.std():.4f}",
        flush=True,
    )

    pin = device.type == "cuda"
    collate = functools.partial(collate_downscale, recon=args.recon)
    print(f"[recon] {args.recon}  (targets in "
          f"{'{0,1}' if args.recon == 'bce' else '{-1,1}'})", flush=True)
    train_loader = DataLoader(
        train_sub, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=args.num_workers > 0, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=args.num_workers > 0, collate_fn=collate,
    )

    # ── model / optim ─────────────────────────────────────────────────────────
    vae = build_vae(args, device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[model] latent_channels={args.latent_channels} params={n_params:,}", flush=True)
    opt = AdamW(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def beta_at(step):
        if args.beta_warmup_steps <= 0:
            return args.beta
        return args.beta * min(1.0, step / args.beta_warmup_steps)

    history = {k: [] for k in
               ("step", "loss", "recon", "val_recon", "val_iou", "val_dice",
                "kl", "beta", "sigma_mean")}
    
    data_iter = cycle(train_loader)

    vae.train()
    t0 = time.time()
    for step in range(args.total_steps + 1):
        x, _phi = next(data_iter)
        x = x.to(device, non_blocking=True)
        beta = beta_at(step)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            x_hat, mu, logvar, z = vae(x)
            loss, rec, kl = vae_loss(x_hat, x, mu, logvar, beta, recon=args.recon)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            opt.step()

        if step % args.log_interval == 0:
            sigma_mean = logvar.mul(0.5).exp().mean().item()
            val_rec, val_iou, val_dice = eval_metrics(vae, val_loader, device, args.recon)
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["recon"].append(rec.item())
            history["val_recon"].append(val_rec)
            history["val_iou"].append(val_iou)
            history["val_dice"].append(val_dice)
            history["kl"].append(kl.item())
            history["beta"].append(beta)
            history["sigma_mean"].append(sigma_mean)
            ips = (step + 1) * args.batch_size / (time.time() - t0)
            print(
                f"step {step:>6}/{args.total_steps}  loss={loss.item():.4f}  "
                f"recon={rec.item():.4f}  val_recon={val_rec:.4f}  "
                f"val_IoU={val_iou:.3f}  val_Dice={val_dice:.3f}  kl={kl.item():.3f}  "
                f"sig={sigma_mean:.3f}  beta={beta:.1e}  {ips:.0f} img/s",
                flush=True,
            )

        if args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
            save_ckpt(vae, args, history, step, n_params, final=False)

    save_ckpt(vae, args, history, args.total_steps, n_params, final=True)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    print(
        f"[done] final val_recon={history['val_recon'][-1]:.4f}  "
        f"val_IoU={history['val_iou'][-1]:.3f}  val_Dice={history['val_dice'][-1]:.3f}  "
        f"sigma={history['sigma_mean'][-1]:.3f}  ({time.time()-t0:.0f}s)",
        flush=True,
    )


def save_ckpt(vae, args, history, step, n_params, final):
    name = "model_final.pt" if final else f"model_{step:06d}.pt"
    path = os.path.join(args.out_dir, name)
    torch.save(
        {
            "state_dict": vae.state_dict(),
            "step": step,
            "params": n_params,
            "latent_channels": args.latent_channels,
            "beta": args.beta,
            "config": vars(args),
            "history": history,
        },
        path,
    )
    print(f"[ckpt] saved {path}", flush=True)


def load_config(path):
    """Read a YAML config and flatten its (cosmetic) sections into a single
    dict of argparse argument names -> values."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    flat = {}
    for key, val in raw.items():
        if isinstance(val, dict):       # section -> merge its keys
            flat.update(val)
        else:                            # top-level scalar -> keep as-is
            flat[key] = val
    return flat


def parse_args():
    # First resolve --config so it can supply argparse defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c","--config", default=None,
                     help="YAML config (configs/vae_train.yaml); CLI overrides it")
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config) if pre_args.config else {}

    p = argparse.ArgumentParser(
        parents=[pre], description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # io  (not required= : may be supplied by --config; validated below)
    p.add_argument("--data_path", default=None, help="path to patches_vae.h5")
    p.add_argument("--out_dir", default=None, help="run output directory")
    # split
    p.add_argument("--val_frac", type=float, default=0.25)
    p.add_argument("--n_bins", type=int, default=10, help="phi strata for the split")
    # model
    p.add_argument("--latent_channels", type=int, default=8)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 4])
    p.add_argument("--attn_ds", type=int, nargs="+", default=[4, 8])
    p.add_argument("--attn_heads", type=int, default=1)
    # objective
    p.add_argument("--recon", choices=["mse", "bce"], default="mse",
                   help="reconstruction loss; bce uses {0,1} targets + logits, "
                        "mse uses {-1,1}")
    p.add_argument("--beta", type=float, default=5e-2, help="max KL weight")
    p.add_argument("--beta_warmup_steps", type=int, default=1000)
    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--total_steps", type=int, default=30000)
    # runtime
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=0,
                   help="checkpoint every N steps (0 = only final)")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    p.add_argument("--seed", type=int, default=0)

    # config supplies defaults; explicit CLI flags still win
    p.set_defaults(**cfg)
    args = p.parse_args()

    missing = [k for k in ("data_path", "out_dir") if getattr(args, k) is None]
    if missing:
        p.error(f"missing required {missing} — set via --config or CLI")
    return args


if __name__ == "__main__":
    main()
