from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, List, Tuple
from pathlib import Path
import bisect

import numpy as np
import h5py
from torch.utils.data import Dataset



@dataclass(frozen=True)
class PatchSpec:
    patch_wdh: Tuple[int, int, int]   # (W, D, H)
    stride_wdh: Tuple[int, int, int]  # (W, D, H)


class MultiH5PatchDataset(Dataset):
    """
    Loads 3D patches from one or more HDF5 files.

    Assumptions:
      - HDF5 dataset stored as (H, D, W)
      - patch/stride specified as (W, D, H)
      - only full patches (drops borders)

    Sharding:
      - shard in [0, num_shards-1]
      - dataset exposes only indices that belong to this shard:
            global_index = shard + k * num_shards

    Returns:
      - x: float32 array [C, D, H, W] (C=1 if add_channel=True else [D,H,W])
      - out dict with:
          - "start_wdh": int64 [3]
          - "file_idx": int64
          - optionally "y"
    """

    def __init__(
        self,
        h5_paths: Sequence[str],
        dset_key: str,
        spec: PatchSpec,
        *,
        classes: Optional[Sequence[int]] = None,
        add_channel: bool = True,
        shard: int = 0,
        num_shards: int = 1,
        cache_file_handles: bool = False,
        normalize: bool | None = True,
        clip_low: float | None = None,
        clip_high: float | None = None,
    ):
        super().__init__()

        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not (0 <= shard < num_shards):
            raise ValueError(f"shard must be in [0, {num_shards-1}], got {shard}")

        self.h5_paths = [str(p) for p in h5_paths]
        if len(self.h5_paths) == 0:
            raise ValueError("h5_paths is empty")

        self.dset_key = dset_key
        self.spec = spec
        self.add_channel = add_channel
        self.shard = shard
        self.num_shards = num_shards
        self.cache_file_handles = cache_file_handles
        self.normalize = normalize
        self.clip_low = clip_low
        self.clip_high = clip_high

        # Optional handle cache (per worker process) to speed repeated reads
        self._handles: Dict[int, h5py.File] = {}

        # --- compute per-file patch counts and prefix sums over the *global* index space ---
        self._file_shapes: List[Tuple[int, int, int]] = []
        self._file_patch_counts: List[int] = []
        self._file_prefix: List[int] = [0]  # length = nfiles+1

        pw, pd, ph = self.spec.patch_wdh
        sw, sd, sh = self.spec.stride_wdh

        for path in self.h5_paths:
            if not Path(path).is_file():
                raise ValueError(f"HDF5 file does not exist: {path}")

            with h5py.File(path, "r") as f:
                if self.dset_key not in f:
                    raise ValueError(f"Dataset key '{self.dset_key}' not found in {path}")
                H, D, W = f[self.dset_key].shape

            self._file_shapes.append((H, D, W))

            # number of valid patch start positions per axis (drop borders)
            nw = max(0, (W - pw) // sw + 1)
            nd = max(0, (D - pd) // sd + 1)
            nh = max(0, (H - ph) // sh + 1)

            count = nw * nd * nh
            self._file_patch_counts.append(count)
            self._file_prefix.append(self._file_prefix[-1] + count)

        self._global_count = self._file_prefix[-1]
        if self._global_count == 0:
            raise ValueError(
                "No patches available with given patch/stride. "
                "Check spatial_size/patch size vs volume dimensions."
            )

        # --- optional classes on the global (unsharded) patch index space ---
        self.classes = None
        if classes is not None:
            if len(classes) != self._global_count:
                raise ValueError(f"classes must have length {self._global_count}, got {len(classes)}")
            self.classes = np.asarray(classes, dtype=np.int64)

        # --- compute how many indices this shard will expose ---
        # indices owned by this shard: shard, shard+num_shards, ...
        self._shard_len = (self._global_count - self.shard + self.num_shards - 1) // self.num_shards
        # (integer ceil of (global_count - shard)/num_shards)

    def __len__(self) -> int:
        return self._shard_len

    # ---------- utilities ----------
    def _global_index_from_shard_index(self, i: int) -> int:
        # map 0..len-1 -> global index
        g = self.shard + i * self.num_shards
        if g >= self._global_count:
            raise IndexError
        return g

    def _locate_file_and_local(self, global_idx: int) -> Tuple[int, int]:
        # find file_idx such that prefix[file_idx] <= global_idx < prefix[file_idx+1]
        file_idx = bisect.bisect_right(self._file_prefix, global_idx) - 1
        local_idx = global_idx - self._file_prefix[file_idx]
        return file_idx, local_idx

    def _local_idx_to_start_wdh(self, file_idx: int, local_idx: int) -> Tuple[int, int, int]:
        # Convert local patch index -> (w0,d0,h0) for that file.
        H, D, W = self._file_shapes[file_idx]
        pw, pd, ph = self.spec.patch_wdh
        sw, sd, sh = self.spec.stride_wdh

        nw = max(0, (W - pw) // sw + 1)
        nd = max(0, (D - pd) // sd + 1)
        nh = max(0, (H - ph) // sh + 1)

        # ordering matches your original list-comprehension: for h in hs for d in ds for w in ws
        # i.e. w is fastest, then d, then h
        w_i = local_idx % nw
        d_i = (local_idx // nw) % nd
        h_i = local_idx // (nw * nd)

        if h_i >= nh:
            raise IndexError("local_idx out of range for computed patch grid")

        w0 = w_i * sw
        d0 = d_i * sd
        h0 = h_i * sh
        return (w0, d0, h0)

    def _get_handle(self, file_idx: int) -> h5py.File:
        # Optional caching. Note: each DataLoader worker is a separate process,
        # so handle caching is per worker and generally safe.
        if not self.cache_file_handles:
            return h5py.File(self.h5_paths[file_idx], "r")

        h = self._handles.get(file_idx)
        if h is None:
            h = h5py.File(self.h5_paths[file_idx], "r")
            self._handles[file_idx] = h
        return h

    def __del__(self):
        # close cached handles
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass

    # ---------- dataset interface ----------
    def __getitem__(self, i: int):
        global_idx = self._global_index_from_shard_index(i)
        file_idx, local_idx = self._locate_file_and_local(global_idx)
        w0, d0, h0 = self._local_idx_to_start_wdh(file_idx, local_idx)
        pw, pd, ph = self.spec.patch_wdh

        # Read patch: dataset stored (H, D, W) => slice [h, d, w]
        if self.cache_file_handles:
            f = self._get_handle(file_idx)
            vol = f[self.dset_key]
            patch = vol[h0 : h0 + ph, d0 : d0 + pd, w0 : w0 + pw]
        else:
            with self._get_handle(file_idx) as f:
                vol = f[self.dset_key]
                patch = vol[h0 : h0 + ph, d0 : d0 + pd, w0 : w0 + pw]

        x = np.asarray(patch, dtype=np.float32)     # [H, D, W] patch (h, d, w)
        # normalize to [-1, 1]
        if self.normalize:
            if self.clip_low is not None and self.clip_high is not None:
                lo = float(self.clip_low)
                hi = float(self.clip_high)

                # avoid divide-by-zero
                if hi > lo:
                    x = np.clip(x, lo, hi)
                    x = 2.0 * (x - lo) / (hi - lo) - 1.0
                else:
                    x = np.zeros_like(x, dtype=np.float32)

            else:
                # fallback: full uint16 range
                x = x / 32767.5 - 1.0

            
        x = np.transpose(x, (1, 0, 2))              # -> [D, H, W]

        if self.add_channel:
            x = x[None, ...]                        # -> [1, D, H, W]

        # out: Dict[str, Any] = {
        #     "start_wdh": np.array([w0, d0, h0], dtype=np.int64),
        #     "file_idx": np.array(file_idx, dtype=np.int64),
        # }
        out: Dict[str, Any] = {}
        if self.classes is not None:
            out["y"] = np.array(self.classes[global_idx], dtype=np.int64)

        return x, out
    

def estimate_u16_clip_bounds(
    h5_paths,
    dset_key: str,
    spec: PatchSpec,
    *,
    n_patches_per_file: int = 64,
    p_low: float = 1.0,
    p_high: float = 99.0,
    seed: int = 0,
):
    """
    Returns (lo, hi) based on sampling patches across all files.
    Uses percentiles over the sampled voxels.
    """
    rng = np.random.default_rng(seed)
    pw, pd, ph = spec.patch_wdh

    samples = []

    for path in h5_paths:
        path = str(path)
        if not Path(path).is_file():
            raise ValueError(f"Missing file: {path}")

        with h5py.File(path, "r") as f:
            vol = f[dset_key]
            H, D, W = vol.shape  # (H, D, W)

            # sample random patch starts
            max_h = H - ph
            max_d = D - pd
            max_w = W - pw
            if max_h < 0 or max_d < 0 or max_w < 0:
                raise ValueError(f"Patch bigger than volume in {path}: vol={vol.shape}, patch={(ph,pd,pw)}")

            for _ in range(n_patches_per_file):
                h0 = int(rng.integers(0, max_h + 1))
                d0 = int(rng.integers(0, max_d + 1))
                w0 = int(rng.integers(0, max_w + 1))
                patch = vol[h0:h0+ph, d0:d0+pd, w0:w0+pw]

                # sample a subset of voxels from the patch to keep memory small
                arr = np.asarray(patch, dtype=np.uint16).ravel()
                if arr.size > 200_000:
                    idx = rng.choice(arr.size, size=200_000, replace=False)
                    arr = arr[idx]
                samples.append(arr)

    all_vals = np.concatenate(samples).astype(np.float32)
    lo = float(np.percentile(all_vals, p_low))
    hi = float(np.percentile(all_vals, p_high))

    if hi <= lo:
        raise ValueError(f"Degenerate bounds: lo={lo}, hi={hi}")
    return lo, hi