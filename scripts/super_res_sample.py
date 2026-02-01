"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""
from typing import Any, Dict
import argparse
import os
import sys

import blobfile as bf
import numpy as np
import torch 
import torch.distributed as dist

import h5py

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.dir)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()



    logger.log("loading data...")

    if args.dataset_type == "image":
        data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    elif args.dataset_type == "hdf5":
        data = load_data_for_worker_h5(
            file_list=args.file_list,
            batch_size=args.batch_size,
            class_cond=args.class_cond,
            dataset_key=args.dataset_key,
        )
        ########
        ## some prototyping
        ########
        batch = next(data)
        def describe_tensor(name, t):
            t_cpu = t.detach().cpu()
            print(
                f"{name:10s} | "
                f"shape={tuple(t_cpu.shape)} "
                f"dtype={t_cpu.dtype} "
                f"min={t_cpu.min().item():.4f} "
                f"max={t_cpu.max().item():.4f}"
            )
        print("\nHDF5 batch preview")
        print("-" * 40)

        describe_tensor("low_res", batch["low_res"])

        if "y" in batch:
            describe_tensor("y", batch["y"])



    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        spatial = (args.spatial_size, ) * args.dims

        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, args.in_channel, *spatial),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        print(sample.shape)
        sys.exit()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def _read_file_list(file_list_path: str):
    with open(file_list_path, "r") as f:
        files = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not files:
        raise ValueError(f"Empty file_list: {file_list_path}")
    return files

def _ensure_nchw_or_ncdhw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert to channel-first:
      - 2D images:  [H,W,C] -> [C,H,W]
      - 3D volumes: [D,H,W] -> [1,D,H,W]
                 or already [C,D,H,W] stays.
    """
    if x.ndim == 2:
        # [H,W] -> [1,H,W]
        return x.unsqueeze(0)
    if x.ndim == 3:
        # Ambiguous: could be [H,W,C] or [D,H,W]
        # Heuristic: if last dim is small, treat as channels-last 2D.
        if x.shape[-1] in (1, 3, 4) and (x.shape[0] > 8 and x.shape[1] > 8):
            # [H,W,C] -> [C,H,W]
            return x.permute(2, 0, 1).contiguous()
        # else treat as [D,H,W] -> [1,D,H,W]
        return x.unsqueeze(0)
    if x.ndim == 4:
        # assume [C,D,H,W] already
        return x
    raise ValueError(f"Unsupported sample shape: {tuple(x.shape)}")


def load_data_for_worker_h5(
    file_list: str,
    batch_size: int,
    class_cond: bool,
    dataset_key: str = "patch_lr",
    label_key: str = "y",
    normalize: str = "uint16_to_-1_1",
    shuffle: bool = True,
    seed: int = 0,
):
    """
    Streaming HDF5 loader that yields dicts like:
      {"low_res": batch} or {"low_res": batch, "y": labels}

    DDP sharding:
      each rank reads files i = rank, rank+world, ...
    """
    files = _read_file_list(file_list)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        rank = 0
        world = 1

    rng = np.random.default_rng(seed + rank)

    buffer = []
    label_buffer = []

    # local epoch loop
    while True:
        order = np.arange(len(files))
        if shuffle:
            rng.shuffle(order)

        # shard by rank over the shuffled order
        for idx in order[rank::world]:
            path = files[int(idx)]

            with h5py.File(path, "r") as f:
                if dataset_key not in f:
                    raise KeyError(f"Missing dataset_key '{dataset_key}' in {path}")
                arr = f[dataset_key][()]  # numpy array

                # Optional label
                if class_cond:
                    if label_key in f:
                        y = f[label_key][()]
                    else:
                        # If you don't have labels, you can map from filename or coords here.
                        raise KeyError(f"class_cond=True but label_key '{label_key}' not found in {path}")

            x = torch.from_numpy(arr)

            # Convert to channel-first tensor
            x = _ensure_nchw_or_ncdhw(x).float()  # [C,H,W] or [C,D,H,W]

            # Normalization (match your old code)
            if normalize == "uint8_to_-1_1":
                # old: batch/127.5 - 1.0 assumes 0..255
                x = x / 127.5 - 1.0
            elif normalize == "uint16_to_-1_1":
                # for CT uint16 0..65535

                lo = float(np.percentile(x, 1.0))
                hi = float(np.percentile(x, 99.0))
                print(lo,hi)
                x = np.clip(x, lo, hi)
                x = 2.0 * (x - lo) / (hi - lo) - 1.0

                # x = x / 32767.5 - 1.0
            elif normalize == "none":
                pass
            else:
                raise ValueError(f"Unknown normalize mode: {normalize}")

            buffer.append(x)
            if class_cond:
                label_buffer.append(np.asarray(y))

            if len(buffer) == batch_size:
                batch = torch.stack(buffer, dim=0)  # [B,C,...]
                res = {"low_res": batch}

                if class_cond:
                    # labels -> tensor (choose long for class indices)
                    yb = torch.from_numpy(np.stack(label_buffer))
                    # if it's scalar class ids, enforce long:
                    if yb.ndim == 1 or (yb.ndim == 2 and yb.shape[1] == 1):
                        yb = yb.long().view(-1)
                    res["y"] = yb

                yield res
                buffer, label_buffer = [], []





def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []





def create_argparser():
    defaults: Dict[str, Any] = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        # Data
        data_dir=None,
        file_list=None,
        dataset_type=None,
        dataset_key="volume",
        dir=None,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
