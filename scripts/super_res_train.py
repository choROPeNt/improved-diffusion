"""
Train a super-resolution model.
"""

from __future__ import annotations
import sys

from typing import Optional, Sequence, Tuple, Any, Dict

import os

import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist

# from improved_diffusion import logger
from improved_diffusion import dist_util, logger
from improved_diffusion.datasets_util import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    _quarter
)


from improved_diffusion.train_util import TrainLoop

from torchinfo import summary

def main():
    
    
    args = create_argparser().parse_args()
    
    
    dist_util.setup_dist()

    logger.configure(dir = args.dir)

    log_rank_gpu_mapping(logger)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())


    # ddpm_torchinfo(
    #     model,
    #     image_size=(256, 256),
    #     low_res_size=(128, 128),
    #     channels=3,
    #     diffusion_steps=4000,
    # )

    logger.log("creating schedule sampler...")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    

    logger.log("creating data loader...")

    large_size = args.spatial_size
    small_size = _quarter(args.spatial_size)

    data = load_superres_data(
        data_dir=args.data_dir,
        file_list=args.file_list,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        large_size=large_size,
        small_size=small_size,
        class_cond=args.class_cond,
    )
    # sys.exit()
    # for batch, cond in data:
    #     print(batch.shape)
    #     print(cond.keys())
    #     print(cond["low_res"].shape)
    #     break
    
    logger.log(f"data loader created: {data}")
    logger.log(f"data type: {type(data)}")
    logger.log(f"dataset path: {args.data_dir}")
    logger.log("training...")
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_superres_data(
    *,
    data_dir: str | None = None,
    file_list: str | None = None,
    dataset_type: str | None  = None,
    batch_size: int,
    large_size: int,
    small_size: int,
    class_cond: bool = False,
    deterministic: bool = False,
):
    """
    Yields (large_batch, model_kwargs) where model_kwargs contains:
      - low_res: downsampled conditioning image
      - optionally y: class labels if class_cond=True
    """

    data = load_data(
        data_dir=data_dir,
        file_list=file_list,
        dataset_type=dataset_type,
        batch_size=batch_size,
        spatial_size=large_size,
        class_cond=class_cond,
        deterministic=deterministic,
    )

    for large_batch, model_kwargs in data:

        # Expect: [N, C, H, W] (2D) or [N, C, D, H, W] (3D)
        ndim = large_batch.ndim

        if ndim == 4:
            # 2D: NCHW
            low_res = F.interpolate(
                large_batch,
                size=(small_size, small_size),
                mode="area",
            )

        elif ndim == 5:
            # 3D: NCDHW
            low_res = F.interpolate(
                large_batch,
                size=(small_size, small_size, small_size),
                mode="area",
            )

        else:
            raise ValueError(
                f"Unsupported input shape {large_batch.shape}. "
                "Expected 4D (NCHW) or 5D (NCDHW)."
            )

        model_kwargs["low_res"] = low_res
        yield large_batch, model_kwargs


def _dist_status():
    if dist.is_available() and dist.is_initialized():
        return {
            "mode": "distributed",
            "backend": dist.get_backend(),
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
        }
    else:
        return {
            "mode": "single-process",
            "backend": None,
            "rank": 0,
            "world_size": 1,
        }



def create_argparser():
    defaults: Dict[str, Any] = dict(
        data_dir=None,       
        file_list=None,
        dataset_type=None, 
        dir=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def log_rank_gpu_mapping(logger, banner="Distributed setup"):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    host = os.uname().nodename
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if torch.cuda.is_available():
        local_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(local_idx)

        # props.pci_bus_id can appear as an int in some builds; format consistently
        pci = getattr(props, "pci_bus_id", None)
        if isinstance(pci, int):
            # best-effort: show both decimal and hex
            pci_str = f"{pci} (0x{pci:x})"
        else:
            pci_str = str(pci)

        msg = (
            f"[rank {rank}/{ws} | host {host}] "
            f"CVD={cvd} | torch_device=cuda:{local_idx} | "
            f"gpu={props.name} | pci_bus_id={pci_str}"
        )
    else:
        msg = f"[rank {rank}/{ws} | host {host}] CVD={cvd} | cuda=False"

    # Print from every rank (debugging). Log banner only on rank 0.
    if rank == 0:
        logger.log(banner)
    print(msg, flush=True)
    sys.stdout.flush()


def ddpm_torchinfo(
    model: torch.nn.Module,
    image_size: Tuple[int, int] = (256, 256),
    channels: int = 3,
    batch_size: int = 1,
    diffusion_steps: int = 1000,
    device: Optional[torch.device] = None,
    low_res_size: Optional[Tuple[int, int]] = None,
    depth: int = 3,
    verbose: int = 1,
    ):
    """
    Torchinfo summary for diffusion UNets.

    Supports:
      - standard DDPM: forward(x, timesteps, ...)
      - super-res DDPM: forward(x, timesteps, low_res, ...) or forward(x, timesteps, low_res=...)

    Parameters
    ----------
    model : nn.Module
        Your diffusion UNet / SuperResModel.
    image_size : (H, W)
        Spatial size for x (usually the training image_size).
    channels : int
        Channels for x (usually 3 or 1).
    batch_size : int
        Batch size for summary forward pass.
    diffusion_steps : int
        Max timestep range. We'll sample timesteps in [0, diffusion_steps).
    device : torch.device or None
        If None, uses model's device.
    low_res_size : (h, w) or None
        If provided, passes low_res conditioning tensor of this size.
    depth : int
        torchinfo depth.
    verbose : int
        torchinfo verbosity.

    Returns
    -------
    torchinfo.ModelStatistics or None
    """
    # Only run on rank 0 if DDP is active
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return None

    model.eval()

    # pick device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    H, W = image_size
    x = torch.randn(batch_size, channels, H, W, device=device)

    t = torch.randint(
        low=0,
        high=int(diffusion_steps),
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )

    # Try calling forward with (x, t, low_res) first if low_res_size is given.
    # If forward expects low_res as kwarg, we catch and retry.
    if low_res_size is not None:
        h, w = low_res_size
        low_res = torch.randn(batch_size, channels, h, w, device=device)

        try:
            return summary(
                model,
                input_data=(x, t, low_res),
                device=device.type,
                depth=depth,
                verbose=verbose,
            )
        except TypeError:
            # fallback: low_res passed as kwarg
            return summary(
                model,
                input_data=(x, t),
                kwargs={"low_res": low_res},
                device=device.type,
                depth=depth,
                verbose=verbose,
            )

    # Standard DDPM case
    return summary(
        model,
        input_data=(x, t),
        device=device.type,
        depth=depth,
        verbose=verbose,
    )



if __name__ == "__main__":

    main()

