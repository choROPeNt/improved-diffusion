"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch 
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1

SETUP_RETRY_COUNT = 3


def setup_dist(allow_mps_single_process: bool = True):
    """
    Setup a distributed process group.

    - CUDA: NCCL (multi-GPU)
    - CPU:  GLOO
    - MPS:  single-process prototyping/sampling with GLOO (WORLD_SIZE=1)
            PYTORCH_ENABLE_MPS_FALLBACK=1 is enabled to allow CPU fallback
            for unsupported MPS ops.
    """
    import os
    import socket
    import torch
    import torch.distributed as dist
    from mpi4py import MPI

    if dist.is_initialized():
        return

    # Enable MPS CPU fallback for unsupported ops
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # ---- Local prototyping path (Apple Silicon / MPS) ----
    if torch.backends.mps.is_available() and allow_mps_single_process and not torch.cuda.is_available():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        dist.init_process_group(backend="gloo", init_method="env://")
        return

    # ---- HPC / MPI path ----
    comm = MPI.COMM_WORLD
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    hostname = "localhost" if backend == "gloo" else socket.gethostbyname(socket.getfqdn())

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Select the torch device.

    Priority:
      1) CUDA (HPC / NVIDIA)
      2) MPS  (Apple Silicon)
      3) CPU
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    import torch
    import torch.distributed as dist

    # No distributed -> nothing to sync
    if not (dist.is_available() and dist.is_initialized()):
        return

    # Single process -> nothing to sync
    if dist.get_world_size() == 1:
        return

    for p in params:
        dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
