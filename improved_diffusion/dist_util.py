"""
Helpers for distributed training.
"""

import io
import os
import socket
import datetime

import blobfile as bf

import torch
import torch.distributed as dist

# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1


# ----------------------------
# MPI helpers (lazy import!)
# ----------------------------
def _mpi_available() -> bool:
    """
    True if we are very likely running under an MPI/PMI launcher (mpiexec/srun).
    This prevents importing mpi4py in plain `python ...` runs, which can hang on HPC.
    """
    return any(
        k in os.environ
        for k in (
            "OMPI_COMM_WORLD_SIZE",  # Open MPI
            "OMPI_COMM_WORLD_RANK",
            "PMI_SIZE",              # PMI/PMIx (often via srun)
            "PMI_RANK",
            "PMIX_RANK",
            "SLURM_PROCID",          # Slurm rank
            "SLURM_NTASKS",
        )
    )


def _get_mpi():
    # Import only when needed
    from mpi4py import MPI
    return MPI


def setup_dist(
    allow_mps_single_process: bool = True,
    timeout_seconds: int = 120,
) -> None:
    """
    Setup a distributed process group.

    - CUDA: NCCL
    - CPU:  GLOO
    - MPS:  optional single-process "gloo" group for convenience

    IMPORTANT:
    - If you run plain `python script.py` on Capella, this function will NOT import
      mpi4py and will NOT initialize distributed (prevents the import hang).
    - If you run under `mpiexec/srun` (multi-rank), it will initialize via MPI.
    """
    if dist.is_available() and dist.is_initialized():
        return

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # ---- Local prototyping path (Apple Silicon / MPS) ----
    if (
        allow_mps_single_process
        and torch.backends.mps.is_available()
        and not torch.cuda.is_available()
    ):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        return

    # ---- Single-process non-MPI path (common on clusters) ----
    # If you didn't start with mpiexec/srun/torchrun, don't touch MPI.
    if not _mpi_available():
        return

    # ---- HPC / MPI path ----
    MPI = _get_mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Pick a master address/port on rank 0 and broadcast.
    if rank == 0:
        # For multi-node NCCL, a routable IP/hostname is needed.
        master_addr = os.environ.get("MASTER_ADDR")
        if not master_addr:
            master_addr = socket.gethostbyname(socket.getfqdn())

        master_port = int(os.environ.get("MASTER_PORT") or _find_free_port())
    else:
        master_addr = None
        master_port = None

    master_addr = comm.bcast(master_addr, root=0)
    master_port = comm.bcast(master_port, root=0)

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Bind CUDA device deterministically
    if torch.cuda.is_available():
        local_gpu = rank % GPUS_PER_NODE
        torch.cuda.set_device(local_gpu)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )


def dev() -> torch.device:
    """
    Select the torch device.

    Priority:
      1) CUDA
      2) MPS
      3) CPU
    """
    if torch.cuda.is_available():
        # Avoid importing mpi4py unless launched under MPI/PMI
        rank = 0
        if _mpi_available():
            try:
                MPI = _get_mpi()
                rank = MPI.COMM_WORLD.Get_rank()
            except Exception:
                rank = 0
        return torch.device(f"cuda:{rank % GPUS_PER_NODE}")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.

    - If MPI is available: rank0 reads + bcast bytes.
    - Otherwise: plain torch.load.
    """
    if _mpi_available():
        MPI = _get_mpi()
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
        else:
            data = None
        data = comm.bcast(data, root=0)
        return torch.load(io.BytesIO(data), **kwargs)

    # Non-MPI fallback
    with bf.BlobFile(path, "rb") as f:
        return torch.load(f, **kwargs)


def sync_params(params):
    """
    Broadcast params from rank 0 to all other ranks (no-op if not distributed).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_world_size() == 1:
        return
    for p in params:
        dist.broadcast(p, src=0)


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])
    finally:
        s.close()