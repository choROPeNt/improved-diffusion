from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image
import sys
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

from improved_diffusion.hdf5_dataset import MultiH5PatchDataset, PatchSpec

from pathlib import Path

import h5py




def load_data(
    *,
    data_dir: str | None = None,
    file_list: str | None = None,     # path to txt file
    dataset_file: str | None = None,  # for hdf5 etc (recommended)
    dataset_type: str | None = None,
    batch_size: int,
    spatial_size: int,
    class_cond: bool = False,
    deterministic: bool = False,
):
    """
    Create a generator over (images, kwargs) pairs.
    """

    if dataset_type not in {"image", "hdf5"}:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # ---------------------------
    # Resolve input(s) per dataset
    # ---------------------------
    if dataset_type == "image":
        print("#"*5)
        # Exactly one of data_dir or file_list
        if (data_dir is None) == (file_list is None):
            raise ValueError("For dataset_type='image' specify exactly one of: --data_dir or --file_list")

        if data_dir is not None:
            if not data_dir:
                raise ValueError("unspecified data directory")
            all_files = _list_image_files_recursively(data_dir)
        else:
            all_files = _read_file_list(file_list)

        if not all_files:
            raise ValueError("No image files found.")

        # Optional class labels
        classes = None
        if class_cond:
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        dataset = ImageDataset(
            spatial_size,
            all_files,
            classes=classes,
            shard=rank,
            num_shards=size,
        )

    elif dataset_type == "hdf5":
        # New: support multiple HDF5 files via MultiH5PatchDataset
        # Prefer --dataset_files (list) or allow --file_list (txt) containing many .h5 paths.
        # (You can keep --dataset_file as a single-file convenience.)

        # collect h5 paths
        h5_paths = None

        # If you still only have dataset_file as a single string argument:
        if dataset_file is not None:
            h5_paths = [dataset_file]

        # If user provided a list file of .h5 paths:
        elif file_list is not None:
            h5_paths = _read_file_list(file_list)  # returns list[str]
            if len(h5_paths) == 0:
                raise ValueError("For dataset_type='hdf5', --file_list is empty.")
        else:
            raise ValueError(
                "For dataset_type='hdf5' specify --dataset_file or --file_list "
                "(a txt file containing one or more .h5 paths)."
            )

        # validate existence
        missing = [p for p in h5_paths if not bf.exists(p)]
        if missing:
            raise ValueError(f"HDF5 file(s) do not exist: {missing}")

        # patch spec (cubic patches)
        spec = PatchSpec(
            patch_wdh=(spatial_size,) * 3,
            stride_wdh=(spatial_size // 2,) * 3,
        )

        dataset = MultiH5PatchDataset(
            h5_paths=h5_paths,
            dset_key="volume",
            spec=spec,
            shard=rank,
            num_shards=size,
            cache_file_handles=True,  # optional; set False if h5py gives trouble with workers
        )

    # ---------------------------
    # DataLoader
    # ---------------------------
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader




def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tiff",".h5"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def _read_file_list(file_list_path: str) -> list[str]:
    p = Path(file_list_path)
    if not p.is_file():
        raise ValueError(f"--file_list must be a file, got: {file_list_path}")

    files: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            files.append(s)

    if not files:
        raise ValueError(f"file_list is empty: {file_list_path}")
    return files

class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        w, h = pil_image.size
        while min(w, h) >= 2 * self.resolution:
            pil_image = pil_image.resize((w // 2, h // 2), resample=Image.Resampling.BOX)
            w, h = pil_image.size

        scale = self.resolution / min(w, h)
        pil_image = pil_image.resize(
            (round(w * scale), round(h * scale)), resample=Image.Resampling.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict



