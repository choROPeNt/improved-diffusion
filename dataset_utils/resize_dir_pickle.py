import os
from pathlib import Path
from PIL import Image
import numpy as np

from improved_diffusion.datasets_util import _list_image_files_recursively


def main():
    hr_dir = "data/hr"
    out_dir = "data/lr"
    out_file = os.path.join(out_dir, "lr_images.npz")

    os.makedirs(out_dir, exist_ok=True)

    files = _list_image_files_recursively(hr_dir)
    files.sort()  # IMPORTANT: deterministic ordering

    all_images = []

    for file in files:
        with Image.open(file) as img:
            img = img.convert("RGB")  # enforce consistent channels

            w, h = img.size
            if w % 4 != 0 or h % 4 != 0:
                raise ValueError(
                    f"Image size not divisible by 4: {file} ({w}x{h})"
                )

            img_lr = img.resize(
                (w // 4, h // 4),
                resample=Image.LANCZOS
            )

            all_images.append(np.asarray(img_lr, dtype=np.uint8))

    if len(all_images) == 0:
        raise RuntimeError("No images found to process.")

    all_images = np.stack(all_images, axis=0)  # (N, H, W, C)

    np.savez_compressed(out_file, arr_0=all_images)

    print(f"Saved {len(all_images)} LR images to {out_file}")


if __name__ == "__main__":
    main()