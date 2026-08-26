from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


RGB_DIRECTORY = Path(
    "/home/user/Projects/Any2Full/datasets/ibims1/"
    "ibims1_core_raw/rgb"
)

OUTPUT_DIRECTORY = Path(
    "/home/user/Projects/Any2Full/experiments/"
    "ibims_replication/da3_bridge_all"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


rgb_paths = sorted(
    RGB_DIRECTORY.glob("*.png")
)

if len(rgb_paths) != 100:
    raise RuntimeError(
        f"Expected 100 RGB images, found {len(rgb_paths)}."
    )


print("Loading DA3-SMALL once...")

model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-SMALL"
)

model = model.to(device)
model.eval()


completed = 0
skipped = 0


for index, rgb_path in enumerate(
    rgb_paths,
    start=1,
):
    output_path = (
        OUTPUT_DIRECTORY
        / f"{rgb_path.stem}_da3small.npy"
    )

    if output_path.exists():
        print(
            f"[{index:03d}/100] "
            f"Skipping existing: {rgb_path.stem}"
        )

        skipped += 1
        continue

    with Image.open(rgb_path) as image:
        original_width, original_height = image.size

    print(
        f"[{index:03d}/100] "
        f"Processing: {rgb_path.stem}"
    )

    with torch.inference_mode():
        prediction = model.inference(
            image=[str(rgb_path)],
            process_res=504,
            process_res_method="upper_bound_resize",
        )

    relative_depth = (
        prediction.depth[0]
        .astype(np.float32)
    )

    relative_depth_original = cv2.resize(
        relative_depth,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    if not np.isfinite(
        relative_depth_original
    ).all():
        raise RuntimeError(
            f"Non-finite DA3 prediction: {rgb_path.stem}"
        )

    temporary_path = output_path.with_suffix(
        ".tmp"
    )

    with temporary_path.open("wb") as handle:
        np.save(
            handle,
            relative_depth_original,
        )

    temporary_path.replace(
        output_path
    )

    print(
        "    shape=",
        relative_depth_original.shape,
        "min=",
        f"{relative_depth_original.min():.4f}",
        "max=",
        f"{relative_depth_original.max():.4f}",
    )

    completed += 1

    del prediction
    del relative_depth
    del relative_depth_original


print()
print("DA3 batch complete.")
print("New predictions:", completed)
print("Skipped predictions:", skipped)
print("Output:", OUTPUT_DIRECTORY)