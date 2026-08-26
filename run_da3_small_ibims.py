from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


RGB_PATH = Path(
    "/home/user/Projects/Any2Full/datasets/ibims1/"
    "ibims1_core_raw/rgb/lectureroom_01.png"
)

OUT_DIR = Path(
    "/home/user/Projects/Any2Full/experiments/"
    "ibims_replication/da3_bridge"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))


with Image.open(RGB_PATH) as img:
    orig_w, orig_h = img.size

print("Original RGB size:", orig_h, orig_w)


print("Loading DA3-SMALL...")

model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-SMALL"
)

model = model.to(device)
model.eval()


print("Running inference...")

prediction = model.inference(
    image=[str(RGB_PATH)],
    process_res=504,
    process_res_method="upper_bound_resize",
)


depth = prediction.depth[0].astype(np.float32)

print("DA3 processed shape:", depth.shape)


depth_original = cv2.resize(
    depth,
    (orig_w, orig_h),
    interpolation=cv2.INTER_LINEAR,
)


processed_path = (
    OUT_DIR / "lectureroom_01_da3small_processed.npy"
)

final_path = (
    OUT_DIR / "lectureroom_01_da3small.npy"
)


np.save(processed_path, depth)

np.save(final_path, depth_original)


print()
print("Saved:", final_path)
print("Final shape:", depth_original.shape)
print("min:", float(depth_original.min()))
print("max:", float(depth_original.max()))
print("mean:", float(depth_original.mean()))
print("finite:", bool(np.isfinite(depth_original).all()))
