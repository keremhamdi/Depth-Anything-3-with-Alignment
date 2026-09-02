#!/usr/bin/env python3
"""NYU-Depth V2 one-line metric-depth benchmark.

This script implements the fixed comparison requested for the indoor project:

* official NYU-Depth V2 654-image test split at 304 x 228;
* two explicit one-line protocols: a paper-style ideal row and an empirical
  RPLidar pattern copied from the project's unsplatted real projections;
* DA3-SMALL, one-line median metric scaling, then Poisson refinement;
* Any2Full predictions produced by the repository's existing run_any2full.py;
* dense metric evaluation over the complete valid image and outside the line;
* multiple automatically selected, locally planar surface probes per image;
* shared-scale GT/DA3/Any2Full panels and paired scene-level bootstrap tests.

The script never aligns a prediction to dense ground truth.  Dense NYU depth is
used only to construct the sparse sensor input and to score the final outputs.
This is evaluation, not training.

Typical workflow
----------------
Prepare the official test split::

    python nyu_1line_metric_benchmark.py prepare \
      --nyu-mat /path/to/nyu_depth_v2_labeled.mat \
      --splits-mat /path/to/splits.mat \
      --data-root datasets/nyu_empirical_rplidar \
      --line-protocol empirical_rplidar \
      --real-template-dir /path/to/prepared/depth_full_points

Run DA3, median metric scaling, and Poisson refinement inside the DA3 environment::

    python nyu_1line_metric_benchmark.py infer-da3 \
      --data-root datasets/nyu_1line_rplidar43 \
      --output-dir experiments/nyu_1line/da3_median \
      --device cuda

Run Any2Full inside the Any2Full environment (the evaluator prints the same
command in the README shipped with this script), then evaluate::

    python nyu_1line_metric_benchmark.py evaluate \
      --data-root /absolute/path/to/datasets/nyu_1line_rplidar43 \
      --da3-dir /absolute/path/to/experiments/nyu_1line/da3_median \
      --any2full-dir /absolute/path/to/experiments/nyu_1line/any2full \
      --output-dir /absolute/path/to/experiments/nyu_1line/evaluation
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image


DEFAULT_WIDTH = 304
DEFAULT_HEIGHT = 228
DEFAULT_LINE_ROW_FRAC = 0.465
DEFAULT_LINE_POINTS = 43
DEFAULT_SPLAT_RADIUS = 0
DEFAULT_MAX_DEPTH_M = 10.0
DEFAULT_OUTSIDE_MARGIN_PX = 10
LINE_PROTOCOLS = ("paper_row", "empirical_rplidar")
BENCHMARK_VERSION = "2.2-poisson-safe-single-pixel-support"
METHOD_LABELS = {
    "da3_median": "DA3-SMALL + median",
    "da3_median_poisson": "DA3-SMALL + median + Poisson",
    "any2full": "Any2Full-vits",
}
PRIMARY_METHODS = ("da3_median_poisson", "any2full")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Extract the official NYU test split and make one-line inputs")
    prepare.add_argument("--nyu-mat", type=Path, required=True)
    prepare.add_argument("--splits-mat", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--width", type=positive_int, default=DEFAULT_WIDTH)
    prepare.add_argument("--height", type=positive_int, default=DEFAULT_HEIGHT)
    prepare.add_argument("--line-row-frac", type=unit_interval, default=DEFAULT_LINE_ROW_FRAC)
    prepare.add_argument("--line-points", type=positive_int, default=DEFAULT_LINE_POINTS)
    prepare.add_argument(
        "--splat-radius",
        type=int,
        default=None,
        help="Optional input replication radius; default 0 for both protocols (one pixel per return)",
    )
    prepare.add_argument(
        "--line-protocol",
        choices=LINE_PROTOCOLS,
        default="paper_row",
        help=(
            "paper_row uses evenly spaced one-line centers; empirical_rplidar transfers "
            "normalized positions from real unsplatted depth_full_points maps"
        ),
    )
    prepare.add_argument(
        "--real-template-dir",
        type=Path,
        default=None,
        help="Directory containing the real unsplatted depth_full_points/*.npy templates",
    )
    prepare.add_argument(
        "--template-seed",
        type=int,
        default=20260901,
        help="Deterministic seed used to assign real templates to NYU scenes",
    )
    prepare.add_argument("--max-depth-m", type=float, default=DEFAULT_MAX_DEPTH_M)
    prepare.add_argument("--limit", type=positive_int, default=None)
    prepare.add_argument("--overwrite", action="store_true")

    infer = sub.add_parser(
        "infer-da3",
        help="Run DA3-SMALL, one-line median metric calibration, and Poisson refinement",
    )
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--checkpoint", default="depth-anything/DA3-SMALL")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--process-res", type=positive_int, default=504)
    infer.add_argument("--limit", type=positive_int, default=None)
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument(
        "--include-poisson",
        action="store_true",
        help="Compatibility flag; Poisson is already enabled by default for the primary method.",
    )
    infer.add_argument(
        "--da3-root",
        type=Path,
        default=Path.cwd(),
        help="DA3 repository root containing the validated Poisson implementation.",
    )
    infer.add_argument("--poisson-rtol", type=float, default=1e-6)
    infer.add_argument("--poisson-maxiter", type=positive_int, default=5000)

    evaluate = sub.add_parser("evaluate", help="Compare dense metric predictions and create every-image panels")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--da3-dir", type=Path, required=True)
    evaluate.add_argument("--any2full-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--da3-poisson-dir", type=Path, default=None)
    evaluate.add_argument("--outside-margin-px", type=int, default=DEFAULT_OUTSIDE_MARGIN_PX)
    evaluate.add_argument("--plot-max-depth-m", type=float, default=DEFAULT_MAX_DEPTH_M)
    evaluate.add_argument("--plot-error-max-m", type=float, default=1.0)
    evaluate.add_argument("--surface-probes", type=positive_int, default=6)
    evaluate.add_argument("--surface-span-max-m", type=float, default=0.18)
    evaluate.add_argument("--bootstrap-samples", type=positive_int, default=10000)
    evaluate.add_argument("--seed", type=int, default=12345)
    evaluate.add_argument("--limit", type=positive_int, default=None)
    evaluate.add_argument("--skip-panels", action="store_true")

    return parser.parse_args()


def ensure_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def ensure_dir(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def save_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV columns for empty row set: {path}")
    names = list(fieldnames or rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_manifest(data_root: Path, limit: int | None = None) -> list[dict[str, str]]:
    path = data_root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run the prepare stage first; missing {path}")
    rows = read_csv(path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No scenes in {path}")
    return rows


def load_npy_2d(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.squeeze(np.load(path)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array at {path}; got {array.shape}")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"Shape mismatch at {path}: {array.shape} != {expected_shape}")
    return array


def resize_rgb(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def resize_depth_nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def read_test_indices(splits_mat: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat

        payload = loadmat(splits_mat)
        candidates = ["testNdxs", "testIdxs", "test_indices", "test"]
        for name in candidates:
            if name in payload:
                values = np.asarray(payload[name]).reshape(-1).astype(np.int64)
                if values.size:
                    # Official Matlab indices are one-based.
                    if values.min() >= 1:
                        values = values - 1
                    return values
    except NotImplementedError:
        pass

    try:
        import h5py

        with h5py.File(splits_mat, "r") as handle:
            for name in ("testNdxs", "testIdxs", "test_indices", "test"):
                if name in handle:
                    values = np.asarray(handle[name]).reshape(-1).astype(np.int64)
                    if values.size and values.min() >= 1:
                        values = values - 1
                    return values
    except Exception as error:
        raise RuntimeError(f"Could not read test indices from {splits_mat}: {error}") from error

    raise KeyError(f"No test-index variable found in {splits_mat}")


def matlab_hdf5_image(dataset: Any, index: int) -> np.ndarray:
    array = np.asarray(dataset[index])
    if array.ndim != 3:
        raise ValueError(f"Unexpected NYU image slice: {array.shape}")
    if array.shape[0] == 3:
        array = np.transpose(array, (2, 1, 0))
    elif array.shape[-1] == 3:
        if array.shape[0] > array.shape[1]:
            array = np.transpose(array, (1, 0, 2))
    else:
        raise ValueError(f"Cannot identify RGB channel axis in {array.shape}")
    return np.asarray(array, dtype=np.uint8)


def matlab_hdf5_depth(dataset: Any, index: int) -> np.ndarray:
    array = np.asarray(dataset[index], dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Unexpected NYU depth slice: {array.shape}")
    # Matlab v7.3 reverses H and W in the HDF5 representation.
    if array.shape[0] > array.shape[1]:
        array = array.T
    return array.astype(np.float32, copy=False)


def splat_centers(
    gt: np.ndarray,
    valid: np.ndarray,
    centers_yx: np.ndarray,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Sample GT at independent centers, then optionally replicate for network input."""
    if splat_radius < 0:
        raise ValueError("--splat-radius cannot be negative")
    height, width = gt.shape
    exact = np.zeros_like(gt, dtype=np.float32)
    kept: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_y, raw_x in np.asarray(centers_yx, dtype=np.int32):
        y = int(np.clip(raw_y, 0, height - 1))
        x = int(np.clip(raw_x, 0, width - 1))
        if (y, x) in seen or not valid[y, x]:
            continue
        seen.add((y, x))
        exact[y, x] = gt[y, x]
        kept.append((y, x))
    if len(kept) < 8:
        raise RuntimeError(f"Only {len(kept)} valid independent line measurements remain")

    sparse_input = np.zeros_like(gt, dtype=np.float32)
    for y, x in kept:
        value = float(gt[y, x])
        y0, y1 = max(0, y - splat_radius), min(height, y + splat_radius + 1)
        x0, x1 = max(0, x - splat_radius), min(width, x + splat_radius + 1)
        # Replication is input preprocessing, never additional sensor evidence.
        sparse_input[y0:y1, x0:x1] = value
    kept_array = np.asarray(kept, dtype=np.int32)
    line_row = int(np.median(kept_array[:, 0]))
    return exact, sparse_input, line_row, kept_array


def make_paper_row(
    gt: np.ndarray,
    valid: np.ndarray,
    row_fraction: float,
    point_count: int,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    height, width = gt.shape
    row = int(round(row_fraction * (height - 1)))
    edge_margin = max(2, splat_radius + 1)
    columns = np.rint(np.linspace(edge_margin, width - 1 - edge_margin, point_count)).astype(int)
    columns = np.unique(columns)
    centers = np.column_stack((np.full(columns.shape, row, dtype=np.int32), columns))
    return splat_centers(gt, valid, centers, splat_radius)


def load_empirical_templates(template_dir: Path) -> list[tuple[Path, np.ndarray]]:
    directory = ensure_dir(template_dir, "real unsplatted template directory")
    paths = sorted(directory.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy templates found in {directory}")
    templates: list[tuple[Path, np.ndarray]] = []
    for path in paths:
        array = load_npy_2d(path)
        centers = np.argwhere(np.isfinite(array) & (array > 0)).astype(np.int32)
        if len(centers) < 8:
            raise RuntimeError(f"Template has fewer than 8 independent points: {path}")
        templates.append((path, centers))
    return templates


def make_empirical_rplidar(
    gt: np.ndarray,
    valid: np.ndarray,
    template_path: Path,
    template_centers_yx: np.ndarray,
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Transfer only normalized real projected positions; NYU GT supplies metric ranges."""
    source = load_npy_2d(template_path)
    source_h, source_w = source.shape
    target_h, target_w = gt.shape
    centers = np.asarray(template_centers_yx, dtype=np.float64).copy()
    centers[:, 0] = np.rint(centers[:, 0] * (target_h - 1) / max(source_h - 1, 1))
    centers[:, 1] = np.rint(centers[:, 1] * (target_w - 1) / max(source_w - 1, 1))
    return splat_centers(gt, valid, centers.astype(np.int32), splat_radius)


def prepare_dataset(args: argparse.Namespace) -> None:
    nyu_mat = ensure_file(args.nyu_mat, "NYU labeled dataset")
    splits_mat = ensure_file(args.splits_mat, "NYU official split file")
    data_root = args.data_root.expanduser().resolve()
    if args.max_depth_m <= 0:
        raise ValueError("--max-depth-m must be positive")
    splat_radius = args.splat_radius
    if splat_radius is None:
        splat_radius = DEFAULT_SPLAT_RADIUS
    if splat_radius < 0:
        raise ValueError("--splat-radius cannot be negative")

    empirical_templates: list[tuple[Path, np.ndarray]] = []
    template_schedule: np.ndarray | None = None
    if args.line_protocol == "empirical_rplidar":
        if args.real_template_dir is None:
            raise ValueError("empirical_rplidar requires --real-template-dir .../prepared/depth_full_points")
        empirical_templates = load_empirical_templates(args.real_template_dir)
        template_schedule = np.random.default_rng(args.template_seed).permutation(len(empirical_templates))

    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("The prepare stage requires h5py: pip install h5py scipy") from error

    indices = read_test_indices(splits_mat)
    if args.limit is not None:
        indices = indices[: args.limit]
    if indices.size == 0:
        raise RuntimeError("The official test split is empty")

    directories = {
        "rgb": data_root / "rgb",
        "gt": data_root / "gt_depth_m",
        "exact": data_root / "sparse_exact_m",
        "input": data_root / "sparse_input_m",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    with h5py.File(nyu_mat, "r") as handle:
        if "images" not in handle or "depths" not in handle:
            raise KeyError(f"{nyu_mat} must contain images and depths")
        images = handle["images"]
        depths = handle["depths"]
        sample_count = int(images.shape[0])

        for order, source_index in enumerate(indices, start=1):
            source_index = int(source_index)
            if not 0 <= source_index < sample_count:
                raise IndexError(f"Split index {source_index} is outside 0..{sample_count - 1}")
            stem = f"nyu_test_{order - 1:04d}__src_{source_index + 1:04d}"
            paths = {
                "rgb": directories["rgb"] / f"{stem}.png",
                "gt": directories["gt"] / f"{stem}.npy",
                "exact": directories["exact"] / f"{stem}.npy",
                "input": directories["input"] / f"{stem}.npy",
            }
            template_name = ""
            selected_template: tuple[Path, np.ndarray] | None = None
            if empirical_templates:
                assert template_schedule is not None
                template_index = int(template_schedule[(order - 1) % len(template_schedule)])
                selected_template = empirical_templates[template_index]
                template_name = selected_template[0].name
            if all(path.is_file() for path in paths.values()) and not args.overwrite:
                gt = load_npy_2d(paths["gt"])
                exact = load_npy_2d(paths["exact"], gt.shape)
                sparse_input = load_npy_2d(paths["input"], gt.shape)
                rows = np.where(exact > 0)[0]
                line_row = int(np.median(rows)) if rows.size else -1
                exact_count = int(np.count_nonzero(exact > 0))
                input_count = int(np.count_nonzero(sparse_input > 0))
                status = "cached"
            else:
                rgb = resize_rgb(matlab_hdf5_image(images, source_index), args.width, args.height)
                gt = resize_depth_nearest(matlab_hdf5_depth(depths, source_index), args.width, args.height)
                valid = np.isfinite(gt) & (gt > 0) & (gt <= args.max_depth_m)
                gt = np.where(valid, gt, 0.0).astype(np.float32)
                if args.line_protocol == "paper_row":
                    exact, sparse_input, line_row, _ = make_paper_row(
                        gt, valid, args.line_row_frac, args.line_points, splat_radius
                    )
                else:
                    assert selected_template is not None
                    exact, sparse_input, line_row, _ = make_empirical_rplidar(
                        gt, valid, selected_template[0], selected_template[1], splat_radius
                    )
                Image.fromarray(rgb, mode="RGB").save(paths["rgb"])
                np.save(paths["gt"], gt)
                np.save(paths["exact"], exact)
                np.save(paths["input"], sparse_input)
                exact_count = int(np.count_nonzero(exact > 0))
                input_count = int(np.count_nonzero(sparse_input > 0))
                status = "saved"

            manifest.append(
                {
                    "scene": stem,
                    "test_order_1based": order,
                    "source_index_1based": source_index + 1,
                    "height": int(gt.shape[0]),
                    "width": int(gt.shape[1]),
                    "line_row": line_row,
                    "line_row_fraction": args.line_row_frac,
                    "line_protocol": args.line_protocol,
                    "real_template": template_name,
                    "independent_measurements": exact_count,
                    "sparse_input_pixels": input_count,
                    "splat_radius": splat_radius,
                    "max_depth_m": args.max_depth_m,
                }
            )
            print(
                f"[{order:3d}/{len(indices)}] {stem} {status}; "
                f"raw points={exact_count}, input pixels={input_count}",
                flush=True,
            )

    save_csv(data_root / "manifest.csv", manifest)
    protocol = {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": "NYU-Depth V2",
        "split": "official testNdxs, expected 654 images when --limit is omitted",
        "resolution": [args.width, args.height],
        "ground_truth": "NYU in-painted depths, camera-forward metric depth in metres",
        "line_protocol": args.line_protocol,
        "line_row_fraction": args.line_row_frac,
        "independent_line_measurements_requested": args.line_points,
        "splat_radius_pixels": splat_radius,
        "sparse_input_rule": "same sparse_input_m float32 NPY is supplied to DA3 alignment and Any2Full",
        "dense_gt_use": "construct sparse input and evaluate only; never used for post-inference alignment",
        "max_valid_depth_m": args.max_depth_m,
        "paper_row_definition": "evenly spaced one-pixel centers on one horizontal row; NYU cannot supply a physical multi-ring LiDAR ring",
        "empirical_rplidar_definition": "normalized (u,v) positions transferred from unsplatted real depth_full_points maps; real depth values are never transferred",
        "real_template_dir": str(args.real_template_dir.expanduser().resolve()) if args.real_template_dir else None,
        "real_template_count": len(empirical_templates),
        "template_seed": args.template_seed,
        "why_splat": "primary benchmark uses radius 0 (one pixel per return); nonzero radius is optional preprocessing ablation only",
    }
    (data_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"\nPrepared {len(manifest)} NYU test images at {data_root}")


def resize_prediction(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if array.shape == target_shape:
        return array.astype(np.float32, copy=False)
    height, width = target_shape
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def extract_da3_depth(prediction: Any) -> np.ndarray:
    depth = prediction.depth if hasattr(prediction, "depth") else prediction["depth"]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    array = np.asarray(depth, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"DA3 returned unexpected depth shape {np.asarray(depth).shape}")
    return array


def median_align(relative: np.ndarray, sparse: np.ndarray) -> tuple[np.ndarray, float]:
    anchors = np.isfinite(sparse) & (sparse > 0) & np.isfinite(relative) & (relative > 0)
    ratios = sparse[anchors] / relative[anchors]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 8:
        raise RuntimeError(f"Only {ratios.size} usable values for median alignment")
    scale = float(np.median(ratios))
    prediction = relative.astype(np.float32) * scale
    if not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise RuntimeError("DA3 median alignment produced non-positive or non-finite values")
    return prediction.astype(np.float32), scale


def load_validated_poisson(da3_root: Path) -> Callable[..., Any]:
    path = da3_root / "experiments/lidar_alignment/ibims/compare_median_poisson_oasis_100.py"
    if not path.is_file():
        raise FileNotFoundError(f"Validated Poisson source missing: {path}")
    spec = importlib.util.spec_from_file_location("validated_ibims_poisson", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "existing_poisson", None)
    if not callable(function):
        raise AttributeError(f"{path} has no callable existing_poisson")
    print(f"Secondary ablation uses existing_poisson{inspect.signature(function)} from {path}")
    return function


def call_poisson(
    function: Callable[..., Any],
    base: np.ndarray,
    sparse: np.ndarray,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    anchors = np.isfinite(sparse) & (sparse > 0)
    aliases = {
        "base": base,
        "base_depth": base,
        "depth": base,
        "initial": base,
        "initial_depth": base,
        "aligned": base,
        "aligned_depth": base,
        "prediction": base,
        "pred": base,
        "sparse": sparse,
        "sparse_depth": sparse,
        "lidar": sparse,
        "lidar_depth": sparse,
        "metric_depth": sparse,
        "anchors": anchors,
        "anchor_mask": anchors,
        "sparse_mask": anchors,
        "valid_mask": anchors,
        "rtol": rtol,
        "tol": rtol,
        "maxiter": maxiter,
        "max_iter": maxiter,
    }
    signature = inspect.signature(function)
    kwargs: dict[str, Any] = {}
    unknown: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            unknown.append(name)
    result = function(base, sparse, anchors, rtol, maxiter) if unknown else function(**kwargs)
    prediction, diagnostics = result if isinstance(result, tuple) else (result, {})
    array = np.squeeze(np.asarray(prediction, dtype=np.float32))
    if array.shape != base.shape:
        raise ValueError(f"Poisson returned {array.shape}; expected {base.shape}")
    diagnostic_payload = (
        dict(diagnostics) if isinstance(diagnostics, dict) else {"value": diagnostics}
    )
    invalid = ~np.isfinite(array) | (array <= 0)
    invalid_count = int(np.count_nonzero(invalid))
    if invalid_count:
        # Metric camera-Z depth must be finite and positive.  A sparse Poisson
        # solve can overshoot below zero at isolated pixels even when its
        # otherwise-valid solution converged.  Repair only those invalid
        # pixels with the already-valid median-scaled prior; leave every valid
        # Poisson pixel untouched and record the intervention transparently.
        array = array.copy()
        array[invalid] = base[invalid]
    if not np.isfinite(array).all() or not (array > 0).all():
        raise RuntimeError("Poisson repair did not produce a valid metric-depth map")
    diagnostic_payload.update(
        {
            "invalid_pixels_repaired_from_median": invalid_count,
            "invalid_pixel_repair_pct": 100.0 * invalid_count / float(array.size),
            "invalid_pixel_repair_strategy": "same-pixel_da3_median_prior",
        }
    )
    return array, diagnostic_payload


def infer_da3(args: argparse.Namespace) -> None:
    data_root = ensure_dir(args.data_root, "prepared NYU data root")
    output_dir = args.output_dir.expanduser().resolve()
    relative_dir = output_dir / "relative"
    median_dir = output_dir / "metric_m"
    poisson_dir = output_dir / "metric_m_poisson"
    for directory in (relative_dir, median_dir):
        directory.mkdir(parents=True, exist_ok=True)
    poisson_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(data_root, args.limit)
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise RuntimeError("Run infer-da3 inside the DA3 conda environment") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    poisson = load_validated_poisson(args.da3_root.expanduser().resolve())
    print(f"Loading {args.checkpoint} on {args.device}", flush=True)
    model = DepthAnything3.from_pretrained(args.checkpoint).to(args.device)
    model.eval()
    fit_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        stem = row["scene"]
        gt = load_npy_2d(data_root / "gt_depth_m" / f"{stem}.npy")
        sparse = load_npy_2d(data_root / "sparse_input_m" / f"{stem}.npy", gt.shape)
        rgb_path = data_root / "rgb" / f"{stem}.png"
        relative_path = relative_dir / f"{stem}.npy"
        median_path = median_dir / f"{stem}.npy"
        poisson_path = poisson_dir / f"{stem}.npy"

        if relative_path.is_file() and not args.overwrite:
            relative = load_npy_2d(relative_path)
            relative = resize_prediction(relative, gt.shape)
            inference_status = "cached"
        else:
            with torch.inference_mode():
                result = model.inference(image=[str(rgb_path)], process_res=args.process_res)
            relative = resize_prediction(extract_da3_depth(result), gt.shape)
            valid_relative = np.isfinite(relative) & (relative > 0)
            if not valid_relative.all():
                if not valid_relative.any():
                    raise RuntimeError(f"DA3 produced no valid depth for {stem}")
                relative = np.where(valid_relative, relative, np.median(relative[valid_relative]))
            np.save(relative_path, relative.astype(np.float32))
            inference_status = "inferred"

        median, scale = median_align(relative, sparse)
        np.save(median_path, median)
        poisson_diag: dict[str, Any] = {}
        poisson_status = "disabled"
        poisson_full_median_fallback = False
        if poisson is not None:
            try:
                refined, poisson_diag = call_poisson(
                    poisson, median, sparse, args.poisson_rtol, args.poisson_maxiter
                )
                repaired_count = int(
                    poisson_diag.get("invalid_pixels_repaired_from_median", 0)
                )
                poisson_status = "locally_repaired" if repaired_count else "ok"
            except (RuntimeError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError) as error:
                # A production pipeline cannot lose an entire sequence because
                # one numerical refiner failed.  Median depth is the declared
                # DA3 ablation and is already a valid metric map, so it is the
                # conservative scene-level fallback.  The exact failure is
                # recorded and printed; shape/interface errors still abort.
                refined = median.copy()
                poisson_full_median_fallback = True
                poisson_status = "full_median_fallback"
                poisson_diag = {
                    "full_median_fallback": True,
                    "fallback_exception_type": type(error).__name__,
                    "fallback_reason": str(error),
                    "invalid_pixels_repaired_from_median": int(median.size),
                    "invalid_pixel_repair_pct": 100.0,
                    "invalid_pixel_repair_strategy": "full_scene_da3_median_prior",
                }
            np.save(poisson_path, refined)

        fit_rows.append(
            {
                "scene": stem,
                "median_scale": scale,
                "independent_measurements": row["independent_measurements"],
                "sparse_input_pixels": int(np.count_nonzero(sparse > 0)),
                "da3_checkpoint": args.checkpoint,
                "process_res": args.process_res,
                "poisson_enabled": bool(poisson is not None),
                "poisson_status": poisson_status,
                "poisson_repaired_invalid_pixels": int(
                    poisson_diag.get("invalid_pixels_repaired_from_median", 0)
                ),
                "poisson_repaired_invalid_pct": float(
                    poisson_diag.get("invalid_pixel_repair_pct", 0.0)
                ),
                "poisson_full_median_fallback": poisson_full_median_fallback,
                "poisson_diagnostics": json.dumps(poisson_diag, default=str, sort_keys=True),
            }
        )
        poisson_note = ""
        if poisson_status == "locally_repaired":
            poisson_note = (
                "; Poisson repaired "
                f"{poisson_diag['invalid_pixels_repaired_from_median']} invalid pixels"
            )
        elif poisson_status == "full_median_fallback":
            poisson_note = "; WARNING Poisson failed, full median fallback saved"
        print(
            f"[{index:3d}/{len(rows)}] {stem} {inference_status}; "
            f"median scale={scale:.6g}{poisson_note}",
            flush=True,
        )

    save_csv(output_dir / "fit_parameters.csv", fit_rows)
    print(f"\nDA3 + median ablation predictions: {median_dir}")
    if poisson is not None:
        print(f"Primary DA3 + median + Poisson predictions: {poisson_dir}")
        locally_repaired_scenes = sum(
            row["poisson_status"] == "locally_repaired" for row in fit_rows
        )
        full_fallback_scenes = sum(
            bool(row["poisson_full_median_fallback"]) for row in fit_rows
        )
        repaired_pixels = sum(
            int(row["poisson_repaired_invalid_pixels"]) for row in fit_rows
        )
        print(
            "Poisson validity summary: "
            f"locally repaired scenes={locally_repaired_scenes}, "
            f"full median fallbacks={full_fallback_scenes}, "
            f"repaired/fallback pixels={repaired_pixels}"
        )


def strict_prediction(path: Path, shape: tuple[int, int], method: str) -> np.ndarray:
    prediction = load_npy_2d(path, shape)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{method} contains non-finite values: {path}")
    if not (prediction > 0).all():
        raise ValueError(f"{method} contains non-positive values: {path}")
    return prediction


def metric_values(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {name: math.nan for name in (
            "pixel_count", "rmse_m", "mae_m", "bias_m", "median_error_m",
            "p90_abs_m", "p95_abs_m", "absrel_pct", "delta1_pct",
            "bad_010_pct", "bad_025_pct", "bad_050_pct", "median_ratio",
        )}
    predicted = prediction[mask].astype(np.float64)
    target = truth[mask].astype(np.float64)
    error = predicted - target
    absolute = np.abs(error)
    ratio = np.maximum(predicted / target, target / predicted)
    return {
        "pixel_count": count,
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mae_m": float(np.mean(absolute)),
        "bias_m": float(np.mean(error)),
        "median_error_m": float(np.median(error)),
        "p90_abs_m": float(np.quantile(absolute, 0.90)),
        "p95_abs_m": float(np.quantile(absolute, 0.95)),
        "absrel_pct": float(100.0 * np.mean(absolute / target)),
        "delta1_pct": float(100.0 * np.mean(ratio < 1.25)),
        "bad_010_pct": float(100.0 * np.mean(absolute > 0.10)),
        "bad_025_pct": float(100.0 * np.mean(absolute > 0.25)),
        "bad_050_pct": float(100.0 * np.mean(absolute > 0.50)),
        "median_ratio": float(np.median(predicted / target)),
    }


def distance_from_sparse_support(sparse_exact: np.ndarray) -> np.ndarray:
    centers = np.isfinite(sparse_exact) & (sparse_exact > 0)
    if not np.any(centers):
        raise RuntimeError("The exact sparse measurement mask is empty")
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~centers).astype(np.float32)
    except ImportError as error:
        raise RuntimeError("Evaluation requires scipy for distance-to-support masks") from error


def region_masks(gt: np.ndarray, sparse_exact: np.ndarray, margin: int) -> dict[str, np.ndarray]:
    if margin < 0:
        raise ValueError("--outside-margin-px cannot be negative")
    valid = np.isfinite(gt) & (gt > 0) & (gt <= DEFAULT_MAX_DEPTH_M)
    yy = np.indices(gt.shape)[0]
    support_distance = distance_from_sparse_support(sparse_exact)
    outside = valid & (support_distance > margin)
    center_rows = np.where(sparse_exact > 0)[0]
    line_row = int(np.median(center_rows))
    return {
        "all_valid": valid,
        "outside_line_primary": outside,
        "above_line": outside & (yy < line_row),
        "below_line": outside & (yy > line_row),
        "outside_0_1m": outside & (gt < 1.0),
        "outside_1_2m": outside & (gt >= 1.0) & (gt < 2.0),
        "outside_2_4m": outside & (gt >= 2.0) & (gt < 4.0),
        "outside_4_10m": outside & (gt >= 4.0) & (gt <= DEFAULT_MAX_DEPTH_M),
    }


@dataclass(frozen=True)
class SurfacePatch:
    code: str
    role: str
    y0: int
    y1: int
    x0: int
    x1: int
    gt_m: float
    gt_span_m: float
    valid_pixels: int
    distance_to_support_px: float


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 2) -> bool:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    return not (
        ay1 + padding <= by0
        or by1 + padding <= ay0
        or ax1 + padding <= bx0
        or bx1 + padding <= ax0
    )


def surface_candidates(
    gt: np.ndarray,
    valid: np.ndarray,
    support_distance: np.ndarray,
    outside_margin: int,
    span_max_m: float,
) -> list[dict[str, Any]]:
    height, width = gt.shape
    patch_height = max(9, int(round(height * 0.065)))
    patch_width = max(13, int(round(width * 0.075)))
    if patch_height % 2 == 0:
        patch_height += 1
    if patch_width % 2 == 0:
        patch_width += 1
    half_h, half_w = patch_height // 2, patch_width // 2
    step_y, step_x = max(5, half_h), max(7, half_w)
    candidates: list[dict[str, Any]] = []
    for center_y in range(half_h, height - half_h, step_y):
        for center_x in range(half_w, width - half_w, step_x):
            y0, y1 = center_y - half_h, center_y + half_h + 1
            x0, x1 = center_x - half_w, center_x + half_w + 1
            patch_support_distance = support_distance[y0:y1, x0:x1]
            if float(np.min(patch_support_distance)) <= outside_margin:
                continue
            patch_valid = valid[y0:y1, x0:x1]
            total = patch_valid.size
            count = int(np.count_nonzero(patch_valid))
            if count < int(math.ceil(0.90 * total)):
                continue
            values = gt[y0:y1, x0:x1][patch_valid]
            q10, median, q90 = np.quantile(values, (0.10, 0.50, 0.90))
            span = float(q90 - q10)
            adaptive_limit = max(span_max_m, 0.06 * float(median))
            if span > adaptive_limit:
                continue
            candidates.append(
                {
                    "box": (y0, y1, x0, x1),
                    "center_y": center_y,
                    "center_x": center_x,
                    "gt_m": float(median),
                    "span_m": span,
                    "valid_pixels": count,
                    "distance_to_support_px": float(support_distance[center_y, center_x]),
                }
            )
    return candidates


def select_surface_patches(
    gt: np.ndarray,
    sparse_exact: np.ndarray,
    outside_margin: int,
    requested: int,
    span_max_m: float,
) -> list[SurfacePatch]:
    valid = np.isfinite(gt) & (gt > 0) & (gt <= DEFAULT_MAX_DEPTH_M)
    support_distance = distance_from_sparse_support(sparse_exact)
    candidates = surface_candidates(gt, valid, support_distance, outside_margin, span_max_m)
    if not candidates:
        return []
    height, width = gt.shape
    median_candidate_depth = float(np.median([candidate["gt_m"] for candidate in candidates]))
    roles: list[tuple[str, Callable[[dict[str, Any]], float]]] = [
        ("near depth", lambda c: c["gt_m"]),
        ("middle depth", lambda c: abs(c["gt_m"] - median_candidate_depth)),
        ("far depth", lambda c: -c["gt_m"]),
        ("far from support", lambda c: -c["distance_to_support_px"]),
        ("upper off-support", lambda c: c["center_y"]),
        ("lower off-support", lambda c: -c["center_y"]),
    ]
    chosen: list[tuple[str, dict[str, Any]]] = []
    for role, key in roles:
        for candidate in sorted(candidates, key=key):
            if all(not overlaps(candidate["box"], other["box"], padding=3) for _, other in chosen):
                chosen.append((role, candidate))
                break
        if len(chosen) >= requested:
            break

    while len(chosen) < requested:
        remaining = [
            candidate
            for candidate in candidates
            if all(not overlaps(candidate["box"], other["box"], padding=3) for _, other in chosen)
        ]
        if not remaining:
            break
        if not chosen:
            selected = remaining[len(remaining) // 2]
        else:
            gt_values = np.asarray([candidate["gt_m"] for candidate in candidates])
            depth_scale = max(float(np.ptp(gt_values)), 1e-3)

            def diversity(candidate: dict[str, Any]) -> float:
                distances = []
                for _, other in chosen:
                    dx = (candidate["center_x"] - other["center_x"]) / max(width - 1, 1)
                    dy = (candidate["center_y"] - other["center_y"]) / max(height - 1, 1)
                    dz = (candidate["gt_m"] - other["gt_m"]) / depth_scale
                    distances.append(dx * dx + dy * dy + dz * dz)
                return min(distances)

            selected = max(remaining, key=diversity)
        chosen.append(("additional stable surface", selected))

    patches: list[SurfacePatch] = []
    for index, (role, candidate) in enumerate(chosen):
        y0, y1, x0, x1 = candidate["box"]
        patches.append(
            SurfacePatch(
                code=chr(ord("A") + index),
                role=role,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                gt_m=float(candidate["gt_m"]),
                gt_span_m=float(candidate["span_m"]),
                valid_pixels=int(candidate["valid_pixels"]),
                distance_to_support_px=float(candidate["distance_to_support_px"]),
            )
        )
    return patches


def evaluate_surface_patches(
    scene: str,
    patches: Sequence[SurfacePatch],
    predictions: dict[str, np.ndarray],
    gt: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = np.isfinite(gt) & (gt > 0) & (gt <= DEFAULT_MAX_DEPTH_M)
    for patch in patches:
        patch_valid = valid[patch.y0 : patch.y1, patch.x0 : patch.x1]
        method_values: dict[str, tuple[float, float, float]] = {}
        for method, prediction in predictions.items():
            values = prediction[patch.y0 : patch.y1, patch.x0 : patch.x1][patch_valid]
            predicted_m = float(np.median(values))
            signed_error = predicted_m - patch.gt_m
            method_values[method] = (predicted_m, signed_error, abs(signed_error))
        primary_errors = {method: method_values[method][2] for method in PRIMARY_METHODS}
        winner = min(primary_errors, key=primary_errors.get)
        for method, (predicted_m, signed_error, absolute_error) in method_values.items():
            rows.append(
                {
                    "scene": scene,
                    "surface": patch.code,
                    "surface_role": patch.role,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "y0": patch.y0,
                    "y1": patch.y1,
                    "x0": patch.x0,
                    "x1": patch.x1,
                    "gt_m": patch.gt_m,
                    "gt_span_p10_p90_m": patch.gt_span_m,
                    "distance_to_lidar_support_px": patch.distance_to_support_px,
                    "pred_m": predicted_m,
                    "signed_error_m": signed_error,
                    "abs_error_m": absolute_error,
                    "winner_between_primary_methods": winner,
                }
            )
    return rows


def draw_patch_boxes(axis: Any, patches: Sequence[SurfacePatch], values: dict[str, float] | None = None) -> None:
    from matplotlib.patches import Rectangle

    colors = ("#00ffff", "#ff00ff", "#ffff00", "#00ff66", "#ff8800", "#66aaff", "#ffffff")
    for index, patch in enumerate(patches):
        color = colors[index % len(colors)]
        rectangle = Rectangle(
            (patch.x0, patch.y0),
            patch.x1 - patch.x0,
            patch.y1 - patch.y0,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
        )
        axis.add_patch(rectangle)
        suffix = f" {values[patch.code]:.2f}m" if values is not None and patch.code in values else ""
        axis.text(
            patch.x0 + 2,
            max(8, patch.y0 - 2),
            f"{patch.code}{suffix}",
            color="black",
            fontsize=7,
            weight="bold",
            bbox={"facecolor": color, "alpha": 0.88, "edgecolor": "none", "pad": 1.0},
        )


def panel_for_scene(
    scene: str,
    line_protocol: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    sparse_exact: np.ndarray,
    predictions: dict[str, np.ndarray],
    patches: Sequence[SurfacePatch],
    surface_rows: Sequence[dict[str, Any]],
    scene_metrics: dict[tuple[str, str], dict[str, float]],
    output: Path,
    plot_max_depth_m: float,
    plot_error_max_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    da3 = predictions["da3_median_poisson"]
    any2full = predictions["any2full"]
    valid = np.isfinite(gt) & (gt > 0) & (gt <= DEFAULT_MAX_DEPTH_M)
    da3_error = np.abs(da3 - gt)
    a2f_error = np.abs(any2full - gt)
    difference = np.where(valid, a2f_error - da3_error, np.nan)

    by_method_surface: dict[str, dict[str, float]] = defaultdict(dict)
    by_surface_primary: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in surface_rows:
        method = str(row["method"])
        code = str(row["surface"])
        by_method_surface[method][code] = float(row["pred_m"])
        if method in PRIMARY_METHODS:
            by_surface_primary[code][method] = {
                "pred_m": float(row["pred_m"]),
                "abs_error_m": float(row["abs_error_m"]),
            }

    figure, axes = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.imshow(rgb)
    yy, xx = np.where(sparse_exact > 0)
    axis.scatter(xx, yy, c=sparse_exact[yy, xx], cmap="turbo", vmin=0, vmax=plot_max_depth_m, s=10, edgecolors="white", linewidths=0.25)
    draw_patch_boxes(axis, patches)
    axis.set_title(f"RGB + one-line LiDAR centers ({len(xx)} independent points)")

    depth_image = axes[0, 1].imshow(gt, cmap="turbo", vmin=0, vmax=plot_max_depth_m)
    draw_patch_boxes(axes[0, 1], patches, {patch.code: patch.gt_m for patch in patches})
    axes[0, 1].set_title("NYU dense GT metric depth (m)")

    table_axis = axes[0, 2]
    table_axis.axis("off")
    table_rows: list[list[str]] = []
    for patch in patches:
        values = by_surface_primary.get(patch.code, {})
        da3_value = values.get("da3_median_poisson", {})
        a2f_value = values.get("any2full", {})
        if not da3_value or not a2f_value:
            continue
        winner = "DA3+P" if da3_value["abs_error_m"] < a2f_value["abs_error_m"] else "A2F"
        table_rows.append(
            [
                patch.code,
                patch.role,
                f"{patch.gt_m:.2f}",
                f"{da3_value['pred_m']:.2f} ({da3_value['pred_m'] - patch.gt_m:+.2f})",
                f"{a2f_value['pred_m']:.2f} ({a2f_value['pred_m'] - patch.gt_m:+.2f})",
                winner,
            ]
        )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=["Probe", "Surface role", "GT m", "DA3+P m (Δ)", "A2F m (Δ)", "Winner"],
        cellLoc="center",
        colWidths=[0.09, 0.22, 0.11, 0.22, 0.22, 0.14],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    table_axis.set_title("Locally stable surface measurements\n(patch medians; lower |error| is better)")

    axes[1, 0].imshow(da3, cmap="turbo", vmin=0, vmax=plot_max_depth_m)
    draw_patch_boxes(axes[1, 0], patches, by_method_surface["da3_median_poisson"])
    da3_metrics = scene_metrics[("da3_median_poisson", "outside_line_primary")]
    axes[1, 0].set_title(
        f"DA3-SMALL + median + Poisson\noutside-support RMSE {da3_metrics['rmse_m']:.3f} m | "
        f"MAE {da3_metrics['mae_m']:.3f} m | bias {da3_metrics['bias_m']:+.3f} m"
    )

    axes[1, 1].imshow(any2full, cmap="turbo", vmin=0, vmax=plot_max_depth_m)
    draw_patch_boxes(axes[1, 1], patches, by_method_surface["any2full"])
    a2f_metrics = scene_metrics[("any2full", "outside_line_primary")]
    axes[1, 1].set_title(
        f"Any2Full metric depth\noutside-support RMSE {a2f_metrics['rmse_m']:.3f} m | "
        f"MAE {a2f_metrics['mae_m']:.3f} m | bias {a2f_metrics['bias_m']:+.3f} m"
    )

    norm = TwoSlopeNorm(vmin=-plot_error_max_m, vcenter=0.0, vmax=plot_error_max_m)
    difference_image = axes[1, 2].imshow(difference, cmap="coolwarm", norm=norm)
    axes[1, 2].set_title("Absolute-error difference: A2F − DA3+P\nblue: A2F better; red: DA3+P better")

    for axis in axes.ravel():
        if axis is not table_axis:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(depth_image, ax=[axes[0, 1], axes[1, 0], axes[1, 1]], shrink=0.72, label="Camera-Z metric depth (m)")
    figure.colorbar(difference_image, ax=axes[1, 2], shrink=0.72, label="|error A2F| − |error DA3| (m)")
    figure.suptitle(
        f"{scene} — full metric-depth comparison — protocol: {line_protocol}\n"
        "All depth panels share one fixed scale; dense GT was not used to align either prediction.",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


class PixelAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sse = 0.0
        self.sae = 0.0
        self.se = 0.0
        self.sabsrel = 0.0
        self.bad010 = 0
        self.bad025 = 0
        self.bad050 = 0
        self.delta1 = 0

    def add(self, prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
        predicted = prediction[mask].astype(np.float64)
        target = truth[mask].astype(np.float64)
        error = predicted - target
        absolute = np.abs(error)
        ratio = np.maximum(predicted / target, target / predicted)
        self.count += int(target.size)
        self.sse += float(np.sum(error * error))
        self.sae += float(np.sum(absolute))
        self.se += float(np.sum(error))
        self.sabsrel += float(np.sum(absolute / target))
        self.bad010 += int(np.count_nonzero(absolute > 0.10))
        self.bad025 += int(np.count_nonzero(absolute > 0.25))
        self.bad050 += int(np.count_nonzero(absolute > 0.50))
        self.delta1 += int(np.count_nonzero(ratio < 1.25))

    def values(self) -> dict[str, float]:
        if self.count == 0:
            return {"pooled_pixel_count": 0}
        return {
            "pooled_pixel_count": self.count,
            "pooled_rmse_m": math.sqrt(self.sse / self.count),
            "pooled_mae_m": self.sae / self.count,
            "pooled_bias_m": self.se / self.count,
            "pooled_absrel_pct": 100.0 * self.sabsrel / self.count,
            "pooled_delta1_pct": 100.0 * self.delta1 / self.count,
            "pooled_bad_010_pct": 100.0 * self.bad010 / self.count,
            "pooled_bad_025_pct": 100.0 * self.bad025 / self.count,
            "pooled_bad_050_pct": 100.0 * self.bad050 / self.count,
        }


def summarize_pixel_metrics(
    rows: Sequence[dict[str, Any]],
    accumulators: dict[tuple[str, str], PixelAccumulator],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["region"]))].append(row)
    result: list[dict[str, Any]] = []
    for key, group in grouped.items():
        method, region = key
        summary: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "region": region,
            "scene_count": len(group),
        }
        for metric in (
            "rmse_m", "mae_m", "bias_m", "p90_abs_m", "p95_abs_m",
            "absrel_pct", "delta1_pct", "bad_010_pct", "bad_025_pct",
            "bad_050_pct", "median_ratio",
        ):
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[f"macro_mean_{metric}"] = float(np.mean(values)) if values.size else math.nan
            summary[f"macro_median_{metric}"] = float(np.median(values)) if values.size else math.nan
        summary.update(accumulators[key].values())
        result.append(summary)
    region_order = {
        "outside_line_primary": 0,
        "all_valid": 1,
        "above_line": 2,
        "below_line": 3,
        "outside_0_1m": 4,
        "outside_1_2m": 5,
        "outside_2_4m": 6,
        "outside_4_10m": 7,
    }
    method_order = {
        method: index
        for index, method in enumerate(("da3_median_poisson", "any2full", "da3_median"))
    }
    return sorted(result, key=lambda row: (region_order.get(row["region"], 99), method_order.get(row["method"], 99)))


def summarize_surfaces(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries: list[dict[str, Any]] = []
    for method, group in grouped.items():
        errors = np.asarray([float(row["signed_error_m"]) for row in group], dtype=np.float64)
        absolute = np.abs(errors)
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "surface_count": int(errors.size),
                "surface_rmse_m": float(np.sqrt(np.mean(errors * errors))),
                "surface_mae_m": float(np.mean(absolute)),
                "surface_bias_m": float(np.mean(errors)),
                "surface_p90_abs_m": float(np.quantile(absolute, 0.90)),
                "surface_within_010_pct": float(100.0 * np.mean(absolute <= 0.10)),
                "surface_within_025_pct": float(100.0 * np.mean(absolute <= 0.25)),
                "surface_within_050_pct": float(100.0 * np.mean(absolute <= 0.50)),
            }
        )
    return sorted(summaries, key=lambda row: float(row["surface_rmse_m"]))


def paired_bootstrap(
    per_scene_rows: Sequence[dict[str, Any]],
    surface_rows: Sequence[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    primary = "outside_line_primary"
    lookup: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["scene"]), str(row["method"])): row
        for row in per_scene_rows
        if row["region"] == primary and row["method"] in PRIMARY_METHODS
    }
    scenes = sorted({scene for scene, method in lookup if all((scene, m) in lookup for m in PRIMARY_METHODS)})
    results: list[dict[str, Any]] = []

    for metric in ("rmse_m", "mae_m", "absolute_bias_m"):
        if metric == "absolute_bias_m":
            differences = np.asarray(
                [
                    abs(float(lookup[(scene, "da3_median_poisson")]["bias_m"]))
                    - abs(float(lookup[(scene, "any2full")]["bias_m"]))
                    for scene in scenes
                ],
                dtype=np.float64,
            )
        else:
            differences = np.asarray(
                [
                    float(lookup[(scene, "da3_median_poisson")][metric])
                    - float(lookup[(scene, "any2full")][metric])
                    for scene in scenes
                ],
                dtype=np.float64,
            )
        bootstrap = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            selected = rng.integers(0, len(differences), len(differences))
            bootstrap[index] = float(np.mean(differences[selected]))
        mean = float(np.mean(differences))
        ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
        results.append(
            {
                "comparison": "DA3-SMALL + median + Poisson minus Any2Full",
                "unit": "m",
                "region": primary,
                "metric": metric,
                "paired_scene_count": len(scenes),
                "mean_difference": mean,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "da3_win_rate_pct": float(100.0 * np.mean(differences < 0)),
                "interpretation": "DA3 better" if ci_high < 0 else "Any2Full better" if ci_low > 0 else "inconclusive",
            }
        )

    surface_by_scene_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in surface_rows:
        if row["method"] in PRIMARY_METHODS:
            surface_by_scene_method[(str(row["scene"]), str(row["method"]))].append(float(row["abs_error_m"]))
    surface_scenes = sorted(
        scene
        for scene in {scene for scene, _ in surface_by_scene_method}
        if all((scene, method) in surface_by_scene_method for method in PRIMARY_METHODS)
    )
    surface_differences = np.asarray(
        [
            np.mean(surface_by_scene_method[(scene, "da3_median_poisson")])
            - np.mean(surface_by_scene_method[(scene, "any2full")])
            for scene in surface_scenes
        ],
        dtype=np.float64,
    )
    if surface_differences.size:
        bootstrap = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            selected = rng.integers(0, len(surface_differences), len(surface_differences))
            bootstrap[index] = float(np.mean(surface_differences[selected]))
        ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
        results.append(
            {
                "comparison": "DA3-SMALL + median + Poisson minus Any2Full",
                "unit": "m",
                "region": "stable_surface_patches_outside_support",
                "metric": "mean_surface_abs_error_m",
                "paired_scene_count": len(surface_scenes),
                "mean_difference": float(np.mean(surface_differences)),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "da3_win_rate_pct": float(100.0 * np.mean(surface_differences < 0)),
                "interpretation": "DA3 better" if ci_high < 0 else "Any2Full better" if ci_low > 0 else "inconclusive",
            }
        )
    return results


def make_leaderboard(
    pixel_summary: Sequence[dict[str, Any]],
    surface_summary: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outside = {
        row["method"]: row
        for row in pixel_summary
        if row["region"] == "outside_line_primary" and row["method"] in PRIMARY_METHODS
    }
    surfaces = {row["method"]: row for row in surface_summary if row["method"] in PRIMARY_METHODS}
    methods = list(PRIMARY_METHODS)
    labels = [METHOD_LABELS[method] for method in methods]
    colors = ["#2878b5", "#e07a1f"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    quantities = [
        ([outside[method]["pooled_rmse_m"] for method in methods], "Outside-support dense RMSE", "metres"),
        ([outside[method]["pooled_mae_m"] for method in methods], "Outside-support dense MAE", "metres"),
        ([surfaces[method]["surface_mae_m"] for method in methods], "Stable-surface MAE", "metres"),
    ]
    for axis, (values, title, unit) in zip(axes, quantities):
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("NYU-Depth V2: one-line dense metric-depth comparison (lower is better)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_report(
    pixel_summary: Sequence[dict[str, Any]],
    surface_summary: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    line_protocol: str,
    poisson_validity: dict[str, Any] | None,
    output: Path,
) -> None:
    outside = [
        row
        for row in pixel_summary
        if row["region"] == "outside_line_primary" and row["method"] in PRIMARY_METHODS
    ]
    outside_ablation = [
        row
        for row in pixel_summary
        if row["region"] == "outside_line_primary" and row["method"] == "da3_median"
    ]
    all_valid = [
        row
        for row in pixel_summary
        if row["region"] == "all_valid" and row["method"] in PRIMARY_METHODS
    ]
    surface_primary = [row for row in surface_summary if row["method"] in PRIMARY_METHODS]
    surface_ablation = [row for row in surface_summary if row["method"] == "da3_median"]
    lines = [
        "# NYU-Depth V2 one-line metric-depth comparison",
        "",
        "## Fixed protocol",
        "",
        f"- **Line protocol:** `{line_protocol}` (see `evaluated_protocol.json` for the complete definition).",
        "- Official 654-image NYU test split at 304 × 228 (unless a smoke-test limit was used).",
        "- The exact same sparse input array is supplied to DA3 alignment and Any2Full.",
        "- The primary DA3 system is DA3-SMALL + one-line global median metric scaling + Poisson refinement.",
        "- DA3-SMALL + median without Poisson is retained only as an ablation.",
        "- Dense GT is never used to align either final prediction.",
        "- Primary result: dense camera-Z RMSE in metres outside a margin around the actual sparse support.",
        "- Surface result: median metric depth over locally stable GT patches outside that support.",
        "- AbsRel is retained only as a secondary literature metric.",
    ]
    if poisson_validity is not None:
        lines.extend(
            [
                "",
                "## Poisson numerical validity",
                "",
                f"- Evaluated DA3 scenes with inference diagnostics: "
                f"{poisson_validity['diagnostic_scene_count']}.",
                f"- Scenes requiring isolated-pixel repair: "
                f"{poisson_validity['locally_repaired_scene_count']}.",
                f"- Scenes requiring a full DA3+median fallback: "
                f"{poisson_validity['full_median_fallback_scene_count']}.",
                f"- Total pixels replaced by the declared median fallback: "
                f"{poisson_validity['repaired_or_fallback_pixel_count']}.",
                "- Every repair is recorded per scene in the DA3 `fit_parameters.csv`; "
                "valid Poisson pixels were not changed.",
            ]
        )
    lines.extend(
        [
            "",
            "## Primary outside-support dense metric result (DA3-SMALL + median + Poisson vs Any2Full)",
            "",
        ]
    )
    for row in sorted(outside, key=lambda item: float(item.get("pooled_rmse_m", math.inf))):
        lines.append(
            f"- **{row['method_label']}**: pooled RMSE {row['pooled_rmse_m']:.4f} m; "
            f"MAE {row['pooled_mae_m']:.4f} m; bias {row['pooled_bias_m']:+.4f} m; "
            f"AbsRel {row['pooled_absrel_pct']:.3f}%."
        )
    lines.extend(["", "## Complete valid image", ""])
    for row in sorted(all_valid, key=lambda item: float(item.get("pooled_rmse_m", math.inf))):
        lines.append(
            f"- **{row['method_label']}**: pooled RMSE {row['pooled_rmse_m']:.4f} m; "
            f"MAE {row['pooled_mae_m']:.4f} m; bias {row['pooled_bias_m']:+.4f} m."
        )
    lines.extend(["", "## Stable-surface measurements", ""])
    for row in surface_primary:
        lines.append(
            f"- **{row['method_label']}**: surface RMSE {row['surface_rmse_m']:.4f} m; "
            f"MAE {row['surface_mae_m']:.4f} m; bias {row['surface_bias_m']:+.4f} m; "
            f"within 25 cm {row['surface_within_025_pct']:.1f}%."
        )
    lines.extend(["", "## Median-only DA3 ablation", ""])
    for row in outside_ablation:
        lines.append(
            f"- **{row['method_label']}** outside-support: pooled RMSE {row['pooled_rmse_m']:.4f} m; "
            f"MAE {row['pooled_mae_m']:.4f} m; bias {row['pooled_bias_m']:+.4f} m."
        )
    for row in surface_ablation:
        lines.append(
            f"- **{row['method_label']}** stable surfaces: RMSE {row['surface_rmse_m']:.4f} m; "
            f"MAE {row['surface_mae_m']:.4f} m."
        )
    lines.extend(["", "## Paired uncertainty (primary methods only)", ""])
    for row in paired:
        lines.append(
            f"- {row['region']} / {row['metric']}: DA3+median+Poisson−Any2Full "
            f"{row['mean_difference']:+.4f} m, "
            f"95% paired scene-bootstrap CI [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] m; "
            f"**{row['interpretation']}**."
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A method is called the stronger NYU indoor metric system only if its outside-support dense RMSE "
            "advantage is supported by the paired bootstrap and is not contradicted by stable-surface MAE. "
            "The NYU result must then be considered together with the already-completed iBims result; the "
            "two datasets are reported separately rather than pooling their pixels.",
            "",
            "NYU's dense depth is Kinect-derived and in-painted. It is valuable complementary evidence but "
            "does not replace iBims' higher-quality laser-scanner ground truth.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_poisson_validity(
    da3_dir: Path,
    selected_scenes: set[str],
) -> dict[str, Any] | None:
    path = da3_dir / "fit_parameters.csv"
    if not path.is_file():
        return None
    fit_rows = [row for row in read_csv(path) if row.get("scene") in selected_scenes]
    if not fit_rows or "poisson_status" not in fit_rows[0]:
        return None
    status_counts: dict[str, int] = defaultdict(int)
    repaired_pixels = 0
    fallback_scenes: list[str] = []
    locally_repaired_scenes: list[str] = []
    for row in fit_rows:
        status = row.get("poisson_status", "unknown")
        status_counts[status] += 1
        repaired_pixels += int(float(row.get("poisson_repaired_invalid_pixels", 0) or 0))
        if status == "full_median_fallback":
            fallback_scenes.append(row["scene"])
        elif status == "locally_repaired":
            locally_repaired_scenes.append(row["scene"])
    return {
        "diagnostic_scene_count": len(fit_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "locally_repaired_scene_count": len(locally_repaired_scenes),
        "locally_repaired_scenes": locally_repaired_scenes,
        "full_median_fallback_scene_count": len(fallback_scenes),
        "full_median_fallback_scenes": fallback_scenes,
        "repaired_or_fallback_pixel_count": repaired_pixels,
    }


def evaluate(args: argparse.Namespace) -> None:
    data_root = ensure_dir(args.data_root, "prepared NYU data root")
    da3_dir = ensure_dir(args.da3_dir, "DA3 output directory")
    any2full_dir = ensure_dir(args.any2full_dir, "Any2Full output directory")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = data_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Prepared protocol metadata is missing: {protocol_path}")
    protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    line_protocol = str(protocol_payload.get("line_protocol", "legacy_unspecified"))
    (output_dir / "evaluated_protocol.json").write_text(
        json.dumps(protocol_payload, indent=2) + "\n", encoding="utf-8"
    )
    if args.plot_max_depth_m <= 0 or args.plot_error_max_m <= 0:
        raise ValueError("plot scales must be positive")
    rows = load_manifest(data_root, args.limit)
    poisson_validity = summarize_poisson_validity(
        da3_dir, {str(row["scene"]) for row in rows}
    )
    if poisson_validity is not None:
        (output_dir / "poisson_validity_summary.json").write_text(
            json.dumps(poisson_validity, indent=2) + "\n", encoding="utf-8"
        )

    da3_metric_dir = da3_dir / "metric_m" if (da3_dir / "metric_m").is_dir() else da3_dir
    if args.da3_poisson_dir is not None:
        poisson_dir = ensure_dir(args.da3_poisson_dir, "DA3 Poisson output directory")
    elif (da3_dir / "metric_m_poisson").is_dir():
        poisson_dir = da3_dir / "metric_m_poisson"
    else:
        raise FileNotFoundError(
            "The primary DA3-SMALL + median + Poisson predictions are missing. "
            f"Expected {da3_dir / 'metric_m_poisson'} or pass --da3-poisson-dir."
        )
    method_dirs: dict[str, Path] = {
        "da3_median_poisson": poisson_dir,
        "any2full": any2full_dir,
        "da3_median": da3_metric_dir,
    }

    per_scene: list[dict[str, Any]] = []
    all_surface_rows: list[dict[str, Any]] = []
    accumulators: dict[tuple[str, str], PixelAccumulator] = defaultdict(PixelAccumulator)
    panels_dir = output_dir / "per_image_panels"
    surface_csv_dir = output_dir / "per_image_surface_tables"

    for index, manifest_row in enumerate(rows, start=1):
        stem = manifest_row["scene"]
        gt = load_npy_2d(data_root / "gt_depth_m" / f"{stem}.npy")
        sparse_exact = load_npy_2d(data_root / "sparse_exact_m" / f"{stem}.npy", gt.shape)
        rgb = np.asarray(Image.open(data_root / "rgb" / f"{stem}.png").convert("RGB"))
        line_row = int(manifest_row["line_row"])
        predictions = {
            method: strict_prediction(directory / f"{stem}.npy", gt.shape, method)
            for method, directory in method_dirs.items()
        }
        masks = region_masks(gt, sparse_exact, args.outside_margin_px)
        scene_metric_lookup: dict[tuple[str, str], dict[str, float]] = {}
        for method, prediction in predictions.items():
            for region, mask in masks.items():
                values = metric_values(prediction, gt, mask)
                scene_metric_lookup[(method, region)] = values
                per_scene.append(
                    {
                        "scene": stem,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "region": region,
                        **values,
                    }
                )
                accumulators[(method, region)].add(prediction, gt, mask)

        patches = select_surface_patches(
            gt,
            sparse_exact,
            args.outside_margin_px,
            args.surface_probes,
            args.surface_span_max_m,
        )
        surface_rows = evaluate_surface_patches(stem, patches, predictions, gt)
        all_surface_rows.extend(surface_rows)
        save_csv(
            surface_csv_dir / f"{stem}__surface_measurements.csv",
            surface_rows,
            fieldnames=(
                "scene", "surface", "surface_role", "method", "method_label",
                "y0", "y1", "x0", "x1", "gt_m", "gt_span_p10_p90_m",
                "distance_to_lidar_support_px",
                "pred_m", "signed_error_m", "abs_error_m", "winner_between_primary_methods",
            ),
        )
        if not args.skip_panels:
            panel_for_scene(
                stem,
                line_protocol,
                rgb,
                gt,
                sparse_exact,
                predictions,
                patches,
                surface_rows,
                scene_metric_lookup,
                panels_dir / f"{stem}__metric_depth_surface_comparison.png",
                args.plot_max_depth_m,
                args.plot_error_max_m,
            )
        da3_rmse = scene_metric_lookup[("da3_median_poisson", "outside_line_primary")]["rmse_m"]
        a2f_rmse = scene_metric_lookup[("any2full", "outside_line_primary")]["rmse_m"]
        print(
            f"[{index:3d}/{len(rows)}] {stem} surfaces={len(patches)} "
            f"outside RMSE: DA3+median+Poisson={da3_rmse:.3f}m A2F={a2f_rmse:.3f}m",
            flush=True,
        )

    pixel_summary = summarize_pixel_metrics(per_scene, accumulators)
    surface_summary = summarize_surfaces(all_surface_rows)
    paired = paired_bootstrap(per_scene, all_surface_rows, args.bootstrap_samples, args.seed)
    save_csv(output_dir / "per_scene_metrics.csv", per_scene)
    save_csv(output_dir / "per_surface_metrics.csv", all_surface_rows)
    save_csv(output_dir / "summary_dense_metric.csv", pixel_summary)
    save_csv(output_dir / "summary_surfaces.csv", surface_summary)
    save_csv(output_dir / "paired_bootstrap_da3_vs_any2full.csv", paired)
    make_leaderboard(pixel_summary, surface_summary, output_dir / "nyu_metric_leaderboard.png")
    write_report(
        pixel_summary,
        surface_summary,
        paired,
        line_protocol,
        poisson_validity,
        output_dir / "comparison_report.md",
    )

    print("\n===== NYU ONE-LINE DENSE METRIC RESULT =====")
    for row in pixel_summary:
        if row["region"] == "outside_line_primary":
            print(
                f"{row['method_label']:<34} RMSE={row['pooled_rmse_m']:.4f} m  "
                f"MAE={row['pooled_mae_m']:.4f} m  bias={row['pooled_bias_m']:+.4f} m  "
                f"AbsRel={row['pooled_absrel_pct']:.3f}%"
            )
    print("\nPaired decisions:")
    for row in paired:
        print(
            f"{row['region']:<38} {row['metric']:<28} "
            f"DA3+P-A2F={row['mean_difference']:+.4f}m "
            f"CI[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] {row['interpretation']}"
        )
    if poisson_validity is not None:
        print(
            "\nPoisson validity: "
            f"locally repaired scenes={poisson_validity['locally_repaired_scene_count']}, "
            f"full median fallbacks={poisson_validity['full_median_fallback_scene_count']}, "
            f"repaired/fallback pixels="
            f"{poisson_validity['repaired_or_fallback_pixel_count']}"
        )
    print(f"\nOutput: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_dataset(args)
    elif args.command == "infer-da3":
        infer_da3(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
