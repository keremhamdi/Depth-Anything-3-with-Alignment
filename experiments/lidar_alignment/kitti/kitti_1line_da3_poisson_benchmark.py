#!/usr/bin/env python3
"""Matched KITTI one-ring benchmark v2 for Any2Full and DA3 refinements.

This script deliberately keeps the SGTBN-style ring-32 extraction already used
by the project.  "SGTBN" describes the input protocol here; the SGTBN network
is not evaluated.

Stages
------
infer:
    Run pretrained DA3-Small independently on each KITTI RGB image and cache
    relative depth as float32 NPY.
evaluate:
    Load the fixed Any2Full and DA3 caches, align DA3 with the single LiDAR
    ring, apply the project's existing and OASIS Poisson refiners, and evaluate
    against KITTI ground truth outside the LiDAR support interval.
all:
    Run both stages.

The authoritative robot-facing output is the float32 ``*_depth_m.npy`` array.
Standard metrics retain the conventional 0--80 m cutoff, while the long-range
analysis and plots use 0--120 m.  KITTI-compatible
uint16 PNG exports use 256 units per metre (not millimetres, because 80,000 mm
does not fit in uint16).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image


METHODS = (
    "any2full",
    "da3_median",
    "existing_poisson",
    "oasis_poisson",
)

LABELS = {
    "any2full": "Any2Full",
    "da3_median": "DA3 + median",
    "existing_poisson": "DA3 + median + existing Poisson",
    "oasis_poisson": "DA3 + median + OASIS Poisson prior",
}

REGIONS = (
    "all_valid",
    "all_valid_120m",
    "outside_support",
    "below_support",
    "above_support",
    "all_nonanchor",
    "outside_support_120m",
    "all_nonanchor_120m",
    "far_80_120m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("infer", "evaluate", "all"), default="all")
    parser.add_argument(
        "--a2f-root",
        type=Path,
        default=Path(os.environ.get("A2F_ROOT", os.environ.get("DA3_LIDAR_DATA_ROOT", "/home/user/Projects/Any2Full"))),
    )
    parser.add_argument("--da3-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default="depth-anything/DA3-SMALL")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=80.0,
        help="Standard KITTI evaluation cutoff; retained for comparable results.",
    )
    parser.add_argument(
        "--long-eval-max-depth-m",
        type=float,
        default=120.0,
        help="Secondary long-range evaluation cutoff.",
    )
    parser.add_argument(
        "--plot-max-depth-m",
        type=float,
        default=120.0,
        help="Upper bound of the shared metric color scale.",
    )
    parser.add_argument("--min-depth-m", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def dataset_paths(args: argparse.Namespace) -> dict[str, Path]:
    kitti = args.a2f_root / "datasets/kitti_sgtbn/depth_selection/val_selection_cropped"
    relative = args.relative_dir or (args.output_dir / "da3_small_relative")
    return {
        "image": kitti / "image",
        "groundtruth": kitti / "groundtruth_depth",
        "intrinsics": kitti / "intrinsics",
        "ring": args.a2f_root / "experiments/sgtbn_replication/generated_1000_sgtbn",
        "any2full": args.a2f_root / "experiments/sgtbn_replication/predictions_1000_sgtbn",
        "relative": relative,
    }


def selected_stems(paths: dict[str, Path], args: argparse.Namespace) -> list[str]:
    stems = sorted(path.stem for path in paths["image"].glob("*.png"))
    if args.scene:
        wanted = set(args.scene)
        stems = [stem for stem in stems if stem in wanted]
        missing = sorted(wanted - set(stems))
        if missing:
            raise FileNotFoundError(f"Requested scenes were not found: {missing}")
    if args.limit is not None:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError(f"No KITTI images found in {paths['image']}")
    return stems


def validate_inputs(paths: dict[str, Path], stems: Iterable[str], require_relative: bool) -> None:
    required_dirs = ("image", "groundtruth", "intrinsics", "ring", "any2full")
    for key in required_dirs:
        if not paths[key].is_dir():
            raise FileNotFoundError(f"Missing {key} directory: {paths[key]}")
    if require_relative and not paths["relative"].is_dir():
        raise FileNotFoundError(f"Missing DA3 relative prediction directory: {paths['relative']}")

    suffixes = {
        "image": ".png",
        "intrinsics": ".txt",
        "ring": ".npy",
        "any2full": ".npy",
        "relative": ".npy",
    }
    for stem in stems:
        gt_stem = groundtruth_stem(stem)
        checks = [paths["image"] / f"{stem}.png", paths["groundtruth"] / f"{gt_stem}.png"]
        for key, suffix in suffixes.items():
            if key in ("image",) or (key == "relative" and not require_relative):
                continue
            checks.append(paths[key] / f"{stem}{suffix}")
        missing = [str(path) for path in checks if not path.is_file()]
        if missing:
            raise FileNotFoundError("Unmatched KITTI scene files:\n" + "\n".join(missing))


def groundtruth_stem(image_stem: str) -> str:
    token = "_sync_image_"
    if token not in image_stem:
        raise ValueError(f"Unexpected KITTI image filename: {image_stem}")
    return image_stem.replace(token, "_sync_groundtruth_depth_", 1)


def load_npy_2d(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D depth array, got {array.shape} from {path}")
    return array


def load_kitti_depth(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if raw.ndim != 2:
        raise ValueError(f"Expected single-channel KITTI depth PNG: {path}")
    return raw.astype(np.float32) / 256.0


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32, copy=False)
    height, width = shape
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def extract_prediction_depth(prediction: Any) -> np.ndarray:
    depth = prediction.depth if hasattr(prediction, "depth") else prediction["depth"]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    array = np.asarray(depth, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"DA3 returned depth with unexpected shape {np.asarray(depth).shape}")
    return array


def run_da3_inference(args: argparse.Namespace, paths: dict[str, Path], stems: list[str]) -> None:
    import torch
    from depth_anything_3.api import DepthAnything3

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    paths["relative"].mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.checkpoint} on {args.device}", flush=True)
    model = DepthAnything3.from_pretrained(args.checkpoint).to(args.device)
    model.eval()

    for index, stem in enumerate(stems, start=1):
        output_path = paths["relative"] / f"{stem}.npy"
        if output_path.is_file():
            cached = load_npy_2d(output_path)
            if np.isfinite(cached).all() and (cached > 0).any():
                print(f"[{index:4d}/{len(stems)}] {stem}  cached", flush=True)
                continue

        rgb_path = paths["image"] / f"{stem}.png"
        with Image.open(rgb_path) as image:
            target_shape = (image.height, image.width)
        with torch.inference_mode():
            prediction = model.inference(image=[str(rgb_path)], process_res=args.process_res)
        depth = resize_depth(extract_prediction_depth(prediction), target_shape)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise RuntimeError(f"DA3 produced no positive finite depth for {stem}")
        if not valid.all():
            fill = float(np.median(depth[valid]))
            depth = np.where(valid, depth, fill)
        np.save(output_path, depth.astype(np.float32))
        print(f"[{index:4d}/{len(stems)}] {stem}  saved {depth.shape}", flush=True)


def load_refiners(da3_root: Path) -> tuple[Callable[..., Any], Callable[..., Any]]:
    module_path = da3_root / "experiments/lidar_alignment/ibims/compare_median_poisson_oasis_100.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            "Cannot reuse the validated Poisson implementations because this file is missing: "
            f"{module_path}"
        )
    spec = importlib.util.spec_from_file_location("validated_ibims_poisson_comparison", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    existing = getattr(module, "existing_poisson", None)
    oasis = getattr(module, "oasis_hard_poisson", None)
    if not callable(existing) or not callable(oasis):
        raise AttributeError(
            f"{module_path} must define callable existing_poisson and oasis_hard_poisson functions"
        )
    print(f"Reusing validated refiners from {module_path}", flush=True)
    print(f"  existing_poisson{inspect.signature(existing)}", flush=True)
    print(f"  oasis_hard_poisson{inspect.signature(oasis)}", flush=True)
    return existing, oasis


def call_refiner(
    function: Callable[..., Any],
    base: np.ndarray,
    sparse: np.ndarray,
    anchors: np.ndarray,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Call the project's wrapper while tolerating keyword-only solver options."""
    signature = inspect.signature(function)
    values: dict[str, Any] = {}
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
    unknown_required = []
    for name, parameter in signature.parameters.items():
        if name in aliases:
            values[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            unknown_required.append(name)
    if unknown_required:
        # The validated wrappers previously used the five positional values below.
        try:
            result = function(base, sparse, anchors, rtol, maxiter)
        except TypeError as error:
            raise TypeError(
                f"Cannot map required parameters {unknown_required} for {function.__name__}{signature}"
            ) from error
    else:
        result = function(**values)

    if isinstance(result, tuple):
        refined, diagnostics = result[0], result[1]
    else:
        refined, diagnostics = result, {}
    refined_array = np.asarray(refined, dtype=np.float32).squeeze()
    if refined_array.shape != base.shape:
        raise ValueError(
            f"{function.__name__} returned {refined_array.shape}; expected {base.shape}"
        )
    if not isinstance(diagnostics, dict):
        diagnostics = {"value": diagnostics}
    return refined_array, json_safe(diagnostics)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def median_align(relative: np.ndarray, sparse: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, float]:
    ratios = sparse[anchors] / np.maximum(relative[anchors], 1e-8)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 3:
        raise RuntimeError(f"Only {ratios.size} valid ratios are available for median alignment")
    scale = float(np.median(ratios))
    return (relative * scale).astype(np.float32), scale


def scene_predictions(
    stem: str,
    paths: dict[str, Path],
    refiners: tuple[Callable[..., Any], Callable[..., Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    gt = load_kitti_depth(paths["groundtruth"] / f"{groundtruth_stem(stem)}.png")
    sparse = load_npy_2d(paths["ring"] / f"{stem}.npy")
    a2f = load_npy_2d(paths["any2full"] / f"{stem}.npy")
    relative = load_npy_2d(paths["relative"] / f"{stem}.npy")
    for name, array in (("sparse", sparse), ("Any2Full", a2f), ("DA3", relative)):
        if array.shape != gt.shape:
            raise ValueError(f"{stem}: {name} shape {array.shape} != GT shape {gt.shape}")

    anchors = np.isfinite(sparse) & (sparse > 0)
    median, scale = median_align(relative, sparse, anchors)
    existing, existing_diag = call_refiner(
        refiners[0], median, sparse, anchors, args.rtol, args.maxiter
    )
    oasis, oasis_diag = call_refiner(
        refiners[1], median, sparse, anchors, args.rtol, args.maxiter
    )
    predictions = {
        "any2full": a2f,
        "da3_median": median,
        "existing_poisson": existing,
        "oasis_poisson": oasis,
    }
    diagnostics = {
        "scene": stem,
        "anchor_count": int(anchors.sum()),
        "support_min_m": float(sparse[anchors].min()),
        "support_max_m": float(sparse[anchors].max()),
        "gt_max_m": float(gt[np.isfinite(gt) & (gt > 0)].max()),
        "gt_pixels_0_80m": int(
            (np.isfinite(gt) & (gt >= args.min_depth_m) & (gt <= args.max_depth_m)).sum()
        ),
        "gt_pixels_80_120m": int(
            (
                np.isfinite(gt)
                & (gt > args.max_depth_m)
                & (gt <= args.long_eval_max_depth_m)
            ).sum()
        ),
        "median_scale": scale,
        "existing_poisson": existing_diag,
        "oasis_poisson": oasis_diag,
    }
    return gt, sparse, predictions, diagnostics


def masks_for_scene(gt: np.ndarray, sparse: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    anchors = np.isfinite(sparse) & (sparse > 0)
    if not anchors.any():
        raise RuntimeError("The single-ring input has no valid anchors")
    valid = np.isfinite(gt) & (gt >= args.min_depth_m) & (gt <= args.max_depth_m)
    valid_long = (
        np.isfinite(gt)
        & (gt >= args.min_depth_m)
        & (gt <= args.long_eval_max_depth_m)
    )
    nonanchor = valid & ~anchors
    nonanchor_long = valid_long & ~anchors
    support_min = float(sparse[anchors].min())
    support_max = float(sparse[anchors].max())
    below = nonanchor & (gt < support_min)
    above = nonanchor & (gt > support_max)
    outside_long = nonanchor_long & ((gt < support_min) | (gt > support_max))
    return {
        "all_valid": valid,
        "all_valid_120m": valid_long,
        "outside_support": below | above,
        "below_support": below,
        "above_support": above,
        "all_nonanchor": nonanchor,
        "outside_support_120m": outside_long,
        "all_nonanchor_120m": nonanchor_long,
        "far_80_120m": nonanchor_long & (gt > args.max_depth_m),
    }


def metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
    max_depth_m: float | None = None,
) -> dict[str, float]:
    usable = mask & np.isfinite(pred)
    count = int(usable.sum())
    if count == 0:
        return {
            "pixel_count": 0,
            "absrel_pct": math.nan,
            "rmse_m": math.nan,
            "mae_m": math.nan,
            "irmse_per_km": math.nan,
            "imae_per_km": math.nan,
            "bias_m": math.nan,
        }
    truth = gt[usable].astype(np.float64)
    cutoff = args.max_depth_m if max_depth_m is None else max_depth_m
    estimate = np.clip(pred[usable].astype(np.float64), args.min_depth_m, cutoff)
    error = estimate - truth
    inverse_error = 1000.0 / estimate - 1000.0 / truth
    return {
        "pixel_count": count,
        "absrel_pct": float(100.0 * np.mean(np.abs(error) / truth)),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
        "irmse_per_km": float(np.sqrt(np.mean(inverse_error**2))),
        "imae_per_km": float(np.mean(np.abs(inverse_error))),
        "bias_m": float(np.mean(error)),
    }


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_scenes(rows: list[dict[str, Any]]) -> set[str]:
    counts: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        counts.setdefault(row["scene"], set()).add((row["method"], row["region"]))
    expected = {(method, region) for method in METHODS for region in REGIONS}
    return {scene for scene, combinations in counts.items() if combinations == expected}


def evaluate_all(
    args: argparse.Namespace,
    paths: dict[str, Path],
    stems: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[Callable[..., Any], Callable[..., Any]]]:
    refiners = load_refiners(args.da3_root)
    metrics_path = args.output_dir / "per_scene_metrics.csv"
    diagnostic_path = args.output_dir / "solver_diagnostics.csv"
    rows = read_csv(metrics_path) if args.resume else []
    diagnostics = read_csv(diagnostic_path) if args.resume else []
    complete = completed_scenes(rows)

    for index, stem in enumerate(stems, start=1):
        if stem in complete:
            print(f"[{index:4d}/{len(stems)}] {stem}  completed", flush=True)
            continue
        # Remove an incomplete attempt before recomputing it.
        rows = [row for row in rows if row.get("scene") != stem]
        diagnostics = [row for row in diagnostics if row.get("scene") != stem]
        gt, sparse, predictions, scene_diag = scene_predictions(stem, paths, refiners, args)
        region_masks = masks_for_scene(gt, sparse, args)
        for method, prediction in predictions.items():
            for region, mask in region_masks.items():
                row: dict[str, Any] = {
                    "scene": stem,
                    "method": method,
                    "method_label": LABELS[method],
                    "region": region,
                }
                cutoff = (
                    args.long_eval_max_depth_m
                    if region
                    in (
                        "all_valid_120m",
                        "outside_support_120m",
                        "all_nonanchor_120m",
                        "far_80_120m",
                    )
                    else args.max_depth_m
                )
                row.update(metrics(gt, prediction, mask, args, max_depth_m=cutoff))
                rows.append(row)
        diagnostics.append(
            {
                **{key: value for key, value in scene_diag.items() if not isinstance(value, dict)},
                "existing_poisson": json.dumps(scene_diag["existing_poisson"], sort_keys=True),
                "oasis_poisson": json.dumps(scene_diag["oasis_poisson"], sort_keys=True),
            }
        )
        atomic_csv(metrics_path, rows)
        atomic_csv(diagnostic_path, diagnostics)
        outside_count = int(region_masks["outside_support"].sum())
        outside_long_count = int(region_masks["outside_support_120m"].sum())
        print(
            f"[{index:4d}/{len(stems)}] {stem}  anchors={scene_diag['anchor_count']} "
            f"outside80={outside_count} outside120={outside_long_count}",
            flush=True,
        )
    return rows, diagnostics, refiners


def finite_float(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value


def aggregate(rows: list[dict[str, Any]], region: str = "outside_support") -> list[dict[str, Any]]:
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method and row["region"] == region]
        item: dict[str, Any] = {
            "method": method,
            "method_label": LABELS[method],
            "region": region,
            "scene_count": len(selected),
            "evaluable_scene_count": sum(finite_float(row, "pixel_count") > 0 for row in selected),
        }
        for key in ("absrel_pct", "rmse_m", "mae_m", "irmse_per_km", "imae_per_km", "bias_m"):
            values = np.asarray([finite_float(row, key) for row in selected], dtype=np.float64)
            values = values[np.isfinite(values)]
            item[f"mean_{key}"] = float(values.mean()) if values.size else math.nan
            item[f"median_{key}"] = float(np.median(values)) if values.size else math.nan
        failure_values = [
            finite_float(row, "absrel_pct")
            for row in selected
            if np.isfinite(finite_float(row, "absrel_pct"))
        ]
        failures = np.asarray([value >= 40.0 for value in failure_values], dtype=np.float64)
        item["failure_scene_rate_ge40_pct"] = float(100.0 * failures.mean()) if failures.size else math.nan
        summary.append(item)
    return sorted(summary, key=lambda row: row["mean_absrel_pct"])


def paired_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    region: str = "outside_support",
) -> list[dict[str, Any]]:
    by_key = {
        (row["scene"], row["method"]): row for row in rows if row["region"] == region
    }
    scenes = sorted({scene for scene, method in by_key if method == "da3_median"})
    rng = np.random.default_rng(args.seed)
    output = []
    for baseline in ("da3_median", "any2full"):
        for method in METHODS:
            if method == baseline:
                continue
            differences = []
            rmse_differences = []
            for scene in scenes:
                first = by_key.get((scene, baseline))
                second = by_key.get((scene, method))
                if first is None or second is None:
                    continue
                differences.append(finite_float(first, "absrel_pct") - finite_float(second, "absrel_pct"))
                rmse_differences.append(finite_float(first, "rmse_m") - finite_float(second, "rmse_m"))
            values = np.asarray(differences, dtype=np.float64)
            rmse_values = np.asarray(rmse_differences, dtype=np.float64)
            valid = np.isfinite(values)
            values = values[valid]
            rmse_values = rmse_values[valid]
            if values.size:
                indices = rng.integers(0, values.size, size=(args.bootstrap_samples, values.size))
                boot = values[indices].mean(axis=1)
                low, high = np.quantile(boot, (0.025, 0.975))
            else:
                low = high = math.nan
            output.append(
                {
                    "baseline": baseline,
                    "baseline_label": LABELS[baseline],
                    "method": method,
                    "method_label": LABELS[method],
                    "region": region,
                    "scene_count": int(values.size),
                    "mean_absrel_improvement_pp": float(values.mean()) if values.size else math.nan,
                    "absrel_win_rate_pct": float(100.0 * np.mean(values > 0)) if values.size else math.nan,
                    "bootstrap_ci_low_pp": float(low),
                    "bootstrap_ci_high_pp": float(high),
                    "mean_rmse_improvement_m": float(rmse_values.mean()) if rmse_values.size else math.nan,
                    "rmse_win_rate_pct": float(100.0 * np.mean(rmse_values > 0)) if rmse_values.size else math.nan,
                }
            )
    return output


def make_summary_plot(
    summary_standard: list[dict[str, Any]],
    summary_long: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    order = [row["method"] for row in reversed(summary_long)]
    standard_by_method = {row["method"]: row for row in summary_standard}
    long_by_method = {row["method"]: row for row in summary_long}
    labels = [LABELS[method] for method in order]
    standard_absrel = [standard_by_method[method]["mean_absrel_pct"] for method in order]
    long_absrel = [long_by_method[method]["mean_absrel_pct"] for method in order]
    long_rmse = [long_by_method[method]["mean_rmse_m"] for method in order]
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756"]
    figure, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for axis, values, title, xlabel in (
        (axes[0], standard_absrel, "All-valid-pixel AbsRel, 0–80 m", "Lower is better (%)"),
        (axes[1], long_absrel, "All-valid-pixel AbsRel, 0–120 m", "Lower is better (%)"),
        (axes[2], long_rmse, "All-valid-pixel RMSE, 0–120 m", "Lower is better (m)"),
    ):
        bars = axis.barh(labels, values, color=colors[: len(values)])
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", padding=4)
    figure.suptitle(
        "KITTI real Velodyne ring-32: main dense metric-depth comparison\n"
        "All valid KITTI GT pixels, matching the Any2Full paper's metric scope; 0–80 m and 0–120 m",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_intrinsics(path: Path) -> tuple[float, float, float, float]:
    numbers = np.fromstring(path.read_text(encoding="utf-8").replace(",", " "), sep=" ")
    if numbers.size >= 9:
        matrix = numbers[:9].reshape(3, 3)
        return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])
    if numbers.size >= 4:
        return tuple(float(value) for value in numbers[:4])  # type: ignore[return-value]
    raise ValueError(f"Could not parse camera intrinsics from {path}")


def probe_points(mask: np.ndarray, count: int = 5) -> list[tuple[int, int]]:
    height, width = mask.shape
    candidates = np.argwhere(mask)
    if not candidates.size:
        return []
    targets = [
        (0.18 * height, 0.15 * width),
        (0.42 * height, 0.35 * width),
        (0.56 * height, 0.52 * width),
        (0.68 * height, 0.70 * width),
        (0.78 * height, 0.88 * width),
    ][:count]
    selected: list[tuple[int, int]] = []
    scale_y = max(height, 1)
    scale_x = max(width, 1)
    for target_y, target_x in targets:
        distance = ((candidates[:, 0] - target_y) / scale_y) ** 2 + (
            (candidates[:, 1] - target_x) / scale_x
        ) ** 2
        for index in np.argsort(distance):
            point = (int(candidates[index, 0]), int(candidates[index, 1]))
            if all((point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 > 30**2 for old in selected):
                selected.append(point)
                break
    return selected


def ray_factor(y: int, x: int, intrinsics: tuple[float, float, float, float]) -> float:
    fx, fy, cx, cy = intrinsics
    return float(math.sqrt(1.0 + ((x - cx) / fx) ** 2 + ((y - cy) / fy) ** 2))


def make_probe_rows(
    points: list[tuple[int, int]],
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
    intrinsics: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    rows = []
    for index, (y, x) in enumerate(points):
        factor = ray_factor(y, x, intrinsics)
        row: dict[str, Any] = {
            "probe": chr(ord("A") + index),
            "x_px": x,
            "y_px": y,
            "gt_forward_z_m": float(gt[y, x]),
            "gt_range_m": float(gt[y, x] * factor),
        }
        for method, prediction in predictions.items():
            row[f"{method}_forward_z_m"] = float(prediction[y, x])
            row[f"{method}_range_m"] = float(prediction[y, x] * factor)
        rows.append(row)
    return rows


def draw_example(
    role: str,
    stem: str,
    gt: np.ndarray,
    sparse: np.ndarray,
    predictions: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    rgb: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    args: argparse.Namespace,
    output: Path,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    points = probe_points(masks["all_valid_120m"])
    probe_rows = make_probe_rows(points, gt, predictions, intrinsics)
    maps = [
        ("rgb", "RGB + real one-ring LiDAR", rgb),
        ("groundtruth", "KITTI semi-dense ground truth", gt),
        *[(method, LABELS[method], predictions[method]) for method in METHODS],
    ]
    figure, axes = plt.subplots(2, 3, figsize=(21, 11.8))
    axes_flat = axes.ravel()
    image_handle = None
    anchor_y, anchor_x = np.nonzero(np.isfinite(sparse) & (sparse > 0))
    for axis, (key, title, data) in zip(axes_flat, maps):
        if key == "rgb":
            axis.imshow(data)
            axis.scatter(
                anchor_x,
                anchor_y,
                c=sparse[anchor_y, anchor_x],
                cmap="turbo",
                vmin=0,
                vmax=args.plot_max_depth_m,
                s=5,
                linewidths=0,
            )
        else:
            masked = np.ma.masked_where(~np.isfinite(data) | (data <= 0), data)
            image_handle = axis.imshow(masked, cmap="turbo", vmin=0, vmax=args.plot_max_depth_m)
        for point_index, (y, x) in enumerate(points):
            label = chr(ord("A") + point_index)
            axis.scatter([x], [y], s=52, facecolor="white", edgecolor="black", linewidth=1.2)
            axis.text(
                x + 7,
                max(y - 7, 8),
                label,
                color="black",
                fontsize=9,
                weight="bold",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "black", "alpha": 0.9},
            )
        if key in METHODS:
            overall_result = metrics(
                gt,
                predictions[key],
                masks["all_valid_120m"],
                args,
                max_depth_m=args.long_eval_max_depth_m,
            )
            outside_result = metrics(
                gt,
                predictions[key],
                masks["outside_support_120m"],
                args,
                max_depth_m=args.long_eval_max_depth_m,
            )
            title += (
                f"\nOverall: AbsRel {overall_result['absrel_pct']:.2f}% | "
                f"RMSE {overall_result['rmse_m']:.2f} m"
                f"  •  Outside: {outside_result['absrel_pct']:.2f}% | "
                f"{outside_result['rmse_m']:.2f} m"
            )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.axis("off")

    if image_handle is not None:
        colorbar_axis = figure.add_axes((0.945, 0.22, 0.012, 0.54))
        colorbar = figure.colorbar(image_handle, cax=colorbar_axis)
        colorbar.set_label(
            f"Camera-forward metric depth Z (m), shared 0–{args.plot_max_depth_m:g} m scale",
            fontsize=11,
        )
        tick_step = 20.0 if args.plot_max_depth_m >= 100 else 10.0
        colorbar.set_ticks(np.arange(0, args.plot_max_depth_m + 0.1, tick_step))

    table_header = "Probe   GT range   Any2Full   DA3 median   Existing P.   OASIS P."
    table_lines = [table_header]
    for row in probe_rows:
        table_lines.append(
            f"  {row['probe']}      {row['gt_range_m']:7.2f} m   "
            f"{row['any2full_range_m']:7.2f} m   {row['da3_median_range_m']:7.2f} m   "
            f"{row['existing_poisson_range_m']:7.2f} m   {row['oasis_poisson_range_m']:7.2f} m"
        )
    figure.text(
        0.5,
        0.018,
        "Representative valid-GT points — Euclidean camera-to-point range\n"
        + "\n".join(table_lines),
        ha="center",
        va="bottom",
        family="monospace",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": "#444", "alpha": 0.96},
    )
    figure.suptitle(
        f"{role.upper()} example — {stem}\n"
        "All methods share the identical ring input, evaluation mask, and full metric scale",
        fontsize=16,
        weight="bold",
    )
    figure.subplots_adjust(left=0.015, right=0.925, top=0.90, bottom=0.19, wspace=0.03, hspace=0.14)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return probe_rows


def encode_kitti_depth(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    clipped = np.clip(depth[valid], 0.0, 65535.0 / 256.0)
    encoded[valid] = np.rint(clipped * 256.0).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(path)


def export_metric_arrays(
    directory: Path,
    gt: np.ndarray,
    sparse: np.ndarray,
    predictions: dict[str, np.ndarray],
    intrinsics: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"groundtruth": gt, "single_ring_lidar": sparse, **predictions}
    for name, depth in arrays.items():
        np.save(directory / f"{name}__depth_m_float32.npy", depth.astype(np.float32))
        encode_kitti_depth(directory / f"{name}__depth_kitti_u16.png", depth)
    metadata = {
        "authoritative_array": "*_depth_m_float32.npy",
        "array_units": "metres",
        "depth_definition": "camera-forward Z depth, not Euclidean ray range",
        "png_encoding": "uint16, metres = stored_value / 256.0; zero is invalid",
        "standard_evaluation_clip_m": [args.min_depth_m, args.max_depth_m],
        "long_range_evaluation_clip_m": [args.min_depth_m, args.long_eval_max_depth_m],
        "visualization_scale_m": [0.0, args.plot_max_depth_m],
        "raw_float32_clipping": "none",
        "uint16_png_max_representable_m": 65535.0 / 256.0,
        "intrinsics": {"fx": intrinsics[0], "fy": intrinsics[1], "cx": intrinsics[2], "cy": intrinsics[3]},
        "range_conversion": "range_m = depth_z_m * sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)",
    }
    (directory / "metric_depth_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def select_examples(
    rows: list[dict[str, Any]],
    champion: str,
    region: str = "all_valid_120m",
) -> list[tuple[str, str]]:
    candidates = [
        row
        for row in rows
        if row["method"] == champion
        and row["region"] == region
        and finite_float(row, "pixel_count") >= 50
        and np.isfinite(finite_float(row, "absrel_pct"))
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda row: finite_float(row, "absrel_pct"))
    values = np.asarray([finite_float(row, "absrel_pct") for row in candidates])
    median_value = float(np.median(values))
    typical = min(candidates, key=lambda row: abs(finite_float(row, "absrel_pct") - median_value))
    return [("best", candidates[0]["scene"]), ("typical", typical["scene"]), ("worst", candidates[-1]["scene"])]


def create_examples(
    selections: list[tuple[str, str]],
    paths: dict[str, Path],
    refiners: tuple[Callable[..., Any], Callable[..., Any]],
    args: argparse.Namespace,
) -> None:
    examples = args.output_dir / "examples_best_typical_worst"
    examples.mkdir(parents=True, exist_ok=True)
    for role, stem in selections:
        gt, sparse, predictions, _ = scene_predictions(stem, paths, refiners, args)
        masks = masks_for_scene(gt, sparse, args)
        rgb = load_rgb(paths["image"] / f"{stem}.png")
        intrinsics = parse_intrinsics(paths["intrinsics"] / f"{stem}.txt")
        probe_rows = draw_example(
            role,
            stem,
            gt,
            sparse,
            predictions,
            masks,
            rgb,
            intrinsics,
            args,
            examples / f"{role}__{stem}__full_metric_comparison.png",
        )
        scene_directory = examples / "metric_arrays" / f"{role}__{stem}"
        export_metric_arrays(scene_directory, gt, sparse, predictions, intrinsics, args)
        atomic_csv(scene_directory / "distance_probes.csv", probe_rows)


def write_report(
    summary_standard: list[dict[str, Any]],
    summary_long: list[dict[str, Any]],
    summary_far: list[dict[str, Any]],
    summary_outside_standard: list[dict[str, Any]],
    summary_outside_long: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    selections: list[tuple[str, str]],
    output: Path,
) -> None:
    lines = [
        "KITTI real one-ring dense metric-depth comparison",
        "Headline region: all valid KITTI ground-truth pixels, as in the Any2Full evaluation protocol",
        "Standard comparable evaluation: 0–80 m",
        "Additional long-range evaluation and visual scale: 0–120 m",
        "Outside-support is reported separately as a diagnostic.",
        "",
        "Standard 0–80 m ranking (lower is better):",
    ]
    for index, row in enumerate(summary_standard, start=1):
        lines.append(
            f"{index}. {row['method_label']}: mean/median AbsRel "
            f"{row['mean_absrel_pct']:.3f}% / {row['median_absrel_pct']:.3f}%; "
            f"mean RMSE {row['mean_rmse_m']:.3f} m; mean MAE {row['mean_mae_m']:.3f} m; "
            f">=40% failures {row['failure_scene_rate_ge40_pct']:.1f}%"
        )
    lines.extend(["", "Long-range 0–120 m ranking (lower is better):"])
    for index, row in enumerate(summary_long, start=1):
        lines.append(
            f"{index}. {row['method_label']}: mean/median AbsRel "
            f"{row['mean_absrel_pct']:.3f}% / {row['median_absrel_pct']:.3f}%; "
            f"mean RMSE {row['mean_rmse_m']:.3f} m; mean MAE {row['mean_mae_m']:.3f} m; "
            f">=40% failures {row['failure_scene_rate_ge40_pct']:.1f}%"
        )
    lines.extend(["", "Isolated 80–120 m GT band:"])
    for row in summary_far:
        lines.append(
            f"- {row['method_label']}: mean AbsRel {row['mean_absrel_pct']:.3f}%; "
            f"mean RMSE {row['mean_rmse_m']:.3f} m; "
            f"evaluable scenes {row['evaluable_scene_count']}/{row['scene_count']}"
        )
    lines.extend(["", "Outside-support diagnostic, 0–80 m:"])
    for row in summary_outside_standard:
        lines.append(
            f"- {row['method_label']}: mean AbsRel {row['mean_absrel_pct']:.3f}%; "
            f"mean RMSE {row['mean_rmse_m']:.3f} m"
        )
    lines.extend(["", "Outside-support diagnostic, 0–120 m:"])
    for row in summary_outside_long:
        lines.append(
            f"- {row['method_label']}: mean AbsRel {row['mean_absrel_pct']:.3f}%; "
            f"mean RMSE {row['mean_rmse_m']:.3f} m"
        )
    lines.extend(["", "Directly against DA3 + median:"])
    for row in paired:
        if (
            row["baseline"] != "da3_median"
            or row["method"] not in ("existing_poisson", "oasis_poisson")
            or row["region"]
            not in (
                "all_valid",
                "all_valid_120m",
                "outside_support",
                "outside_support_120m",
            )
        ):
            continue
        lines.append(
            f"{row['method_label']} [{row['region']}]: mean AbsRel improvement "
            f"{row['mean_absrel_improvement_pp']:+.3f} pp; wins {row['absrel_win_rate_pct']:.1f}%; "
            f"95% paired bootstrap CI [{row['bootstrap_ci_low_pp']:+.3f}, "
            f"{row['bootstrap_ci_high_pp']:+.3f}] pp; "
            f"mean RMSE improvement {row['mean_rmse_improvement_m']:+.3f} m"
        )
    lines.extend(
        [
            "",
            "Selected full-scale examples:",
            *[f"- {role}: {stem}" for role, stem in selections],
            "",
            "Robot-facing outputs:",
            "The float32 NPY files store camera-forward metric depth Z in metres.",
            "The uint16 PNG files use KITTI encoding: metres = value / 256; zero is invalid.",
            "The plots use one shared 0–120 m color scale. Their probe table reports",
            "Euclidean camera-to-point range in metres, computed with that frame's intrinsics.",
            "Values beyond 120 m remain unchanged in float32 arrays and in the numerical probe table;",
            "only their display color saturates at the top of the 120 m colorbar.",
            "",
            "OASIS note: this is the training-free hard-anchor Poisson prior stage, not the full trained OASIS-DC network.",
            "Protocol note: this is a custom real one-ring KITTI test. The Any2Full paper's KITTI Sparse-LiDAR table uses denser LiDAR protocols, so its published number is not expected to match this one-ring result.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.long_eval_max_depth_m < args.max_depth_m:
        raise ValueError("--long-eval-max-depth-m must be >= --max-depth-m")
    if args.plot_max_depth_m < args.long_eval_max_depth_m:
        raise ValueError("--plot-max-depth-m must be >= --long-eval-max-depth-m")
    args.da3_root = args.da3_root.resolve()
    args.a2f_root = args.a2f_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.relative_dir is not None:
        args.relative_dir = args.relative_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(args)
    stems = selected_stems(paths, args)
    validate_inputs(paths, stems, require_relative=False)

    print("\n========== MATCHED KITTI INPUT ==========")
    print(f"Scenes: {len(stems)}")
    print(f"RGB/GT: {paths['image'].parent}")
    print(f"Real one-ring inputs: {paths['ring']}")
    print(f"Fixed Any2Full predictions: {paths['any2full']}")
    print(f"DA3-Small relative cache: {paths['relative']}")

    if args.stage in ("infer", "all"):
        run_da3_inference(args, paths, stems)
    if args.stage == "infer":
        return

    validate_inputs(paths, stems, require_relative=True)
    rows, diagnostics, refiners = evaluate_all(args, paths, stems)
    summary_standard = aggregate(rows, region="all_valid")
    summary_long = aggregate(rows, region="all_valid_120m")
    summary_far = aggregate(rows, region="far_80_120m")
    summary_outside_standard = aggregate(rows, region="outside_support")
    summary_outside_long = aggregate(rows, region="outside_support_120m")
    paired = (
        paired_rows(rows, args, region="all_valid")
        + paired_rows(rows, args, region="all_valid_120m")
        + paired_rows(rows, args, region="outside_support")
        + paired_rows(rows, args, region="outside_support_120m")
    )
    atomic_csv(args.output_dir / "summary_all_valid_0_80m.csv", summary_standard)
    atomic_csv(args.output_dir / "summary_all_valid_0_120m.csv", summary_long)
    atomic_csv(args.output_dir / "summary_far_80_120m.csv", summary_far)
    atomic_csv(
        args.output_dir / "diagnostic_outside_support_0_80m.csv",
        summary_outside_standard,
    )
    atomic_csv(
        args.output_dir / "diagnostic_outside_support_0_120m.csv",
        summary_outside_long,
    )
    atomic_csv(args.output_dir / "paired_comparisons.csv", paired)
    make_summary_plot(
        summary_standard,
        summary_long,
        args.output_dir / "00_main_all_valid_comparison.png",
    )

    champion = summary_long[0]["method"]
    selections = select_examples(rows, champion)
    create_examples(selections, paths, refiners, args)
    write_report(
        summary_standard,
        summary_long,
        summary_far,
        summary_outside_standard,
        summary_outside_long,
        paired,
        selections,
        args.output_dir / "comparison_report.txt",
    )
    print(f"\nWrote matched comparison to: {args.output_dir}")
    print("Open 00_main_all_valid_comparison.png and comparison_report.txt first.")
    print("Then open examples_best_typical_worst once; it contains the three full-scale panels.")


if __name__ == "__main__":
    main()
