#!/usr/bin/env python3
"""Paired iBims indoor comparison: one-line versus simulated four-line LiDAR.

Both conditions use the same cached DA3-SMALL relative-depth prediction, the
same median metric alignment, and the same validated existing-Poisson refiner.
The only changed input is LiDAR support:

* one-line: the established iBims v2.1/32 m sparse sensor map;
* four-line: four one-pixel horizontal lines at fixed normalized image rows,
  using the one-line map's measured x-columns and iBims GT ranges at those rays.

Dense iBims ground truth is used only to simulate the hypothetical four-line
returns and to score the completed maps. It is never used for post-inference
alignment outside the selected sparse anchors. No training or sensor noise is
added.

The evaluator reports both median-only and median+Poisson results. Every
one-line/four-line comparison uses an identical pixel mask. Successful scenes
are saved immediately and ``--resume`` safely continues an interrupted run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt


VERSION = "1.0"
METHOD_LABELS = {
    "one_line_median": "1 line — DA3 + median",
    "four_line_median": "4 lines — DA3 + median",
    "one_line_poisson": "1 line — DA3 + median + existing Poisson",
    "four_line_poisson": "4 lines — DA3 + median + existing Poisson",
}
REGION_LABELS = {
    "all_valid": "All valid GT pixels",
    "outside_original_line_common": "Outside original 1-line support (common mask)",
    "outside_both_patterns_common": "Outside both input patterns (common far-field mask)",
}
PAIR_SPECS = (
    ("median", "one_line_median", "four_line_median"),
    ("poisson", "one_line_poisson", "four_line_poisson"),
)
LOWER_IS_BETTER = {
    "rmse_m": True,
    "absrel_pct": True,
    "mae_m": True,
    "bias_m": False,
    "delta1_pct": False,
    "bad_050_pct": True,
    "bad_100_pct": True,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-root", type=Path, required=True)
    parser.add_argument("--a2f-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional extra root searched for iBims files.",
    )
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--one-line-dir", type=Path)
    parser.add_argument("--da3-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--row-fracs",
        type=float,
        nargs=4,
        default=(0.20, 0.40, 0.60, 0.80),
        metavar=("R1", "R2", "R3", "R4"),
    )
    parser.add_argument("--sensor-min-depth-m", type=float, default=0.10)
    parser.add_argument("--sensor-max-depth-m", type=float, default=32.0)
    parser.add_argument(
        "--eval-max-depth-m",
        type=float,
        default=0.0,
        help="Zero means use every valid iBims GT depth.",
    )
    parser.add_argument("--outside-margin-px", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--expected-scenes", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scene")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument("--plot-max-depth-m", type=float, default=10.0)
    parser.add_argument("--plot-gain-max-m", type=float, default=1.0)
    return parser.parse_args()


def search_roots(args: argparse.Namespace) -> list[Path]:
    raw: list[Path | None] = [args.a2f_root, args.data_root]
    if os.environ.get("DA3_LIDAR_DATA_ROOT"):
        raw.append(Path(os.environ["DA3_LIDAR_DATA_ROOT"]))
    result: list[Path] = []
    seen: set[str] = set()
    for item in raw:
        if item is None:
            continue
        resolved = item.expanduser().resolve()
        if resolved.exists() and str(resolved) not in seen:
            result.append(resolved)
            seen.add(str(resolved))
    return result


def resolve_gt(explicit: Path | None, roots: list[Path]) -> Path:
    if explicit is not None:
        directory = explicit.expanduser().resolve()
        if not directory.is_dir() or next(directory.glob("*.mat"), None) is None:
            raise FileNotFoundError(f"iBims MAT directory is invalid: {directory}")
        return directory
    relatives = (
        "datasets/ibims1/ibims1_core_mat",
        "datasets/ibims1_core_mat",
        "data/ibims1/ibims1_core_mat",
        "data/ibims1_core_mat",
        "ibims1_core_mat",
    )
    for root in roots:
        for relative in relatives:
            directory = root / relative
            if directory.is_dir() and next(directory.glob("*.mat"), None):
                return directory.resolve()
    counts: Counter[Path] = Counter()
    for root in roots:
        for path in root.rglob("*.mat"):
            if "ibims" in str(path).lower():
                counts[path.parent.resolve()] += 1
    if not counts:
        raise FileNotFoundError("Could not locate the iBims core MAT files")
    return max(counts, key=counts.get)


def resolve_npy_dir(
    explicit: Path | None,
    roots: list[Path],
    names: Iterable[str],
    label: str,
) -> Path:
    names = tuple(names)
    if explicit is not None:
        directory = explicit.expanduser().resolve()
        if not directory.is_dir() or next(directory.glob("*.npy"), None) is None:
            raise FileNotFoundError(f"{label} is invalid: {directory}")
        return directory
    for root in roots:
        for name in names:
            for relative in (
                f"experiments/ibims_replication/{name}",
                f"ibims_replication/{name}",
                name,
            ):
                directory = root / relative
                if directory.is_dir() and next(directory.glob("*.npy"), None):
                    return directory.resolve()
    found: list[tuple[int, int, Path]] = []
    for root in roots:
        for priority, name in enumerate(names):
            for directory in root.rglob(name):
                if directory.is_dir():
                    count = sum(1 for _ in directory.glob("*.npy"))
                    if count:
                        found.append((priority, -count, directory.resolve()))
    if not found:
        raise FileNotFoundError(f"Could not locate {label}")
    return min(found)[2]


def mat_field(record: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(record[name])
    except Exception as error:
        raise KeyError(f"Missing iBims MAT field {name!r}") from error


def load_ibims(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = loadmat(path)["data"][0, 0]
    gt = np.squeeze(mat_field(record, "depth")).astype(np.float32)
    valid = (
        (np.squeeze(mat_field(record, "mask_invalid")) == 1)
        & (np.squeeze(mat_field(record, "mask_transp")) == 1)
        & np.isfinite(gt)
        & (gt > 0)
    )
    rgb = np.squeeze(mat_field(record, "rgb"))
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if np.nanmax(rgb) <= 1:
            rgb *= 255
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.shape[:2] != gt.shape:
        raise ValueError(f"RGB/GT shape mismatch in {path}: {rgb.shape[:2]} vs {gt.shape}")
    return gt, valid, rgb


def npy_path(directory: Path, scene: str, da3: bool = False) -> Path:
    names = (
        [f"{scene}_da3small.npy", f"{scene}_da3.npy", f"{scene}.npy"]
        if da3
        else [f"{scene}.npy", f"{scene}_sensor.npy", f"{scene}_sparse.npy"]
    )
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    matches = sorted(directory.glob(f"{scene}*.npy"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve {scene} in {directory}: {matches}")


def load_npy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    array = np.squeeze(np.load(path)).astype(np.float32)
    if array.shape != shape:
        raise ValueError(f"{path}: {array.shape} != {shape}")
    return array


def load_poisson(da3_root: Path) -> Callable[..., Any]:
    path = (
        da3_root
        / "experiments/lidar_alignment/ibims/compare_median_poisson_oasis_100.py"
    )
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
    print(f"Reusing existing_poisson{inspect.signature(function)} from {path}")
    return function


def call_poisson(
    function: Callable[..., Any],
    base: np.ndarray,
    sparse: np.ndarray,
    anchors: np.ndarray,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    result = (
        function(base, sparse, anchors, rtol, maxiter)
        if unknown
        else function(**kwargs)
    )
    prediction, diagnostics = result if isinstance(result, tuple) else (result, {})
    prediction = np.squeeze(np.asarray(prediction, dtype=np.float32))
    if prediction.shape != base.shape:
        raise ValueError(f"Poisson returned {prediction.shape}; expected {base.shape}")
    diagnostic_payload = (
        dict(diagnostics)
        if isinstance(diagnostics, dict)
        else {"value": diagnostics}
    )
    invalid = ~np.isfinite(prediction) | (prediction <= 0)
    invalid_count = int(np.count_nonzero(invalid))
    if invalid_count:
        prediction = prediction.copy()
        prediction[invalid] = base[invalid]
    if not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise RuntimeError("Poisson output remains invalid after same-pixel base repair")
    diagnostic_payload.update(
        {
            "invalid_pixels_repaired_from_median": invalid_count,
            "invalid_pixel_repair_pct": 100.0 * invalid_count / prediction.size,
        }
    )
    return prediction.astype(np.float32), diagnostic_payload


def sanitize_one_line(
    sparse: np.ndarray,
    valid: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = (
        valid
        & np.isfinite(sparse)
        & (sparse >= min_depth_m)
        & (sparse <= max_depth_m)
    )
    result = np.zeros_like(sparse, dtype=np.float32)
    result[anchors] = sparse[anchors]
    if int(np.count_nonzero(anchors)) < 8:
        raise RuntimeError(f"Only {int(np.count_nonzero(anchors))} valid one-line anchors")
    return result, anchors


def simulate_four_lines(
    gt: np.ndarray,
    valid: np.ndarray,
    one_line_sparse: np.ndarray,
    row_fracs: Iterable[float],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    height, _ = gt.shape
    columns = np.unique(np.where(one_line_sparse > 0)[1])
    if columns.size < 5:
        raise RuntimeError(f"Only {columns.size} source x-columns")
    rows = [int(round(fraction * (height - 1))) for fraction in row_fracs]
    if len(set(rows)) != 4:
        raise ValueError(f"Duplicate projected four-line rows: {rows}")
    sparse = np.zeros_like(gt, dtype=np.float32)
    for row in rows:
        usable = (
            valid[row, columns]
            & np.isfinite(gt[row, columns])
            & (gt[row, columns] >= min_depth_m)
            & (gt[row, columns] <= max_depth_m)
        )
        sparse[row, columns[usable]] = gt[row, columns[usable]]
    anchors = sparse > 0
    if int(np.count_nonzero(anchors)) < 12:
        raise RuntimeError(f"Only {int(np.count_nonzero(anchors))} simulated four-line anchors")
    return sparse, anchors, rows, int(columns.size)


def median_align(
    relative: np.ndarray,
    sparse: np.ndarray,
    anchors: np.ndarray,
) -> tuple[np.ndarray, float]:
    usable = anchors & np.isfinite(relative) & (relative > 0)
    ratios = sparse[usable] / relative[usable]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 8:
        raise RuntimeError(f"Only {ratios.size} usable median-scale ratios")
    scale = float(np.median(ratios))
    prediction = relative.astype(np.float32) * scale
    invalid = ~np.isfinite(prediction) | (prediction <= 0)
    if invalid.any():
        raise RuntimeError("DA3 median alignment produced invalid metric depth")
    return prediction.astype(np.float32), scale


def common_masks(
    valid: np.ndarray,
    one_line_anchors: np.ndarray,
    four_line_anchors: np.ndarray,
    margin_px: int,
) -> dict[str, np.ndarray]:
    if margin_px < 0:
        raise ValueError("--outside-margin-px cannot be negative")
    one_distance = distance_transform_edt(~one_line_anchors)
    four_distance = distance_transform_edt(~four_line_anchors)
    return {
        "all_valid": valid.copy(),
        "outside_original_line_common": valid & (one_distance > margin_px),
        "outside_both_patterns_common": (
            valid
            & (one_distance > margin_px)
            & (four_distance > margin_px)
        ),
    }


def metrics(
    prediction: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    usable = mask & np.isfinite(prediction) & (prediction > 0)
    count = int(np.count_nonzero(usable))
    if count == 0:
        raise RuntimeError("An evaluation region contains no usable pixels")
    predicted = prediction[usable].astype(np.float64)
    target = gt[usable].astype(np.float64)
    error = predicted - target
    absolute = np.abs(error)
    ratio = np.maximum(predicted / target, target / predicted)
    return {
        "pixel_count": count,
        "absrel_pct": float(100.0 * np.mean(absolute / target)),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mae_m": float(np.mean(absolute)),
        "bias_m": float(np.mean(error)),
        "delta1_pct": float(100.0 * np.mean(ratio < 1.25)),
        "bad_050_pct": float(100.0 * np.mean(absolute > 0.50)),
        "bad_100_pct": float(100.0 * np.mean(absolute > 1.00)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def configuration_payload(
    args: argparse.Namespace,
    gt_dir: Path,
    sparse_dir: Path,
    da3_dir: Path,
) -> dict[str, Any]:
    return {
        "benchmark": "iBims paired one-line versus four-line DA3 comparison",
        "version": VERSION,
        "gt_dir": str(gt_dir),
        "one_line_dir": str(sparse_dir),
        "cached_da3_dir": str(da3_dir),
        "row_fracs": list(args.row_fracs),
        "sensor_min_depth_m": args.sensor_min_depth_m,
        "sensor_max_depth_m": args.sensor_max_depth_m,
        "eval_max_depth_m": args.eval_max_depth_m,
        "outside_margin_px": args.outside_margin_px,
        "poisson_rtol": args.rtol,
        "poisson_maxiter": args.maxiter,
        "four_line_horizontal_sampling": "unique x-columns present in each v2.1 one-line sensor map",
        "four_line_vertical_sampling": "fixed normalized image rows",
        "one_pixel_per_four_line_return": True,
        "noise": "none",
        "dense_gt_use": "simulate four-line ranges and evaluate only; never post-inference alignment outside sparse anchors",
        "primary_comparison_region": "outside_original_line_common",
        "primary_metrics": ["rmse_m", "absrel_pct"],
        "common_mask_rule": "each paired 1-line/4-line comparison uses exactly the same evaluation pixels",
    }


def configuration_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_or_write_protocol(path: Path, payload: dict[str, Any], resume: bool) -> None:
    payload = dict(payload)
    payload["configuration_sha256"] = configuration_hash(payload)
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        old_hash = old.get("configuration_sha256")
        if old_hash != payload["configuration_sha256"]:
            raise RuntimeError(
                f"Output directory contains a different experiment configuration: {path}. "
                "Use a new output directory."
            )
        if not resume:
            raise FileExistsError(
                f"Existing experiment found at {path.parent}; pass --resume or use a new output directory"
            )
        return
    atomic_json(path, payload)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["region"]))].append(row)
    result: list[dict[str, Any]] = []
    metric_names = tuple(LOWER_IS_BETTER)
    for (method, region), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "region": region,
            "region_label": REGION_LABELS[region],
            "scene_count": len(group),
            "mean_pixel_count": float(np.mean([float(row["pixel_count"]) for row in group])),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in group], dtype=np.float64)
            item[f"mean_{metric_name}"] = float(np.mean(values))
            item[f"median_{metric_name}"] = float(np.median(values))
        result.append(item)
    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if samples <= 0:
        return math.nan, math.nan
    chunk_size = max(1, min(samples, 1000))
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        take = min(chunk_size, remaining)
        indices = rng.integers(0, values.size, size=(take, values.size))
        means.append(np.mean(values[indices], axis=1))
        remaining -= take
    distribution = np.concatenate(means)
    low, high = np.quantile(distribution, (0.025, 0.975))
    return float(low), float(high)


def paired_improvements(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {
        (str(row["scene"]), str(row["region"]), str(row["method"])): row
        for row in rows
    }
    scenes = sorted({str(row["scene"]) for row in rows})
    result: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for pair_name, one_method, four_method in PAIR_SPECS:
        for region in REGION_LABELS:
            for metric_name in ("rmse_m", "absrel_pct", "mae_m", "delta1_pct"):
                one_values: list[float] = []
                four_values: list[float] = []
                for scene in scenes:
                    one_key = (scene, region, one_method)
                    four_key = (scene, region, four_method)
                    if one_key in lookup and four_key in lookup:
                        one_values.append(float(lookup[one_key][metric_name]))
                        four_values.append(float(lookup[four_key][metric_name]))
                one = np.asarray(one_values, dtype=np.float64)
                four = np.asarray(four_values, dtype=np.float64)
                if one.size == 0:
                    continue
                delta = four - one
                lower_is_better = LOWER_IS_BETTER[metric_name]
                if lower_is_better:
                    improvement = one - four
                    denominator = float(np.mean(one))
                    wins = four < one
                else:
                    improvement = four - one
                    denominator = abs(float(np.mean(one)))
                    wins = four > one
                low, high = bootstrap_mean_ci(improvement, bootstrap_samples, rng)
                result.append(
                    {
                        "pipeline": pair_name,
                        "region": region,
                        "region_label": REGION_LABELS[region],
                        "metric": metric_name,
                        "scene_count": int(one.size),
                        "one_line_mean": float(np.mean(one)),
                        "four_line_mean": float(np.mean(four)),
                        "four_minus_one_mean": float(np.mean(delta)),
                        "improvement_mean": float(np.mean(improvement)),
                        "relative_improvement_pct": (
                            100.0 * float(np.mean(improvement)) / denominator
                            if denominator > 0
                            else math.nan
                        ),
                        "four_line_win_rate_pct": float(100.0 * np.mean(wins)),
                        "improvement_bootstrap_ci95_low": low,
                        "improvement_bootstrap_ci95_high": high,
                        "better_direction": "lower" if lower_is_better else "higher",
                    }
                )
    return result


def improvement_lookup(
    paired_rows: list[dict[str, Any]],
    pipeline: str,
    region: str,
    metric: str,
) -> dict[str, Any]:
    for row in paired_rows:
        if (
            row["pipeline"] == pipeline
            and row["region"] == region
            and row["metric"] == metric
        ):
            return row
    raise KeyError((pipeline, region, metric))


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    expected_scenes: int,
) -> None:
    scenes = sorted({str(row["scene"]) for row in rows})
    status = "COMPLETE" if len(scenes) == expected_scenes else "PROVISIONAL / PARTIAL"
    aggregate_lookup = {
        (row["method"], row["region"]): row for row in summary_rows
    }
    lines = [
        "# iBims paired one-line versus four-line indoor result",
        "",
        f"**Status:** {status} — {len(scenes)}/{expected_scenes} scenes",
        "",
        "- Backbone: cached DA3-SMALL relative depth (identical for both conditions).",
        "- Alignment: global median metric scale (identical algorithm).",
        "- Refinement: validated existing Poisson (identical algorithm and parameters).",
        "- One-line input: established iBims v2.1/32 m sensor maps.",
        "- Four-line input: rows 20%, 40%, 60%, and 80%; same horizontal x-columns; one pixel per valid return; no noise.",
        "- Dense GT is used only to simulate the hypothetical four-line returns and evaluate predictions.",
        "- Every paired comparison uses a common evaluation mask.",
        "",
        "## Primary complete-system result: DA3 + median + Poisson",
        "",
        "| Common evaluation region | 1-line RMSE | 4-line RMSE | RMSE improvement | 1-line AbsRel | 4-line AbsRel | AbsRel improvement | RMSE win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in REGION_LABELS:
        rmse = improvement_lookup(paired_rows, "poisson", region, "rmse_m")
        absrel = improvement_lookup(paired_rows, "poisson", region, "absrel_pct")
        lines.append(
            f"| {REGION_LABELS[region]} | {rmse['one_line_mean']:.4f} m | "
            f"{rmse['four_line_mean']:.4f} m | {rmse['relative_improvement_pct']:+.2f}% | "
            f"{absrel['one_line_mean']:.3f}% | {absrel['four_line_mean']:.3f}% | "
            f"{absrel['relative_improvement_pct']:+.2f}% | {rmse['four_line_win_rate_pct']:.1f}% |"
        )
    primary_rmse = improvement_lookup(
        paired_rows, "poisson", "outside_original_line_common", "rmse_m"
    )
    primary_absrel = improvement_lookup(
        paired_rows, "poisson", "outside_original_line_common", "absrel_pct"
    )
    lines.extend(
        [
            "",
            "### Paired uncertainty on the primary outside-original-line region",
            "",
            f"- RMSE improvement (1 line − 4 lines): {primary_rmse['improvement_mean']:+.4f} m; "
            f"95% scene-bootstrap CI [{primary_rmse['improvement_bootstrap_ci95_low']:+.4f}, "
            f"{primary_rmse['improvement_bootstrap_ci95_high']:+.4f}] m.",
            f"- AbsRel improvement (1 line − 4 lines): {primary_absrel['improvement_mean']:+.3f} percentage points; "
            f"95% scene-bootstrap CI [{primary_absrel['improvement_bootstrap_ci95_low']:+.3f}, "
            f"{primary_absrel['improvement_bootstrap_ci95_high']:+.3f}] points.",
            "",
            "## Alignment-only ablation: DA3 + median",
            "",
            "| Common evaluation region | 1-line RMSE | 4-line RMSE | RMSE improvement | 1-line AbsRel | 4-line AbsRel | AbsRel improvement |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for region in REGION_LABELS:
        rmse = improvement_lookup(paired_rows, "median", region, "rmse_m")
        absrel = improvement_lookup(paired_rows, "median", region, "absrel_pct")
        lines.append(
            f"| {REGION_LABELS[region]} | {rmse['one_line_mean']:.4f} m | "
            f"{rmse['four_line_mean']:.4f} m | {rmse['relative_improvement_pct']:+.2f}% | "
            f"{absrel['one_line_mean']:.3f}% | {absrel['four_line_mean']:.3f}% | "
            f"{absrel['relative_improvement_pct']:+.2f}% |"
        )
    one_anchors = np.asarray([float(row["one_line_anchor_count"]) for row in fit_rows])
    four_anchors = np.asarray([float(row["four_line_anchor_count"]) for row in fit_rows])
    one_repairs = int(sum(int(float(row.get("one_poisson_repaired_pixels", 0))) for row in fit_rows))
    four_repairs = int(sum(int(float(row.get("four_poisson_repaired_pixels", 0))) for row in fit_rows))
    lines.extend(
        [
            "",
            "## Input and numerical diagnostics",
            "",
            f"- One-line anchors per scene: mean {np.mean(one_anchors):.1f}, median {np.median(one_anchors):.0f}.",
            f"- Four-line anchors per scene: mean {np.mean(four_anchors):.1f}, median {np.median(four_anchors):.0f}.",
            f"- Invalid Poisson pixels repaired from the corresponding median prior: one-line {one_repairs}; four-line {four_repairs}.",
            "",
            "## Interpretation rule",
            "",
            "A positive improvement percentage means the four-line input reduced error. "
            "The primary region is outside the original one-line support because it directly tests whether added vertical coverage improves the previously unobserved part of the image. "
            "The outside-both-patterns region tests whether the benefit also propagates to pixels far from every input line in either condition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_panel(
    role: str,
    scene: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    one_sparse: np.ndarray,
    four_sparse: np.ndarray,
    one_prediction: np.ndarray,
    four_prediction: np.ndarray,
    one_metrics: dict[str, Any],
    four_metrics: dict[str, Any],
    output: Path,
    depth_max_m: float,
    gain_max_m: float,
) -> None:
    gt_show = np.where(valid, gt, np.nan)
    one_show = np.where(valid, one_prediction, np.nan)
    four_show = np.where(valid, four_prediction, np.nan)
    gain = np.where(
        valid,
        np.abs(one_prediction - gt) - np.abs(four_prediction - gt),
        np.nan,
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    y, x = np.where(one_sparse > 0)
    axes[0, 0].scatter(x, y, s=4, c="cyan")
    axes[0, 0].set_title(f"RGB + original 1 line ({len(x)} anchors)")
    axes[0, 1].imshow(rgb)
    y, x = np.where(four_sparse > 0)
    axes[0, 1].scatter(x, y, s=3, c="cyan")
    axes[0, 1].set_title(f"RGB + simulated 4 lines ({len(x)} anchors)")
    depth_image = axes[0, 2].imshow(gt_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[0, 2].set_title("iBims metric GT")
    axes[1, 0].imshow(one_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[1, 0].set_title(
        "1-line DA3 + median + Poisson\n"
        f"outside-original RMSE {float(one_metrics['rmse_m']):.3f} m | "
        f"AbsRel {float(one_metrics['absrel_pct']):.2f}%"
    )
    axes[1, 1].imshow(four_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[1, 1].set_title(
        "4-line DA3 + median + Poisson\n"
        f"outside-original RMSE {float(four_metrics['rmse_m']):.3f} m | "
        f"AbsRel {float(four_metrics['absrel_pct']):.2f}%"
    )
    gain_image = axes[1, 2].imshow(
        gain,
        cmap="RdBu_r",
        vmin=-gain_max_m,
        vmax=gain_max_m,
    )
    axes[1, 2].set_title("Absolute-error reduction\nred = 4 lines better; blue = worse")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.colorbar(
        depth_image,
        ax=[axes[0, 2], axes[1, 0], axes[1, 1]],
        shrink=0.75,
        label="Depth (m)",
    )
    figure.colorbar(
        gain_image,
        ax=axes[1, 2],
        shrink=0.75,
        label="|error 1-line| − |error 4-line| (m)",
    )
    rmse_improvement = 100.0 * (
        float(one_metrics["rmse_m"]) - float(four_metrics["rmse_m"])
    ) / float(one_metrics["rmse_m"])
    figure.suptitle(
        f"{role.upper()} FOUR-LINE IMPROVEMENT — {scene}\n"
        f"outside-original-line RMSE improvement {rmse_improvement:+.2f}%"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def create_example_panels(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    gt_dir: Path,
    sparse_dir: Path,
    prediction_root: Path,
) -> None:
    lookup = {
        (str(row["scene"]), str(row["region"]), str(row["method"])): row
        for row in rows
    }
    scene_scores: list[tuple[float, str]] = []
    for scene in sorted({str(row["scene"]) for row in rows}):
        one = lookup[(scene, "outside_original_line_common", "one_line_poisson")]
        four = lookup[(scene, "outside_original_line_common", "four_line_poisson")]
        score = float(one["rmse_m"]) - float(four["rmse_m"])
        scene_scores.append((score, scene))
    scene_scores.sort()
    if not scene_scores:
        return
    choices = [
        ("worst", scene_scores[0][1]),
        ("typical", scene_scores[len(scene_scores) // 2][1]),
        ("best", scene_scores[-1][1]),
    ]
    output = args.output_dir / "examples_best_typical_worst"
    for role, scene in choices:
        gt, valid, rgb = load_ibims(gt_dir / f"{scene}.mat")
        if args.eval_max_depth_m > 0:
            valid &= gt <= args.eval_max_depth_m
        raw_sparse = load_npy(npy_path(sparse_dir, scene), gt.shape)
        one_sparse, _ = sanitize_one_line(
            raw_sparse,
            valid,
            args.sensor_min_depth_m,
            args.sensor_max_depth_m,
        )
        four_sparse, _, _, _ = simulate_four_lines(
            gt,
            valid,
            one_sparse,
            args.row_fracs,
            args.sensor_min_depth_m,
            args.sensor_max_depth_m,
        )
        one_prediction = load_npy(
            prediction_root / "one_line_poisson" / f"{scene}.npy", gt.shape
        )
        four_prediction = load_npy(
            prediction_root / "four_line_poisson" / f"{scene}.npy", gt.shape
        )
        comparison_panel(
            role,
            scene,
            rgb,
            gt,
            valid,
            one_sparse,
            four_sparse,
            one_prediction,
            four_prediction,
            lookup[(scene, "outside_original_line_common", "one_line_poisson")],
            lookup[(scene, "outside_original_line_common", "four_line_poisson")],
            output / f"{role}__{scene}__1line_vs_4line.png",
            args.plot_max_depth_m,
            args.plot_gain_max_m,
        )


def main() -> None:
    args = arguments()
    args.da3_root = args.da3_root.expanduser().resolve()
    args.a2f_root = args.a2f_root.expanduser().resolve()
    if args.data_root is not None:
        args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(not 0 < fraction < 1 for fraction in args.row_fracs):
        raise ValueError("--row-fracs values must be strictly between 0 and 1")
    if sorted(args.row_fracs) != list(args.row_fracs) or len(set(args.row_fracs)) != 4:
        raise ValueError("--row-fracs must contain four unique increasing values")
    if not 0 <= args.outside_margin_px:
        raise ValueError("--outside-margin-px cannot be negative")
    if args.sensor_min_depth_m <= 0 or args.sensor_max_depth_m <= args.sensor_min_depth_m:
        raise ValueError("Invalid sensor depth range")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.expected_scenes <= 0:
        raise ValueError("--expected-scenes must be positive")

    roots = search_roots(args)
    gt_dir = resolve_gt(args.gt_dir, roots)
    sparse_dir = resolve_npy_dir(
        args.one_line_dir,
        roots,
        ("v2_1_sensor", "v2_sensor"),
        "v2.1 one-line sparse maps",
    )
    da3_dir = resolve_npy_dir(
        args.da3_dir,
        roots,
        ("da3_bridge_all",),
        "cached DA3-SMALL relative predictions",
    )
    print(
        f"GT: {gt_dir}\n"
        f"one-line v2.1: {sparse_dir}\n"
        f"cached DA3: {da3_dir}\n"
        f"output: {args.output_dir}",
        flush=True,
    )
    protocol = configuration_payload(args, gt_dir, sparse_dir, da3_dir)
    validate_or_write_protocol(args.output_dir / "protocol.json", protocol, args.resume)
    poisson = load_poisson(args.da3_root)

    scenes = sorted(path.stem for path in gt_dir.glob("*.mat"))
    if args.scene:
        scenes = [scene for scene in scenes if scene == args.scene]
    if args.limit is not None:
        scenes = scenes[: args.limit]
    if not scenes:
        raise RuntimeError("No iBims scenes selected")
    if args.scene is None and args.limit is None and len(scenes) != args.expected_scenes:
        raise RuntimeError(
            f"Expected {args.expected_scenes} iBims scenes but found {len(scenes)} in {gt_dir}"
        )

    metrics_path = args.output_dir / "per_scene_paired_metrics.csv"
    fit_path = args.output_dir / "per_scene_fit_diagnostics.csv"
    metric_rows: list[dict[str, Any]] = list(read_csv(metrics_path)) if args.resume else []
    fit_rows: list[dict[str, Any]] = list(read_csv(fit_path)) if args.resume else []
    prediction_root = args.output_dir / "predictions_m"
    for method in METHOD_LABELS:
        (prediction_root / method).mkdir(parents=True, exist_ok=True)
    expected_keys = {
        (method, region) for method in METHOD_LABELS for region in REGION_LABELS
    }

    for index, scene in enumerate(scenes, start=1):
        old_keys = {
            (str(row.get("method")), str(row.get("region")))
            for row in metric_rows
            if row.get("scene") == scene
        }
        predictions_exist = all(
            (prediction_root / method / f"{scene}.npy").is_file()
            for method in METHOD_LABELS
        )
        if args.resume and expected_keys <= old_keys and predictions_exist:
            print(f"[{index:3d}/{len(scenes)}] {scene} resume-skip", flush=True)
            continue

        gt, valid, _ = load_ibims(gt_dir / f"{scene}.mat")
        if args.eval_max_depth_m > 0:
            valid &= gt <= args.eval_max_depth_m
        raw_sparse = load_npy(npy_path(sparse_dir, scene), gt.shape)
        relative = load_npy(npy_path(da3_dir, scene, da3=True), gt.shape)
        one_sparse, one_anchors = sanitize_one_line(
            raw_sparse,
            valid,
            args.sensor_min_depth_m,
            args.sensor_max_depth_m,
        )
        four_sparse, four_anchors, line_rows, source_column_count = simulate_four_lines(
            gt,
            valid,
            one_sparse,
            args.row_fracs,
            args.sensor_min_depth_m,
            args.sensor_max_depth_m,
        )
        one_median, one_scale = median_align(relative, one_sparse, one_anchors)
        four_median, four_scale = median_align(relative, four_sparse, four_anchors)
        one_poisson, one_diagnostics = call_poisson(
            poisson,
            one_median,
            one_sparse,
            one_anchors,
            args.rtol,
            args.maxiter,
        )
        four_poisson, four_diagnostics = call_poisson(
            poisson,
            four_median,
            four_sparse,
            four_anchors,
            args.rtol,
            args.maxiter,
        )
        predictions = {
            "one_line_median": one_median,
            "four_line_median": four_median,
            "one_line_poisson": one_poisson,
            "four_line_poisson": four_poisson,
        }
        masks = common_masks(
            valid,
            one_anchors,
            four_anchors,
            args.outside_margin_px,
        )
        new_rows: list[dict[str, Any]] = []
        for method, prediction in predictions.items():
            np.save(
                prediction_root / method / f"{scene}.npy",
                prediction.astype(np.float32),
            )
            for region, mask in masks.items():
                new_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "region": region,
                        "region_label": REGION_LABELS[region],
                        "one_line_anchor_count": int(np.count_nonzero(one_anchors)),
                        "four_line_anchor_count": int(np.count_nonzero(four_anchors)),
                        **metrics(prediction, gt, mask),
                    }
                )
        metric_rows = [row for row in metric_rows if row.get("scene") != scene]
        metric_rows.extend(new_rows)
        metric_rows.sort(
            key=lambda row: (str(row["scene"]), str(row["method"]), str(row["region"]))
        )
        fit_row = {
            "scene": scene,
            "one_line_anchor_count": int(np.count_nonzero(one_anchors)),
            "four_line_anchor_count": int(np.count_nonzero(four_anchors)),
            "source_x_column_count": source_column_count,
            "four_line_rows": ";".join(map(str, line_rows)),
            "one_line_median_scale": one_scale,
            "four_line_median_scale": four_scale,
            "one_poisson_repaired_pixels": int(
                one_diagnostics.get("invalid_pixels_repaired_from_median", 0)
            ),
            "four_poisson_repaired_pixels": int(
                four_diagnostics.get("invalid_pixels_repaired_from_median", 0)
            ),
            "one_poisson_diagnostics": json.dumps(
                one_diagnostics, default=str, sort_keys=True
            ),
            "four_poisson_diagnostics": json.dumps(
                four_diagnostics, default=str, sort_keys=True
            ),
        }
        fit_rows = [row for row in fit_rows if row.get("scene") != scene]
        fit_rows.append(fit_row)
        fit_rows.sort(key=lambda row: str(row["scene"]))
        write_csv(metrics_path, metric_rows)
        write_csv(fit_path, fit_rows)

        current = {
            (row["method"], row["region"]): row
            for row in new_rows
        }
        one_primary = current[("one_line_poisson", "outside_original_line_common")]
        four_primary = current[("four_line_poisson", "outside_original_line_common")]
        improvement_pct = 100.0 * (
            float(one_primary["rmse_m"]) - float(four_primary["rmse_m"])
        ) / float(one_primary["rmse_m"])
        print(
            f"[{index:3d}/{len(scenes)}] {scene} "
            f"anchors {int(np.count_nonzero(one_anchors))}->{int(np.count_nonzero(four_anchors))}; "
            f"outside-original RMSE {float(one_primary['rmse_m']):.3f}->{float(four_primary['rmse_m']):.3f} m "
            f"({improvement_pct:+.2f}%)",
            flush=True,
        )

    selected = set(scenes)
    chosen_metrics = [row for row in metric_rows if str(row["scene"]) in selected]
    chosen_fit = [row for row in fit_rows if str(row["scene"]) in selected]
    required_count = len(scenes) * len(METHOD_LABELS) * len(REGION_LABELS)
    if len(chosen_metrics) != required_count:
        raise RuntimeError(
            f"Incomplete result table: {len(chosen_metrics)} rows; expected {required_count}"
        )
    summary_rows = aggregate(chosen_metrics)
    paired_rows = paired_improvements(
        chosen_metrics,
        args.bootstrap_samples,
        args.seed,
    )
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_improvement.csv", paired_rows)
    write_report(
        args.output_dir / "comparison_report.md",
        chosen_metrics,
        summary_rows,
        paired_rows,
        chosen_fit,
        args.expected_scenes,
    )
    if not args.skip_panels:
        create_example_panels(
            args,
            chosen_metrics,
            gt_dir,
            sparse_dir,
            prediction_root,
        )
    print("\n===== PAIRED ONE-LINE VS FOUR-LINE RESULT =====\n")
    print((args.output_dir / "comparison_report.md").read_text(encoding="utf-8"))
    print(f"Full report: {args.output_dir / 'comparison_report.md'}")
    if not args.skip_panels:
        print(f"Visual comparisons: {args.output_dir / 'examples_best_typical_worst'}")


if __name__ == "__main__":
    main()
