#!/usr/bin/env python3
"""Fair three-scene iBims comparison: DA3 versus Any2Full with identical four-line input.

The script has two stages:

``prepare``
    Export the exact maximum-coverage four-line sparse maps already saved by
    ``ibims_current_vs_max_coverage_3scene.py``.  It also exports matching RGB
    images for the repository's existing ``run_any2full.py`` runner and keeps
    GT/masks in separate evaluation-only directories.  SHA-256 hashes audit
    that the sparse NPY supplied to Any2Full is byte-identical to the saved map
    that produced the DA3 result.

``evaluate``
    Compare the native Any2Full-vits predictions with the already-saved
    DA3-SMALL + global median + validated Poisson predictions.  Both methods
    are scored on identical iBims masks, including the primary region outside
    the original one-line support.  Per-scene depth/error panels and the same
    GT-selected upper/middle/lower surface probes are produced.

Dense GT is never supplied to either model except for the four sparse ranges
that are already fixed in the saved input NPY.  This is a provisional
three-scene complete-pipeline comparison, not a state-of-the-art claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


VERSION = "1.0"
MAX_COVERAGE_FRACS = (0.125, 0.375, 0.625, 0.875)
METHODS = {
    "da3_max_coverage": "DA3-SMALL + median + Poisson",
    "any2full_max_coverage": "Any2Full-vits (native pipeline)",
}
REGIONS = {
    "all_valid": "All valid GT pixels",
    "outside_original_line": "Outside original 1-line support (primary)",
    "outside_shared_four_line": "Outside the shared 4-line input support",
}
LOWER_METRICS = ("rmse_m", "absrel_pct", "mae_m")


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Export the exact saved four-line input for the Any2Full runner.",
    )
    prepare.add_argument("--placement-output", type=Path, required=True)
    prepare.add_argument("--prepared-data-root", type=Path, required=True)
    prepare.add_argument("--expected-scenes", type=positive_int, default=3)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Compare Any2Full predictions against the saved DA3 predictions.",
    )
    evaluate.add_argument("--placement-output", type=Path, required=True)
    evaluate.add_argument("--prepared-data-root", type=Path, required=True)
    evaluate.add_argument("--any2full-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--expected-scenes", type=positive_int, default=3)
    evaluate.add_argument("--bootstrap-samples", type=positive_int, default=10000)
    evaluate.add_argument("--seed", type=int, default=12345)
    evaluate.add_argument("--plot-max-depth-m", type=float, default=10.0)
    evaluate.add_argument("--plot-error-max-m", type=float, default=1.0)
    evaluate.add_argument("--skip-panels", action="store_true")

    subparsers.add_parser("self-test", help="Run lightweight numerical checks.")
    return parser.parse_args()


def import_paired_module(script_dir: Path) -> Any:
    path = script_dir / "ibims_1line_vs_4line_da3_comparison.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required sibling evaluator is missing: {path}\n"
            "Keep this script beside ibims_1line_vs_4line_da3_comparison.py."
        )
    spec = importlib.util.spec_from_file_location("ibims_paired_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("load_ibims", "load_npy", "sanitize_one_line", "metrics")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"Sibling evaluator is missing helpers: {missing}")
    return module


def ensure_directory(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_protocol_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise KeyError(f"Protocol does not contain {label}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Protocol {label} no longer exists: {path}")
    return path


def placement_context(
    placement_output: Path,
    expected_scenes: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    placement = read_json(placement_output / "protocol.json")
    fractions = tuple(float(value) for value in placement.get("max_coverage_row_fracs", []))
    if len(fractions) != 4 or not np.allclose(fractions, MAX_COVERAGE_FRACS):
        raise RuntimeError(
            "Placement output is not the maximum-coverage 12.5/37.5/62.5/87.5% "
            f"experiment: {fractions}"
        )
    reference_output = resolve_protocol_path(
        placement.get("reference_output"), "reference_output"
    )
    reference = read_json(reference_output / "protocol.json")
    scenes = [str(value) for value in placement.get("scenes", [])]
    if len(scenes) != expected_scenes or len(set(scenes)) != expected_scenes:
        raise RuntimeError(
            f"Placement protocol contains {len(scenes)} unique scenes; "
            f"expected {expected_scenes}: {scenes}"
        )
    metric_rows = read_csv(placement_output / "per_scene_placement_metrics.csv")
    metric_scenes = sorted({row["scene"] for row in metric_rows})
    if sorted(scenes) != metric_scenes:
        raise RuntimeError(
            "Placement metric scenes do not match protocol scenes: "
            f"protocol={sorted(scenes)}, metrics={metric_scenes}"
        )
    return placement, reference, scenes


def exact_anchor_count(
    placement_output: Path,
    scene: str,
) -> int:
    rows = read_csv(placement_output / "per_scene_placement_metrics.csv")
    matches = [
        row
        for row in rows
        if row.get("scene") == scene
        and row.get("method") == "max_coverage_4line"
        and row.get("region") == "all_valid"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot find one max-coverage metric row for {scene}")
    return int(round(float(matches[0]["anchor_count"])))


def assert_expected_files(directory: Path, expected: set[str], suffix: str) -> None:
    present = {path.name for path in directory.glob(f"*{suffix}") if path.is_file()}
    unexpected = sorted(present - expected)
    if unexpected:
        raise RuntimeError(
            f"{directory} contains unexpected {suffix} files: {unexpected}. "
            "Use a fresh prepared-data directory so Any2Full sees only these scenes."
        )


def prepare_data(args: argparse.Namespace) -> None:
    placement_output = ensure_directory(args.placement_output, "placement output")
    prepared_root = args.prepared_data_root.expanduser().resolve()
    placement, reference, scenes = placement_context(
        placement_output, args.expected_scenes
    )
    paired = import_paired_module(Path(__file__).resolve().parent)
    gt_dir = resolve_protocol_path(reference.get("gt_dir"), "gt_dir")
    eval_max = float(reference.get("eval_max_depth_m", 0.0))
    sensor_min = float(reference.get("sensor_min_depth_m", 0.10))
    sensor_max = float(reference.get("sensor_max_depth_m", 32.0))

    directories = {
        "rgb": prepared_root / "rgb",
        "sparse": prepared_root / "sparse_input_m",
        "gt": prepared_root / "evaluation_only" / "gt_m",
        "valid": prepared_root / "evaluation_only" / "valid_mask",
        "one": prepared_root / "evaluation_only" / "one_line_anchor_mask",
        "four": prepared_root / "evaluation_only" / "four_line_anchor_mask",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    expected_png = {f"{scene}.png" for scene in scenes}
    expected_npy = {f"{scene}.npy" for scene in scenes}
    assert_expected_files(directories["rgb"], expected_png, ".png")
    assert_expected_files(directories["sparse"], expected_npy, ".npy")

    manifest_rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, start=1):
        gt, valid, rgb = paired.load_ibims(gt_dir / f"{scene}.mat")
        if eval_max > 0:
            valid &= gt <= eval_max
        source_sparse_path = (
            placement_output / "sparse_inputs_m" / "max_coverage_4line" / f"{scene}.npy"
        )
        one_sparse_path = (
            placement_output / "sparse_inputs_m" / "one_line" / f"{scene}.npy"
        )
        da3_path = (
            placement_output / "predictions_m" / "max_coverage_4line" / f"{scene}.npy"
        )
        for path in (source_sparse_path, one_sparse_path, da3_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        sparse = paired.load_npy(source_sparse_path, gt.shape)
        one_sparse = paired.load_npy(one_sparse_path, gt.shape)
        da3 = paired.load_npy(da3_path, gt.shape)
        four_anchors = np.isfinite(sparse) & (sparse >= sensor_min) & (sparse <= sensor_max)
        if np.any((sparse != 0) & ~four_anchors):
            raise RuntimeError(f"{scene}: exact four-line map contains invalid nonzero values")
        one_clean, one_anchors = paired.sanitize_one_line(
            one_sparse, valid, sensor_min, sensor_max
        )
        if not np.array_equal(one_clean > 0, one_sparse > 0):
            raise RuntimeError(f"{scene}: saved one-line map is not already sanitized")
        if np.any(four_anchors & ~valid):
            raise RuntimeError(f"{scene}: exact four-line anchors include invalid GT pixels")
        anchor_count = int(np.count_nonzero(four_anchors))
        recorded_count = exact_anchor_count(placement_output, scene)
        if anchor_count != recorded_count:
            raise RuntimeError(
                f"{scene}: sparse file has {anchor_count} anchors but the recorded DA3 "
                f"run reports {recorded_count}"
            )
        if not np.all(np.isfinite(da3[valid]) & (da3[valid] > 0)):
            raise RuntimeError(f"{scene}: saved DA3 prediction is invalid on evaluation pixels")

        rgb_path = directories["rgb"] / f"{scene}.png"
        sparse_path = directories["sparse"] / f"{scene}.npy"
        gt_path = directories["gt"] / f"{scene}.npy"
        valid_path = directories["valid"] / f"{scene}.npy"
        one_path = directories["one"] / f"{scene}.npy"
        four_path = directories["four"] / f"{scene}.npy"
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        shutil.copy2(source_sparse_path, sparse_path)
        np.save(gt_path, gt.astype(np.float32))
        np.save(valid_path, valid.astype(np.bool_))
        np.save(one_path, one_anchors.astype(np.bool_))
        np.save(four_path, four_anchors.astype(np.bool_))
        if sha256_file(source_sparse_path) != sha256_file(sparse_path):
            raise RuntimeError(f"{scene}: sparse input changed during export")

        row_values = sorted(int(value) for value in np.unique(np.where(four_anchors)[0]))
        manifest_rows.append(
            {
                "scene": scene,
                "rgb": str(rgb_path.relative_to(prepared_root)),
                "sparse_input_m": str(sparse_path.relative_to(prepared_root)),
                "gt_m_evaluation_only": str(gt_path.relative_to(prepared_root)),
                "valid_mask_evaluation_only": str(valid_path.relative_to(prepared_root)),
                "one_line_mask_evaluation_only": str(one_path.relative_to(prepared_root)),
                "four_line_mask_evaluation_only": str(four_path.relative_to(prepared_root)),
                "source_da3_prediction": str(da3_path),
                "image_height": gt.shape[0],
                "image_width": gt.shape[1],
                "physical_anchor_count": anchor_count,
                "nonempty_rows": ";".join(str(value) for value in row_values),
                "source_sparse_sha256": sha256_file(source_sparse_path),
                "exported_sparse_sha256": sha256_file(sparse_path),
                "rgb_sha256": sha256_file(rgb_path),
            }
        )
        print(
            f"[{index}/{len(scenes)}] {scene}: exported {anchor_count} exact physical anchors",
            flush=True,
        )

    write_csv(prepared_root / "manifest.csv", manifest_rows)
    protocol = {
        "benchmark": "iBims exact maximum-coverage four-line input for DA3 versus Any2Full",
        "version": VERSION,
        "source_placement_output": str(placement_output),
        "source_placement_protocol_sha256": sha256_file(placement_output / "protocol.json"),
        "source_reference_protocol_sha256": sha256_file(
            resolve_protocol_path(placement.get("reference_output"), "reference_output")
            / "protocol.json"
        ),
        "scenes": scenes,
        "row_fractions": list(MAX_COVERAGE_FRACS),
        "outside_margin_px": int(reference.get("outside_margin_px", 10)),
        "sensor_min_depth_m": sensor_min,
        "sensor_max_depth_m": sensor_max,
        "eval_max_depth_m": eval_max,
        "model_visible_directories": ["rgb", "sparse_input_m"],
        "evaluation_only_directory": "evaluation_only",
        "input_identity_rule": (
            "sparse_input_m files are byte-identical copies of the exact maps used "
            "for the saved DA3 maximum-coverage prediction"
        ),
        "dense_gt_use": (
            "GT is evaluation-only here; the four values per sampled column were "
            "already fixed by the prior simulation"
        ),
        "noise": "none",
    }
    atomic_json(prepared_root / "protocol.json", protocol)
    print("\nPREPARATION PASSED")
    print(f"RGB for Any2Full: {directories['rgb']}")
    print(f"Exact sparse depth for Any2Full: {directories['sparse']}")
    print(f"Physical scenes: {len(scenes)}")


def load_manifest(prepared_root: Path, expected_scenes: int) -> list[dict[str, str]]:
    rows = read_csv(prepared_root / "manifest.csv")
    scenes = [row.get("scene", "") for row in rows]
    if len(rows) != expected_scenes or len(set(scenes)) != expected_scenes or "" in scenes:
        raise RuntimeError(
            f"Prepared manifest has {len(rows)} rows/{len(set(scenes))} unique scenes; "
            f"expected {expected_scenes}"
        )
    return rows


def load_npy_2d(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.squeeze(np.load(path))
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D NPY at {path}; got {value.shape}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{path}: {value.shape} != {shape}")
    return value


def load_prediction(path: Path, shape: tuple[int, int], valid: np.ndarray) -> np.ndarray:
    value = load_npy_2d(path, shape).astype(np.float32)
    invalid = valid & (~np.isfinite(value) | (value <= 0))
    if np.any(invalid):
        raise RuntimeError(
            f"{path}: {int(np.count_nonzero(invalid))} invalid prediction pixels "
            "inside the evaluation mask; no repair or GT clipping is permitted"
        )
    return value


def comparison_masks(
    valid: np.ndarray,
    one_anchors: np.ndarray,
    four_anchors: np.ndarray,
    margin_px: int,
) -> dict[str, np.ndarray]:
    if margin_px < 0:
        raise ValueError("outside margin cannot be negative")
    one_distance = distance_transform_edt(~one_anchors)
    four_distance = distance_transform_edt(~four_anchors)
    masks = {
        "all_valid": valid.copy(),
        "outside_original_line": valid & (one_distance > margin_px),
        "outside_shared_four_line": valid & (four_distance > margin_px),
    }
    for name, mask in masks.items():
        if not np.any(mask):
            raise RuntimeError(f"Evaluation region is empty: {name}")
    return masks


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["region"]))].append(row)
    result: list[dict[str, Any]] = []
    for (method, region), group in sorted(groups.items()):
        result.append(
            {
                "method": method,
                "method_label": METHODS[method],
                "region": region,
                "region_label": REGIONS[region],
                "scene_count": len(group),
                "mean_rmse_m": float(np.mean([float(row["rmse_m"]) for row in group])),
                "mean_absrel_pct": float(
                    np.mean([float(row["absrel_pct"]) for row in group])
                ),
                "mean_mae_m": float(np.mean([float(row["mae_m"]) for row in group])),
                "mean_delta1_pct": float(
                    np.mean([float(row["delta1_pct"]) for row in group])
                ),
            }
        )
    return result


def bootstrap_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_improvements(
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["scene"]), str(row["method"]), str(row["region"])): row
        for row in rows
    }
    scenes = sorted({str(row["scene"]) for row in rows})
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for region in REGIONS:
        for metric in (*LOWER_METRICS, "delta1_pct"):
            da3 = np.asarray(
                [
                    float(lookup[(scene, "da3_max_coverage", region)][metric])
                    for scene in scenes
                ],
                dtype=np.float64,
            )
            any2full = np.asarray(
                [
                    float(lookup[(scene, "any2full_max_coverage", region)][metric])
                    for scene in scenes
                ],
                dtype=np.float64,
            )
            if metric in LOWER_METRICS:
                improvement = da3 - any2full
                wins = any2full < da3
                denominator = float(np.mean(da3))
            else:
                improvement = any2full - da3
                wins = any2full > da3
                denominator = abs(float(np.mean(da3)))
            low, high = bootstrap_ci(improvement, samples, rng)
            result.append(
                {
                    "region": region,
                    "region_label": REGIONS[region],
                    "metric": metric,
                    "scene_count": len(scenes),
                    "da3_mean": float(np.mean(da3)),
                    "any2full_mean": float(np.mean(any2full)),
                    "any2full_improvement_mean": float(np.mean(improvement)),
                    "any2full_relative_improvement_pct": (
                        100.0 * float(np.mean(improvement)) / denominator
                        if denominator > 0
                        else math.nan
                    ),
                    "any2full_win_rate_pct": float(100.0 * np.mean(wins)),
                    "improvement_bootstrap_ci95_low": low,
                    "improvement_bootstrap_ci95_high": high,
                    "positive_means": "Any2Full is better",
                }
            )
    return result


def metric_lookup(
    rows: list[dict[str, Any]], scene: str, method: str, region: str
) -> dict[str, Any]:
    for row in rows:
        if row["scene"] == scene and row["method"] == method and row["region"] == region:
            return row
    raise KeyError((scene, method, region))


def summary_lookup(
    rows: list[dict[str, Any]], method: str, region: str
) -> dict[str, Any]:
    for row in rows:
        if row["method"] == method and row["region"] == region:
            return row
    raise KeyError((method, region))


def improvement_lookup(
    rows: list[dict[str, Any]], region: str, metric: str
) -> dict[str, Any]:
    for row in rows:
        if row["region"] == region and row["metric"] == metric:
            return row
    raise KeyError((region, metric))


def surface_measurement(
    scene: str,
    roi: dict[str, str],
    gt: np.ndarray,
    valid: np.ndarray,
    da3: np.ndarray,
    any2full: np.ndarray,
) -> dict[str, Any]:
    top = int(round(float(roi["top_px"])))
    left = int(round(float(roi["left_px"])))
    height = int(round(float(roi["height_px"])))
    width = int(round(float(roi["width_px"])))
    region = np.zeros_like(valid, dtype=bool)
    region[top : top + height, left : left + width] = True
    mask = valid & region
    target = gt[mask].astype(np.float64)
    da3_values = da3[mask].astype(np.float64)
    a2f_values = any2full[mask].astype(np.float64)
    if target.size == 0:
        raise RuntimeError(f"{scene}/{roi['surface']}: empty diagnostic surface")
    da3_error = da3_values - target
    a2f_error = a2f_values - target
    da3_rmse = float(np.sqrt(np.mean(da3_error * da3_error)))
    a2f_rmse = float(np.sqrt(np.mean(a2f_error * a2f_error)))
    da3_absrel = float(100.0 * np.mean(np.abs(da3_error) / target))
    a2f_absrel = float(100.0 * np.mean(np.abs(a2f_error) / target))
    return {
        "scene": scene,
        "surface": roi["surface"],
        "top_px": top,
        "left_px": left,
        "height_px": height,
        "width_px": width,
        "valid_pixel_count": int(target.size),
        "gt_plane_fit_rmse_m": float(roi["gt_plane_fit_rmse_m"]),
        "gt_mean_m": float(np.mean(target)),
        "gt_median_m": float(np.median(target)),
        "da3_mean_m": float(np.mean(da3_values)),
        "da3_bias_m": float(np.mean(da3_error)),
        "da3_rmse_m": da3_rmse,
        "da3_absrel_pct": da3_absrel,
        "any2full_mean_m": float(np.mean(a2f_values)),
        "any2full_bias_m": float(np.mean(a2f_error)),
        "any2full_rmse_m": a2f_rmse,
        "any2full_absrel_pct": a2f_absrel,
        "any2full_rmse_improvement_m": da3_rmse - a2f_rmse,
        "any2full_rmse_improvement_pct": (
            100.0 * (da3_rmse - a2f_rmse) / da3_rmse if da3_rmse > 0 else math.nan
        ),
        "rmse_winner": "Any2Full" if a2f_rmse < da3_rmse else "DA3",
    }


def depth_panel(
    scene: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    four_anchors: np.ndarray,
    primary_mask: np.ndarray,
    da3: np.ndarray,
    any2full: np.ndarray,
    metric_rows: list[dict[str, Any]],
    output: Path,
    depth_max_m: float,
    error_max_m: float,
) -> None:
    da3_metric = metric_lookup(
        metric_rows, scene, "da3_max_coverage", "outside_original_line"
    )
    a2f_metric = metric_lookup(
        metric_rows, scene, "any2full_max_coverage", "outside_original_line"
    )
    gt_show = np.where(valid, gt, np.nan)
    da3_show = np.where(valid, da3, np.nan)
    a2f_show = np.where(valid, any2full, np.nan)
    da3_error = np.where(valid, np.abs(da3 - gt), np.nan)
    a2f_error = np.where(valid, np.abs(any2full - gt), np.nan)
    gain = da3_error - a2f_error

    figure, axes = plt.subplots(2, 4, figsize=(22, 10.5), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    y, x = np.where(four_anchors)
    axes[0, 0].scatter(x, y, s=3, c="cyan", linewidths=0)
    axes[0, 0].set_title(f"Identical 4-line input ({len(x)} physical anchors)")
    depth_image = axes[0, 1].imshow(
        gt_show, cmap="turbo", vmin=0, vmax=depth_max_m
    )
    axes[0, 1].set_title("iBims metric GT")
    axes[0, 2].imshow(da3_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[0, 2].set_title(
        "DA3 + median + Poisson\n"
        f"RMSE {float(da3_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(da3_metric['absrel_pct']):.2f}%"
    )
    axes[0, 3].imshow(a2f_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[0, 3].set_title(
        "Any2Full-vits native\n"
        f"RMSE {float(a2f_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(a2f_metric['absrel_pct']):.2f}%"
    )
    axes[1, 0].imshow(
        primary_mask.astype(np.uint8),
        cmap=ListedColormap(["black", "white"]),
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Primary common mask\noutside original 1-line support")
    error_image = axes[1, 1].imshow(
        da3_error, cmap="magma", vmin=0, vmax=error_max_m
    )
    axes[1, 1].set_title("DA3 absolute error")
    axes[1, 2].imshow(a2f_error, cmap="magma", vmin=0, vmax=error_max_m)
    axes[1, 2].set_title("Any2Full absolute error")
    gain_image = axes[1, 3].imshow(
        gain, cmap="RdBu_r", vmin=-error_max_m, vmax=error_max_m
    )
    axes[1, 3].set_title("Error difference\nred = Any2Full better; blue = DA3 better")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.colorbar(
        depth_image,
        ax=axes[0, 1:].ravel().tolist(),
        shrink=0.72,
        label="Depth (m)",
    )
    figure.colorbar(
        error_image,
        ax=[axes[1, 1], axes[1, 2]],
        shrink=0.72,
        label="Absolute error (m)",
    )
    figure.colorbar(
        gain_image,
        ax=axes[1, 3],
        shrink=0.72,
        label="|DA3 error| - |Any2Full error| (m)",
    )
    relative = 100.0 * (
        float(da3_metric["rmse_m"]) - float(a2f_metric["rmse_m"])
    ) / float(da3_metric["rmse_m"])
    figure.suptitle(
        f"{scene}: exact same RGB and maximum-coverage four-line LiDAR\n"
        f"Primary-region Any2Full RMSE change versus DA3: {relative:+.2f}%"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def surface_panel(
    scene: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    da3: np.ndarray,
    any2full: np.ndarray,
    rows: list[dict[str, Any]],
    output: Path,
    depth_max_m: float,
) -> None:
    scene_rows = sorted(
        [row for row in rows if row["scene"] == scene],
        key=lambda row: int(row["top_px"]),
    )
    if len(scene_rows) != 3:
        raise RuntimeError(f"Expected three diagnostic surfaces for {scene}")
    figure, axes = plt.subplots(3, 4, figsize=(17.5, 12), constrained_layout=True)
    depth_image = None
    for row_index, row in enumerate(scene_rows):
        top = int(row["top_px"])
        left = int(row["left_px"])
        height = int(row["height_px"])
        width = int(row["width_px"])
        slices = np.s_[top : top + height, left : left + width]
        surface = str(row["surface"]).title()
        axes[row_index, 0].imshow(rgb[slices])
        axes[row_index, 0].set_title(
            f"{surface} GT-selected surface\n"
            f"GT plane residual {float(row['gt_plane_fit_rmse_m']):.3f} m"
        )
        depth_image = axes[row_index, 1].imshow(
            np.where(valid[slices], gt[slices], np.nan),
            cmap="turbo",
            vmin=0,
            vmax=depth_max_m,
        )
        axes[row_index, 1].set_title(f"GT\nmean {float(row['gt_mean_m']):.3f} m")
        axes[row_index, 2].imshow(
            np.where(valid[slices], da3[slices], np.nan),
            cmap="turbo",
            vmin=0,
            vmax=depth_max_m,
        )
        axes[row_index, 2].set_title(
            "DA3\n"
            f"mean {float(row['da3_mean_m']):.3f} m | "
            f"RMSE {float(row['da3_rmse_m']):.3f} m"
        )
        axes[row_index, 3].imshow(
            np.where(valid[slices], any2full[slices], np.nan),
            cmap="turbo",
            vmin=0,
            vmax=depth_max_m,
        )
        axes[row_index, 3].set_title(
            "Any2Full\n"
            f"mean {float(row['any2full_mean_m']):.3f} m | "
            f"RMSE {float(row['any2full_rmse_m']):.3f} m | "
            f"{float(row['any2full_rmse_improvement_pct']):+.1f}%"
        )
    for axis in axes.flat:
        axis.set_axis_off()
    if depth_image is not None:
        figure.colorbar(
            depth_image,
            ax=axes[:, 1:].ravel().tolist(),
            shrink=0.72,
            label="Depth (m)",
        )
    figure.suptitle(
        f"{scene}: same GT-selected surfaces, DA3 versus Any2Full\n"
        "Positive percentage means Any2Full has lower surface RMSE"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_report(
    path: Path,
    scenes: list[str],
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# iBims exact four-line DA3 versus Any2Full comparison",
        "",
        f"**Status:** PROVISIONAL COMPLETE-PIPELINE SMOKE TEST — {len(scenes)} scenes",
        "",
        "- Identical RGB images and byte-identical maximum-coverage sparse-depth NPY files are supplied for the paired comparison.",
        "- Four-line placement: 12.5%, 37.5%, 62.5%, and 87.5% of image height; same saved x-columns and noiseless metric returns.",
        "- DA3 condition: cached DA3-SMALL relative depth + global median metric alignment + validated existing Poisson.",
        "- Any2Full condition: Any2Full-vits native metric prediction; no extra median alignment, Poisson refinement, GT scaling, clipping, or repair.",
        "- Dense GT and evaluation masks are stored separately and are not passed to Any2Full.",
        "- Every comparison uses exactly the same evaluation pixels. Scene metrics are averaged with equal scene weight.",
        "- Physical anchors are counted before Any2Full's native internal resizing; resizing cannot add independent measurements.",
        "",
        "## Aggregate comparison",
        "",
        "A positive change means Any2Full reduced the error relative to DA3.",
        "",
        "| Common evaluation region | DA3 RMSE | Any2Full RMSE | Any2Full RMSE change | DA3 AbsRel | Any2Full AbsRel | Any2Full AbsRel change | Any2Full RMSE win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        da3 = summary_lookup(summary_rows, "da3_max_coverage", region)
        a2f = summary_lookup(summary_rows, "any2full_max_coverage", region)
        rmse = improvement_lookup(paired_rows, region, "rmse_m")
        absrel = improvement_lookup(paired_rows, region, "absrel_pct")
        lines.append(
            f"| {REGIONS[region]} | {float(da3['mean_rmse_m']):.4f} m | "
            f"{float(a2f['mean_rmse_m']):.4f} m | "
            f"{float(rmse['any2full_relative_improvement_pct']):+.2f}% | "
            f"{float(da3['mean_absrel_pct']):.3f}% | "
            f"{float(a2f['mean_absrel_pct']):.3f}% | "
            f"{float(absrel['any2full_relative_improvement_pct']):+.2f}% | "
            f"{float(rmse['any2full_win_rate_pct']):.1f}% |"
        )
    primary_rmse = improvement_lookup(paired_rows, "outside_original_line", "rmse_m")
    primary_absrel = improvement_lookup(
        paired_rows, "outside_original_line", "absrel_pct"
    )
    lines.extend(
        [
            "",
            "### Paired uncertainty on the primary region",
            "",
            f"- Any2Full RMSE improvement (DA3 - Any2Full): {float(primary_rmse['any2full_improvement_mean']):+.4f} m; 95% scene-bootstrap CI [{float(primary_rmse['improvement_bootstrap_ci95_low']):+.4f}, {float(primary_rmse['improvement_bootstrap_ci95_high']):+.4f}] m.",
            f"- Any2Full AbsRel improvement (DA3 - Any2Full): {float(primary_absrel['any2full_improvement_mean']):+.3f} percentage points; 95% scene-bootstrap CI [{float(primary_absrel['improvement_bootstrap_ci95_low']):+.3f}, {float(primary_absrel['improvement_bootstrap_ci95_high']):+.3f}] points.",
            "- With only three scenes, these intervals are descriptive and cannot establish general superiority.",
            "",
            "## Per-scene primary-region comparison",
            "",
            "| Scene | DA3 RMSE | Any2Full RMSE | Any2Full change | DA3 AbsRel | Any2Full AbsRel | Any2Full change | RMSE winner |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scene in scenes:
        da3 = metric_lookup(metric_rows, scene, "da3_max_coverage", "outside_original_line")
        a2f = metric_lookup(
            metric_rows, scene, "any2full_max_coverage", "outside_original_line"
        )
        da3_rmse = float(da3["rmse_m"])
        a2f_rmse = float(a2f["rmse_m"])
        da3_absrel = float(da3["absrel_pct"])
        a2f_absrel = float(a2f["absrel_pct"])
        lines.append(
            f"| {scene} | {da3_rmse:.4f} m | {a2f_rmse:.4f} m | "
            f"{100.0 * (da3_rmse - a2f_rmse) / da3_rmse:+.2f}% | "
            f"{da3_absrel:.3f}% | {a2f_absrel:.3f}% | "
            f"{100.0 * (da3_absrel - a2f_absrel) / da3_absrel:+.2f}% | "
            f"{'Any2Full' if a2f_rmse < da3_rmse else 'DA3'} |"
        )
    lines.extend(
        [
            "",
            "## Same GT-selected diagnostic surfaces",
            "",
            "These are the unchanged upper/middle/lower locally smooth patches selected by GT planarity in the placement study; neither method chose them.",
            "",
            "| Scene | Surface | GT mean | DA3 mean | Any2Full mean | DA3 RMSE | Any2Full RMSE | Any2Full change | Winner |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(surface_rows, key=lambda item: (str(item["scene"]), int(item["top_px"]))):
        lines.append(
            f"| {row['scene']} | {str(row['surface']).title()} | "
            f"{float(row['gt_mean_m']):.3f} m | {float(row['da3_mean_m']):.3f} m | "
            f"{float(row['any2full_mean_m']):.3f} m | "
            f"{float(row['da3_rmse_m']):.3f} m | "
            f"{float(row['any2full_rmse_m']):.3f} m | "
            f"{float(row['any2full_rmse_improvement_pct']):+.2f}% | "
            f"{row['rmse_winner']} |"
        )
    anchor_counts = [int(row["physical_anchor_count"]) for row in manifest_rows]
    lines.extend(
        [
            "",
            "## Input audit",
            "",
            f"- Physical four-line anchors per scene: mean {np.mean(anchor_counts):.1f}, median {np.median(anchor_counts):.0f}, range {min(anchor_counts)}-{max(anchor_counts)}.",
            "- Every exported sparse NPY passed the byte-level SHA-256 identity check against the DA3 placement experiment input.",
            "- Any2Full relative-disparity sidecar files are excluded from metric evaluation; only `<scene>.npy` is scored.",
            "",
            "## Interpretation",
            "",
            "This answers which complete pipeline is better on these exact three images under the same simulated four-line evidence. It does not establish an indoor four-line SOTA or an optimal physical sensor geometry. A broader, predeclared development/test split is required before making either claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    placement_output = ensure_directory(args.placement_output, "placement output")
    prepared_root = ensure_directory(args.prepared_data_root, "prepared data root")
    any2full_dir = ensure_directory(args.any2full_dir, "Any2Full prediction directory")
    output_dir = args.output_dir.expanduser().resolve()
    if args.plot_max_depth_m <= 0 or args.plot_error_max_m <= 0:
        raise ValueError("plot limits must be positive")
    placement, reference, scenes = placement_context(
        placement_output, args.expected_scenes
    )
    prepared_protocol = read_json(prepared_root / "protocol.json")
    if prepared_protocol.get("source_placement_protocol_sha256") != sha256_file(
        placement_output / "protocol.json"
    ):
        raise RuntimeError(
            "Prepared data was not created from this placement output. Rerun prepare "
            "or point evaluate at the matching directories."
        )
    manifest_rows = load_manifest(prepared_root, args.expected_scenes)
    manifest_scenes = [row["scene"] for row in manifest_rows]
    if manifest_scenes != scenes:
        raise RuntimeError(
            f"Prepared scene order differs from placement protocol: {manifest_scenes} vs {scenes}"
        )
    expected_predictions = {f"{scene}.npy" for scene in scenes}
    metric_predictions = {
        path.name
        for path in any2full_dir.glob("*.npy")
        if path.is_file() and not path.stem.endswith("_rel")
    }
    missing = sorted(expected_predictions - metric_predictions)
    extras = sorted(metric_predictions - expected_predictions)
    if missing or extras:
        raise RuntimeError(
            "Any2Full metric prediction set must contain exactly the three prepared scenes. "
            f"missing={missing}, extras={extras}"
        )
    failed_path = any2full_dir / "failed_pairs.txt"
    if failed_path.is_file() and failed_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Any2Full reported failed inputs: {failed_path}")

    paired = import_paired_module(Path(__file__).resolve().parent)
    margin = int(reference.get("outside_margin_px", 10))
    surface_source = read_csv(placement_output / "surface_measurements.csv")
    surface_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in surface_source:
        if row.get("scene") in scenes:
            surface_by_scene[str(row["scene"])].append(row)
    for scene in scenes:
        if {row.get("surface") for row in surface_by_scene[scene]} != {
            "upper",
            "middle",
            "lower",
        }:
            raise RuntimeError(f"Placement output lacks three diagnostic surfaces for {scene}")

    metric_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    arrays_for_panels: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    audit_rows: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifest_rows, start=1):
        scene = manifest["scene"]
        sparse_path = prepared_root / manifest["sparse_input_m"]
        if sha256_file(sparse_path) != manifest["source_sparse_sha256"]:
            raise RuntimeError(f"{scene}: prepared sparse input hash no longer matches DA3 input")
        gt = load_npy_2d(prepared_root / manifest["gt_m_evaluation_only"]).astype(
            np.float32
        )
        valid = load_npy_2d(
            prepared_root / manifest["valid_mask_evaluation_only"], gt.shape
        ).astype(bool)
        one_anchors = load_npy_2d(
            prepared_root / manifest["one_line_mask_evaluation_only"], gt.shape
        ).astype(bool)
        four_anchors = load_npy_2d(
            prepared_root / manifest["four_line_mask_evaluation_only"], gt.shape
        ).astype(bool)
        sparse = load_npy_2d(sparse_path, gt.shape).astype(np.float32)
        if not np.array_equal(sparse > 0, four_anchors):
            raise RuntimeError(f"{scene}: prepared four-line mask differs from sparse input")
        physical_count = int(np.count_nonzero(four_anchors))
        if physical_count != int(manifest["physical_anchor_count"]):
            raise RuntimeError(f"{scene}: physical anchor count changed after preparation")
        rgb_path = prepared_root / manifest["rgb"]
        if sha256_file(rgb_path) != manifest["rgb_sha256"]:
            raise RuntimeError(f"{scene}: RGB changed after preparation")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != gt.shape:
            raise ValueError(f"{scene}: RGB/GT shape mismatch")
        da3_path = Path(manifest["source_da3_prediction"]).expanduser().resolve()
        expected_da3_path = (
            placement_output / "predictions_m" / "max_coverage_4line" / f"{scene}.npy"
        ).resolve()
        if da3_path != expected_da3_path:
            raise RuntimeError(f"{scene}: manifest points at a different DA3 prediction")
        da3 = load_prediction(da3_path, gt.shape, valid)
        any2full = load_prediction(any2full_dir / f"{scene}.npy", gt.shape, valid)
        masks = comparison_masks(valid, one_anchors, four_anchors, margin)
        predictions = {
            "da3_max_coverage": da3,
            "any2full_max_coverage": any2full,
        }
        for method, prediction in predictions.items():
            for region, mask in masks.items():
                metric_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "method_label": METHODS[method],
                        "region": region,
                        "region_label": REGIONS[region],
                        "physical_anchor_count": physical_count,
                        **paired.metrics(prediction, gt, mask),
                    }
                )
        for roi in surface_by_scene[scene]:
            surface_rows.append(
                surface_measurement(scene, roi, gt, valid, da3, any2full)
            )
        audit_rows.append(
            {
                "scene": scene,
                "physical_anchor_count": physical_count,
                "rgb_sha256": sha256_file(rgb_path),
                "sparse_sha256": sha256_file(sparse_path),
                "sparse_matches_da3_source": True,
                "da3_prediction": str(da3_path),
                "any2full_prediction": str((any2full_dir / f"{scene}.npy").resolve()),
                "prediction_shape": f"{gt.shape[0]}x{gt.shape[1]}",
            }
        )
        arrays_for_panels[scene] = (
            rgb,
            gt,
            valid,
            one_anchors,
            four_anchors,
            da3,
            any2full,
        )
        da3_primary = metric_lookup(
            metric_rows, scene, "da3_max_coverage", "outside_original_line"
        )
        a2f_primary = metric_lookup(
            metric_rows, scene, "any2full_max_coverage", "outside_original_line"
        )
        gain = 100.0 * (
            float(da3_primary["rmse_m"]) - float(a2f_primary["rmse_m"])
        ) / float(da3_primary["rmse_m"])
        print(
            f"[{index}/{len(scenes)}] {scene}: DA3->Any2Full primary RMSE "
            f"{float(da3_primary['rmse_m']):.3f}->{float(a2f_primary['rmse_m']):.3f} m "
            f"({gain:+.2f}%)",
            flush=True,
        )

    summary_rows = aggregate_metrics(metric_rows)
    paired_rows = paired_improvements(
        metric_rows, args.bootstrap_samples, args.seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_scene_metrics.csv", metric_rows)
    write_csv(output_dir / "summary_equal_scene_weight.csv", summary_rows)
    write_csv(output_dir / "paired_improvement.csv", paired_rows)
    write_csv(output_dir / "surface_measurements.csv", surface_rows)
    write_csv(output_dir / "input_audit.csv", audit_rows)
    evaluation_protocol = {
        "benchmark": "iBims exact maximum-coverage four-line DA3 versus Any2Full",
        "version": VERSION,
        "status": "provisional_three_scene_smoke_test",
        "source_placement_output": str(placement_output),
        "prepared_data_root": str(prepared_root),
        "any2full_prediction_dir": str(any2full_dir),
        "scenes": scenes,
        "row_fractions": list(MAX_COVERAGE_FRACS),
        "outside_margin_px": margin,
        "primary_region": "outside_original_line",
        "equal_scene_weight": True,
        "prediction_repairs": "none",
        "prediction_gt_alignment": "none",
        "common_mask_rule": "both methods are evaluated on identical pixels",
    }
    atomic_json(output_dir / "protocol.json", evaluation_protocol)

    if not args.skip_panels:
        visual_root = output_dir / "visual_comparisons"
        for scene in scenes:
            rgb, gt, valid, one_anchors, four_anchors, da3, any2full = arrays_for_panels[
                scene
            ]
            masks = comparison_masks(valid, one_anchors, four_anchors, margin)
            depth_panel(
                scene,
                rgb,
                gt,
                valid,
                four_anchors,
                masks["outside_original_line"],
                da3,
                any2full,
                metric_rows,
                visual_root / f"{scene}__depth_and_error.png",
                args.plot_max_depth_m,
                args.plot_error_max_m,
            )
            surface_panel(
                scene,
                rgb,
                gt,
                valid,
                da3,
                any2full,
                surface_rows,
                visual_root / f"{scene}__surface_comparison.png",
                args.plot_max_depth_m,
            )

    report_path = output_dir / "comparison_report.md"
    write_report(
        report_path,
        scenes,
        metric_rows,
        summary_rows,
        paired_rows,
        surface_rows,
        manifest_rows,
    )
    print("\n===== EXACT FOUR-LINE DA3 VS ANY2FULL =====\n")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Full report: {report_path}")
    if not args.skip_panels:
        print(f"Depth and surface panels: {output_dir / 'visual_comparisons'}")
    print(f"Input audit: {output_dir / 'input_audit.csv'}")


def self_test() -> None:
    gt = np.full((20, 24), 2.0, dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    one = np.zeros_like(valid)
    four = np.zeros_like(valid)
    one[10, 2:22:3] = True
    for row in (2, 7, 12, 17):
        four[row, 2:22:3] = True
    masks = comparison_masks(valid, one, four, 1)
    if not all(np.any(mask) for mask in masks.values()):
        raise AssertionError("mask test failed")
    da3 = np.full_like(gt, 2.2)
    any2full = np.full_like(gt, 2.1)
    fake_rows: list[dict[str, Any]] = []
    for method, prediction in (
        ("da3_max_coverage", da3),
        ("any2full_max_coverage", any2full),
    ):
        for region, mask in masks.items():
            error = prediction[mask].astype(np.float64) - gt[mask].astype(np.float64)
            fake_rows.append(
                {
                    "scene": "test",
                    "method": method,
                    "region": region,
                    "rmse_m": float(np.sqrt(np.mean(error * error))),
                    "absrel_pct": float(100 * np.mean(np.abs(error) / gt[mask])),
                    "mae_m": float(np.mean(np.abs(error))),
                    "delta1_pct": 100.0,
                }
            )
    paired = paired_improvements(fake_rows, 100, 7)
    rmse = improvement_lookup(paired, "all_valid", "rmse_m")
    if not math.isclose(float(rmse["any2full_improvement_mean"]), 0.1, abs_tol=1e-5):
        raise AssertionError(rmse)
    print("SELF-TEST PASSED")


def main() -> None:
    args = arguments()
    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "self-test":
        self_test()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
