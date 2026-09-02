#!/usr/bin/env python3
"""Compare current and maximum-coverage four-line layouts on the same iBims scenes.

This companion evaluator reads the exact scene list and protocol from a prior
``ibims_1line_vs_4line_da3_comparison.py`` smoke test.  It then recomputes three
conditions with the same cached DA3 relative depth and the same validated
median-plus-Poisson pipeline:

* the established one-line input;
* the current four-line layout at 20%, 40%, 60%, and 80% image height;
* the maximum-coverage layout at 12.5%, 37.5%, 62.5%, and 87.5% image height.

All layout comparisons use common evaluation pixels.  Dense ground truth is
used only to simulate hypothetical four-line returns, evaluate predictions,
and choose deterministic locally smooth diagnostic patches.  It never changes
the dense prediction outside the selected sparse anchors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import distance_transform_edt


VERSION = "1.0"
CURRENT_FRACS = (0.20, 0.40, 0.60, 0.80)
MAX_COVERAGE_FRACS = (0.125, 0.375, 0.625, 0.875)
METHODS = {
    "one_line": "1 line",
    "current_4line": "Current 4 lines (20/40/60/80%)",
    "max_coverage_4line": "Maximum coverage (12.5/37.5/62.5/87.5%)",
}
REGIONS = {
    "all_valid": "All valid GT pixels",
    "outside_original_line_common": "Outside original 1-line support",
    "outside_all_input_patterns_common": "Outside all three input patterns",
}
SURFACE_BANDS = (
    ("upper", 0.00, 1.0 / 3.0),
    ("middle", 1.0 / 3.0, 2.0 / 3.0),
    ("lower", 2.0 / 3.0, 1.00),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-root", type=Path, required=True)
    parser.add_argument("--a2f-root", type=Path, required=True)
    parser.add_argument(
        "--reference-output",
        type=Path,
        required=True,
        help="Output directory from the completed 3-scene current-layout smoke test.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-max-depth-m", type=float, default=10.0)
    parser.add_argument("--plot-error-max-m", type=float, default=1.0)
    parser.add_argument("--surface-height-frac", type=float, default=0.14)
    parser.add_argument("--surface-width-frac", type=float, default=0.16)
    return parser.parse_args()


def import_paired_module(script_dir: Path) -> Any:
    path = script_dir / "ibims_1line_vs_4line_da3_comparison.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required sibling evaluator is missing: {path}\n"
            "Place both scripts in the same ibims directory."
        )
    spec = importlib.util.spec_from_file_location("ibims_paired_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "load_ibims",
        "load_npy",
        "npy_path",
        "sanitize_one_line",
        "simulate_four_lines",
        "median_align",
        "load_poisson",
        "call_poisson",
        "metrics",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"Sibling evaluator is missing helpers: {missing}")
    return module


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def protocol_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_protocol(path: Path, payload: dict[str, Any], resume: bool) -> None:
    payload = dict(payload)
    payload["configuration_sha256"] = protocol_hash(payload)
    if path.is_file():
        old = read_json(path)
        if old.get("configuration_sha256") != payload["configuration_sha256"]:
            raise RuntimeError(
                f"{path.parent} contains a different experiment. Use a new output directory."
            )
        if not resume:
            raise FileExistsError(
                f"{path.parent} already exists; pass --resume or use a new output directory."
            )
        return
    atomic_json(path, payload)


def resolve_protocol_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise KeyError(f"Reference protocol does not contain {label}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference {label} no longer exists: {path}")
    return path


def reference_scenes(reference_output: Path, expected: int) -> list[str]:
    rows = read_csv(reference_output / "per_scene_paired_metrics.csv")
    scenes = sorted({str(row.get("scene", "")) for row in rows if row.get("scene")})
    if len(scenes) != expected:
        raise RuntimeError(
            f"Reference output contains {len(scenes)} scenes, expected {expected}: {scenes}"
        )
    return scenes


def common_masks(
    valid: np.ndarray,
    one_anchors: np.ndarray,
    current_anchors: np.ndarray,
    maximum_anchors: np.ndarray,
    margin_px: int,
) -> dict[str, np.ndarray]:
    distances = [
        distance_transform_edt(~anchors)
        for anchors in (one_anchors, current_anchors, maximum_anchors)
    ]
    return {
        "all_valid": valid.copy(),
        "outside_original_line_common": valid & (distances[0] > margin_px),
        "outside_all_input_patterns_common": (
            valid
            & (distances[0] > margin_px)
            & (distances[1] > margin_px)
            & (distances[2] > margin_px)
        ),
    }


def fit_plane_score(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    yy, xx = np.where(valid)
    values = depth[valid].astype(np.float64)
    coverage = float(values.size / depth.size)
    if values.size < 20:
        return math.inf, coverage, math.inf
    x_scale = max(float(depth.shape[1] - 1), 1.0)
    y_scale = max(float(depth.shape[0] - 1), 1.0)
    design = np.column_stack(
        (
            2.0 * xx.astype(np.float64) / x_scale - 1.0,
            2.0 * yy.astype(np.float64) / y_scale - 1.0,
            np.ones(values.size, dtype=np.float64),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    plane_rmse = float(np.sqrt(np.mean(residual * residual)))
    median_depth = float(np.median(values))
    relative_plane_rmse = plane_rmse / max(median_depth, 1e-6)
    score = relative_plane_rmse + 0.20 * (1.0 - coverage)
    return score, coverage, plane_rmse


def select_surface_rois(
    gt: np.ndarray,
    valid: np.ndarray,
    height_fraction: float,
    width_fraction: float,
) -> list[dict[str, Any]]:
    """Choose one deterministic, locally planar GT patch in each vertical third."""
    height, width = gt.shape
    box_h = min(height, max(16, int(round(height * height_fraction))))
    box_w = min(width, max(16, int(round(width * width_fraction))))
    stride_y = max(4, box_h // 4)
    stride_x = max(4, box_w // 4)
    selected: list[dict[str, Any]] = []
    for label, low_fraction, high_fraction in SURFACE_BANDS:
        band_top = int(math.floor(low_fraction * height))
        band_bottom = int(math.ceil(high_fraction * height))
        candidates: list[tuple[float, int, int, float, float]] = []
        minimum_top = max(0, band_top)
        maximum_top = min(height - box_h, max(minimum_top, band_bottom - box_h))
        y_values = list(range(minimum_top, maximum_top + 1, stride_y))
        if not y_values or y_values[-1] != maximum_top:
            y_values.append(maximum_top)
        x_values = list(range(0, max(1, width - box_w + 1), stride_x))
        if not x_values or x_values[-1] != width - box_w:
            x_values.append(width - box_w)
        for top in sorted(set(y_values)):
            for left in sorted(set(x_values)):
                patch_valid = valid[top : top + box_h, left : left + box_w]
                score, coverage, plane_rmse = fit_plane_score(
                    gt[top : top + box_h, left : left + box_w],
                    patch_valid,
                )
                if coverage >= 0.80 and math.isfinite(score):
                    candidates.append((score, top, left, coverage, plane_rmse))
        if not candidates:
            raise RuntimeError(f"Could not find a valid {label} diagnostic surface patch")
        score, top, left, coverage, plane_rmse = min(candidates)
        selected.append(
            {
                "surface": label,
                "top": top,
                "left": left,
                "height": box_h,
                "width": box_w,
                "valid_fraction": coverage,
                "gt_plane_fit_rmse_m": plane_rmse,
                "selection_score": score,
            }
        )
    return selected


def surface_measurement(
    scene: str,
    roi: dict[str, Any],
    gt: np.ndarray,
    valid: np.ndarray,
    current: np.ndarray,
    maximum: np.ndarray,
    current_rows: list[int],
    maximum_rows: list[int],
) -> dict[str, Any]:
    top = int(roi["top"])
    left = int(roi["left"])
    height = int(roi["height"])
    width = int(roi["width"])
    region = np.zeros_like(valid, dtype=bool)
    region[top : top + height, left : left + width] = True
    mask = region & valid
    target = gt[mask].astype(np.float64)
    current_values = current[mask].astype(np.float64)
    maximum_values = maximum[mask].astype(np.float64)
    current_error = current_values - target
    maximum_error = maximum_values - target
    center_y = top + (height - 1) / 2.0
    current_distance = min(abs(center_y - row) for row in current_rows)
    maximum_distance = min(abs(center_y - row) for row in maximum_rows)
    current_rmse = float(np.sqrt(np.mean(current_error * current_error)))
    maximum_rmse = float(np.sqrt(np.mean(maximum_error * maximum_error)))
    current_absrel = float(100.0 * np.mean(np.abs(current_error) / target))
    maximum_absrel = float(100.0 * np.mean(np.abs(maximum_error) / target))
    return {
        "scene": scene,
        "surface": roi["surface"],
        "top_px": top,
        "left_px": left,
        "height_px": height,
        "width_px": width,
        "valid_pixel_count": int(target.size),
        "valid_fraction": float(roi["valid_fraction"]),
        "gt_plane_fit_rmse_m": float(roi["gt_plane_fit_rmse_m"]),
        "center_distance_to_current_line_px": float(current_distance),
        "center_distance_to_max_coverage_line_px": float(maximum_distance),
        "gt_mean_m": float(np.mean(target)),
        "gt_median_m": float(np.median(target)),
        "current_mean_m": float(np.mean(current_values)),
        "current_median_m": float(np.median(current_values)),
        "current_bias_m": float(np.mean(current_error)),
        "current_mae_m": float(np.mean(np.abs(current_error))),
        "current_rmse_m": current_rmse,
        "current_absrel_pct": current_absrel,
        "max_coverage_mean_m": float(np.mean(maximum_values)),
        "max_coverage_median_m": float(np.median(maximum_values)),
        "max_coverage_bias_m": float(np.mean(maximum_error)),
        "max_coverage_mae_m": float(np.mean(np.abs(maximum_error))),
        "max_coverage_rmse_m": maximum_rmse,
        "max_coverage_absrel_pct": maximum_absrel,
        "max_vs_current_rmse_improvement_m": current_rmse - maximum_rmse,
        "max_vs_current_rmse_improvement_pct": (
            100.0 * (current_rmse - maximum_rmse) / current_rmse
            if current_rmse > 0
            else math.nan
        ),
        "max_vs_current_absrel_improvement_points": current_absrel - maximum_absrel,
    }


def metric_lookup(
    rows: list[dict[str, Any]], scene: str, method: str, region: str
) -> dict[str, Any]:
    for row in rows:
        if row["scene"] == scene and row["method"] == method and row["region"] == region:
            return row
    raise KeyError((scene, method, region))


def draw_lines_and_rois(
    axis: Any,
    rgb: np.ndarray,
    line_rows: list[int],
    rois: list[dict[str, Any]],
    title: str,
    line_color: str,
) -> None:
    axis.imshow(rgb)
    for row in line_rows:
        axis.axhline(row, color=line_color, linewidth=1.2, alpha=0.95)
    colors = {"upper": "white", "middle": "yellow", "lower": "magenta"}
    for roi in rois:
        color = colors[str(roi["surface"])]
        rectangle = Rectangle(
            (int(roi["left"]), int(roi["top"])),
            int(roi["width"]),
            int(roi["height"]),
            fill=False,
            edgecolor=color,
            linewidth=2.0,
        )
        axis.add_patch(rectangle)
        axis.text(
            int(roi["left"]) + 3,
            int(roi["top"]) + 13,
            str(roi["surface"])[0].upper(),
            color=color,
            weight="bold",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )
    axis.set_title(title)
    axis.set_axis_off()


def depth_panel(
    scene: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    predictions: dict[str, np.ndarray],
    current_rows: list[int],
    maximum_rows: list[int],
    rois: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    output: Path,
    depth_max_m: float,
    error_max_m: float,
) -> None:
    region = "outside_original_line_common"
    current_metric = metric_lookup(metric_rows, scene, "current_4line", region)
    maximum_metric = metric_lookup(metric_rows, scene, "max_coverage_4line", region)
    gt_show = np.where(valid, gt, np.nan)
    current_error = np.where(valid, np.abs(predictions["current_4line"] - gt), np.nan)
    maximum_error = np.where(valid, np.abs(predictions["max_coverage_4line"] - gt), np.nan)
    gain = current_error - maximum_error

    figure, axes = plt.subplots(2, 4, figsize=(21, 10.5), constrained_layout=True)
    draw_lines_and_rois(
        axes[0, 0], rgb, current_rows, rois,
        "Current placement: 20/40/60/80%", "cyan"
    )
    draw_lines_and_rois(
        axes[0, 1], rgb, maximum_rows, rois,
        "Maximum coverage: 12.5/37.5/62.5/87.5%", "lime"
    )
    depth_image = axes[0, 2].imshow(gt_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[0, 2].set_title("iBims metric GT")
    gain_image = axes[0, 3].imshow(
        gain, cmap="RdBu_r", vmin=-error_max_m, vmax=error_max_m
    )
    axes[0, 3].set_title("Absolute-error reduction\nred = maximum coverage better")

    axes[1, 0].imshow(
        np.where(valid, predictions["one_line"], np.nan),
        cmap="turbo", vmin=0, vmax=depth_max_m,
    )
    axes[1, 0].set_title("Original 1-line result")
    axes[1, 1].imshow(
        np.where(valid, predictions["current_4line"], np.nan),
        cmap="turbo", vmin=0, vmax=depth_max_m,
    )
    axes[1, 1].set_title(
        "Current four-line result\n"
        f"RMSE {float(current_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(current_metric['absrel_pct']):.2f}%"
    )
    axes[1, 2].imshow(
        np.where(valid, predictions["max_coverage_4line"], np.nan),
        cmap="turbo", vmin=0, vmax=depth_max_m,
    )
    axes[1, 2].set_title(
        "Maximum-coverage result\n"
        f"RMSE {float(maximum_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(maximum_metric['absrel_pct']):.2f}%"
    )
    axes[1, 3].imshow(maximum_error, cmap="magma", vmin=0, vmax=error_max_m)
    axes[1, 3].set_title("Maximum-coverage absolute error")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.colorbar(
        depth_image,
        ax=[axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]],
        shrink=0.78,
        label="Depth (m)",
    )
    figure.colorbar(
        gain_image,
        ax=axes[0, 3],
        shrink=0.78,
        label="|current error| - |maximum-coverage error| (m)",
    )
    rmse_gain = 100.0 * (
        float(current_metric["rmse_m"]) - float(maximum_metric["rmse_m"])
    ) / float(current_metric["rmse_m"])
    figure.suptitle(
        f"{scene}: current versus maximum-coverage placement\n"
        f"Outside-original-line RMSE change: {rmse_gain:+.2f}%"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def surface_panel(
    scene: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    predictions: dict[str, np.ndarray],
    rois: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    output: Path,
    depth_max_m: float,
) -> None:
    lookup = {(row["scene"], row["surface"]): row for row in measurements}
    figure, axes = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
    depth_image = None
    for row_index, roi in enumerate(rois):
        top, left = int(roi["top"]), int(roi["left"])
        height, width = int(roi["height"]), int(roi["width"])
        slices = np.s_[top : top + height, left : left + width]
        surface = str(roi["surface"])
        measurement = lookup[(scene, surface)]
        axes[row_index, 0].imshow(rgb[slices])
        axes[row_index, 0].set_title(
            f"{surface.title()} locally smooth region\n"
            f"GT plane residual {float(measurement['gt_plane_fit_rmse_m']):.3f} m"
        )
        depth_image = axes[row_index, 1].imshow(
            np.where(valid[slices], gt[slices], np.nan),
            cmap="turbo", vmin=0, vmax=depth_max_m,
        )
        axes[row_index, 1].set_title(
            f"GT\nmean {float(measurement['gt_mean_m']):.3f} m"
        )
        axes[row_index, 2].imshow(
            np.where(valid[slices], predictions["current_4line"][slices], np.nan),
            cmap="turbo", vmin=0, vmax=depth_max_m,
        )
        axes[row_index, 2].set_title(
            "Current placement\n"
            f"mean {float(measurement['current_mean_m']):.3f} m | "
            f"RMSE {float(measurement['current_rmse_m']):.3f} m"
        )
        axes[row_index, 3].imshow(
            np.where(valid[slices], predictions["max_coverage_4line"][slices], np.nan),
            cmap="turbo", vmin=0, vmax=depth_max_m,
        )
        axes[row_index, 3].set_title(
            "Maximum coverage\n"
            f"mean {float(measurement['max_coverage_mean_m']):.3f} m | "
            f"RMSE {float(measurement['max_coverage_rmse_m']):.3f} m | "
            f"gain {float(measurement['max_vs_current_rmse_improvement_pct']):+.1f}%"
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
        f"{scene}: GT surface measurements versus both four-line placements\n"
        "Regions are selected from GT local planarity only; predictions do not choose them"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


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
            }
        )
    return result


def summary_lookup(
    rows: list[dict[str, Any]], method: str, region: str
) -> dict[str, Any]:
    for row in rows:
        if row["method"] == method and row["region"] == region:
            return row
    raise KeyError((method, region))


def write_report(
    path: Path,
    scenes: list[str],
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# iBims current versus maximum-coverage four-line placement",
        "",
        f"**Status:** PROVISIONAL PLACEMENT SMOKE TEST — {len(scenes)} scenes",
        "",
        "- Same scenes, cached DA3-SMALL maps, global median alignment, and validated Poisson parameters in every condition.",
        "- Current placement: 20%, 40%, 60%, and 80% of image height.",
        "- Maximum-coverage placement: 12.5%, 37.5%, 62.5%, and 87.5% of image height.",
        "- Both four-line layouts use the same x-columns and one noiseless GT-simulated return per valid ray.",
        "- Dense GT is used only for simulated returns, evaluation, and diagnostic surface selection.",
        "- This three-scene result is exploratory and cannot establish the optimal physical beam angles.",
        "",
        "## Complete-system comparison: DA3 + median + Poisson",
        "",
        "| Common evaluation region | 1-line RMSE | Current 4-line RMSE | Max-coverage RMSE | Max vs current | 1-line AbsRel | Current AbsRel | Max-coverage AbsRel | Max vs current |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        one = summary_lookup(summary_rows, "one_line", region)
        current = summary_lookup(summary_rows, "current_4line", region)
        maximum = summary_lookup(summary_rows, "max_coverage_4line", region)
        current_rmse = float(current["mean_rmse_m"])
        maximum_rmse = float(maximum["mean_rmse_m"])
        current_absrel = float(current["mean_absrel_pct"])
        maximum_absrel = float(maximum["mean_absrel_pct"])
        rmse_gain = 100.0 * (current_rmse - maximum_rmse) / current_rmse
        absrel_gain = 100.0 * (current_absrel - maximum_absrel) / current_absrel
        lines.append(
            f"| {REGIONS[region]} | {float(one['mean_rmse_m']):.4f} m | "
            f"{current_rmse:.4f} m | {maximum_rmse:.4f} m | {rmse_gain:+.2f}% | "
            f"{float(one['mean_absrel_pct']):.3f}% | {current_absrel:.3f}% | "
            f"{maximum_absrel:.3f}% | {absrel_gain:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Per-scene primary-region comparison",
            "",
            "The primary region is outside the original one-line support and is identical for every method.",
            "",
            "| Scene | Current RMSE | Max-coverage RMSE | RMSE change | Current AbsRel | Max-coverage AbsRel | AbsRel change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene in scenes:
        current = metric_lookup(
            metric_rows, scene, "current_4line", "outside_original_line_common"
        )
        maximum = metric_lookup(
            metric_rows, scene, "max_coverage_4line", "outside_original_line_common"
        )
        current_rmse = float(current["rmse_m"])
        maximum_rmse = float(maximum["rmse_m"])
        current_absrel = float(current["absrel_pct"])
        maximum_absrel = float(maximum["absrel_pct"])
        lines.append(
            f"| {scene} | {current_rmse:.4f} m | {maximum_rmse:.4f} m | "
            f"{100.0 * (current_rmse - maximum_rmse) / current_rmse:+.2f}% | "
            f"{current_absrel:.3f}% | {maximum_absrel:.3f}% | "
            f"{100.0 * (current_absrel - maximum_absrel) / current_absrel:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic locally smooth surface measurements",
            "",
            "One upper, middle, and lower patch is selected per scene using only valid-GT coverage and local plane-fit residual. This is a diagnostic, not a tuned evaluation mask.",
            "",
            "| Scene | Surface | GT mean | Current mean | Max-coverage mean | Current RMSE | Max-coverage RMSE | RMSE change |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(surface_rows, key=lambda item: (item["scene"], item["top_px"])):
        lines.append(
            f"| {row['scene']} | {str(row['surface']).title()} | "
            f"{float(row['gt_mean_m']):.3f} m | {float(row['current_mean_m']):.3f} m | "
            f"{float(row['max_coverage_mean_m']):.3f} m | "
            f"{float(row['current_rmse_m']):.3f} m | "
            f"{float(row['max_coverage_rmse_m']):.3f} m | "
            f"{float(row['max_vs_current_rmse_improvement_pct']):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive change means maximum coverage reduced error relative to the current placement. The decisive placement study must use a separate development subset and then a held-out test; these same three smoke-test scenes are only for directional evidence and visual inspection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    args.da3_root = args.da3_root.expanduser().resolve()
    args.a2f_root = args.a2f_root.expanduser().resolve()
    args.reference_output = args.reference_output.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.expected_scenes <= 0:
        raise ValueError("--expected-scenes must be positive")
    if not 0.04 <= args.surface_height_frac <= 0.30:
        raise ValueError("--surface-height-frac must be between 0.04 and 0.30")
    if not 0.04 <= args.surface_width_frac <= 0.30:
        raise ValueError("--surface-width-frac must be between 0.04 and 0.30")
    if args.plot_max_depth_m <= 0 or args.plot_error_max_m <= 0:
        raise ValueError("Plot limits must be positive")

    paired = import_paired_module(Path(__file__).resolve().parent)
    reference_protocol = read_json(args.reference_output / "protocol.json")
    reference_fracs = tuple(float(value) for value in reference_protocol.get("row_fracs", []))
    if len(reference_fracs) != 4 or not np.allclose(reference_fracs, CURRENT_FRACS):
        raise RuntimeError(
            "The reference output is not the 20/40/60/80% current-layout experiment: "
            f"{reference_fracs}"
        )
    scenes = reference_scenes(args.reference_output, args.expected_scenes)
    gt_dir = resolve_protocol_path(reference_protocol.get("gt_dir"), "gt_dir")
    sparse_dir = resolve_protocol_path(
        reference_protocol.get("one_line_dir"), "one_line_dir"
    )
    da3_dir = resolve_protocol_path(
        reference_protocol.get("cached_da3_dir"), "cached_da3_dir"
    )
    sensor_min = float(reference_protocol.get("sensor_min_depth_m", 0.10))
    sensor_max = float(reference_protocol.get("sensor_max_depth_m", 32.0))
    eval_max = float(reference_protocol.get("eval_max_depth_m", 0.0))
    margin = int(reference_protocol.get("outside_margin_px", 10))
    rtol = float(reference_protocol.get("poisson_rtol", 1e-6))
    maxiter = int(reference_protocol.get("poisson_maxiter", 5000))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "benchmark": "iBims current versus maximum-coverage four-line placement",
        "version": VERSION,
        "reference_output": str(args.reference_output),
        "reference_configuration_sha256": reference_protocol.get("configuration_sha256"),
        "scenes": scenes,
        "current_row_fracs": list(CURRENT_FRACS),
        "max_coverage_row_fracs": list(MAX_COVERAGE_FRACS),
        "sensor_min_depth_m": sensor_min,
        "sensor_max_depth_m": sensor_max,
        "eval_max_depth_m": eval_max,
        "outside_margin_px": margin,
        "poisson_rtol": rtol,
        "poisson_maxiter": maxiter,
        "surface_selection": "one locally planar GT-only patch in each vertical third",
        "gt_use": "simulate sparse four-line returns, evaluate, and select diagnostic patches only",
        "common_mask_rule": "layout comparisons use identical pixels",
    }
    validate_protocol(args.output_dir / "protocol.json", protocol, args.resume)
    poisson = paired.load_poisson(args.da3_root)

    metrics_path = args.output_dir / "per_scene_placement_metrics.csv"
    surfaces_path = args.output_dir / "surface_measurements.csv"
    metric_rows: list[dict[str, Any]] = list(read_csv(metrics_path)) if args.resume else []
    surface_rows: list[dict[str, Any]] = list(read_csv(surfaces_path)) if args.resume else []
    prediction_root = args.output_dir / "predictions_m"
    sparse_root = args.output_dir / "sparse_inputs_m"
    for method in METHODS:
        (prediction_root / method).mkdir(parents=True, exist_ok=True)
        (sparse_root / method).mkdir(parents=True, exist_ok=True)

    for index, scene in enumerate(scenes, start=1):
        complete_metric_keys = {
            (str(row.get("method")), str(row.get("region")))
            for row in metric_rows
            if row.get("scene") == scene
        }
        complete_surfaces = {
            str(row.get("surface"))
            for row in surface_rows
            if row.get("scene") == scene
        }
        prediction_files_exist = all(
            (prediction_root / method / f"{scene}.npy").is_file()
            for method in METHODS
        )
        expected_metric_keys = {
            (method, region) for method in METHODS for region in REGIONS
        }
        if (
            args.resume
            and expected_metric_keys <= complete_metric_keys
            and {"upper", "middle", "lower"} <= complete_surfaces
            and prediction_files_exist
        ):
            print(f"[{index}/{len(scenes)}] {scene} resume-skip", flush=True)
            continue

        gt, valid, rgb = paired.load_ibims(gt_dir / f"{scene}.mat")
        if eval_max > 0:
            valid &= gt <= eval_max
        raw_sparse = paired.load_npy(paired.npy_path(sparse_dir, scene), gt.shape)
        relative = paired.load_npy(
            paired.npy_path(da3_dir, scene, da3=True), gt.shape
        )
        one_sparse, one_anchors = paired.sanitize_one_line(
            raw_sparse, valid, sensor_min, sensor_max
        )
        current_sparse, current_anchors, current_rows, source_columns = (
            paired.simulate_four_lines(
                gt, valid, one_sparse, CURRENT_FRACS, sensor_min, sensor_max
            )
        )
        maximum_sparse, maximum_anchors, maximum_rows, maximum_columns = (
            paired.simulate_four_lines(
                gt, valid, one_sparse, MAX_COVERAGE_FRACS, sensor_min, sensor_max
            )
        )
        if source_columns != maximum_columns:
            raise RuntimeError("Four-line layouts used different source x-columns")

        sparse_inputs = {
            "one_line": (one_sparse, one_anchors),
            "current_4line": (current_sparse, current_anchors),
            "max_coverage_4line": (maximum_sparse, maximum_anchors),
        }
        predictions: dict[str, np.ndarray] = {}
        scale_by_method: dict[str, float] = {}
        repair_by_method: dict[str, int] = {}
        for method, (sparse, anchors) in sparse_inputs.items():
            median, scale = paired.median_align(relative, sparse, anchors)
            prediction, diagnostics = paired.call_poisson(
                poisson, median, sparse, anchors, rtol, maxiter
            )
            predictions[method] = prediction
            scale_by_method[method] = scale
            repair_by_method[method] = int(
                diagnostics.get("invalid_pixels_repaired_from_median", 0)
            )
            np.save(prediction_root / method / f"{scene}.npy", prediction.astype(np.float32))
            np.save(sparse_root / method / f"{scene}.npy", sparse.astype(np.float32))

        masks = common_masks(
            valid, one_anchors, current_anchors, maximum_anchors, margin
        )
        new_metrics: list[dict[str, Any]] = []
        for method, prediction in predictions.items():
            anchors = sparse_inputs[method][1]
            for region, mask in masks.items():
                new_metrics.append(
                    {
                        "scene": scene,
                        "method": method,
                        "method_label": METHODS[method],
                        "region": region,
                        "region_label": REGIONS[region],
                        "anchor_count": int(np.count_nonzero(anchors)),
                        "median_scale": scale_by_method[method],
                        "poisson_repaired_pixels": repair_by_method[method],
                        **paired.metrics(prediction, gt, mask),
                    }
                )
        metric_rows = [row for row in metric_rows if row.get("scene") != scene]
        metric_rows.extend(new_metrics)
        metric_rows.sort(
            key=lambda row: (str(row["scene"]), str(row["method"]), str(row["region"]))
        )
        write_csv(metrics_path, metric_rows)

        rois = select_surface_rois(
            gt,
            valid,
            args.surface_height_frac,
            args.surface_width_frac,
        )
        new_surfaces = [
            surface_measurement(
                scene,
                roi,
                gt,
                valid,
                predictions["current_4line"],
                predictions["max_coverage_4line"],
                current_rows,
                maximum_rows,
            )
            for roi in rois
        ]
        surface_rows = [row for row in surface_rows if row.get("scene") != scene]
        surface_rows.extend(new_surfaces)
        surface_rows.sort(key=lambda row: (str(row["scene"]), int(float(row["top_px"]))))
        write_csv(surfaces_path, surface_rows)

        visual_root = args.output_dir / "visual_comparisons"
        depth_panel(
            scene,
            rgb,
            gt,
            valid,
            predictions,
            current_rows,
            maximum_rows,
            rois,
            new_metrics,
            visual_root / f"{scene}__depth_maps.png",
            args.plot_max_depth_m,
            args.plot_error_max_m,
        )
        surface_panel(
            scene,
            rgb,
            gt,
            valid,
            predictions,
            rois,
            new_surfaces,
            visual_root / f"{scene}__surface_measurements.png",
            args.plot_max_depth_m,
        )

        current_primary = metric_lookup(
            new_metrics, scene, "current_4line", "outside_original_line_common"
        )
        maximum_primary = metric_lookup(
            new_metrics, scene, "max_coverage_4line", "outside_original_line_common"
        )
        gain = 100.0 * (
            float(current_primary["rmse_m"]) - float(maximum_primary["rmse_m"])
        ) / float(current_primary["rmse_m"])
        print(
            f"[{index}/{len(scenes)}] {scene} current->max-coverage "
            f"RMSE {float(current_primary['rmse_m']):.3f}->"
            f"{float(maximum_primary['rmse_m']):.3f} m ({gain:+.2f}%)",
            flush=True,
        )

    selected = set(scenes)
    chosen_metrics = [row for row in metric_rows if str(row["scene"]) in selected]
    chosen_surfaces = [row for row in surface_rows if str(row["scene"]) in selected]
    expected_metrics = len(scenes) * len(METHODS) * len(REGIONS)
    expected_surfaces = len(scenes) * len(SURFACE_BANDS)
    if len(chosen_metrics) != expected_metrics:
        raise RuntimeError(
            f"Incomplete metric table: {len(chosen_metrics)} rows, expected {expected_metrics}"
        )
    if len(chosen_surfaces) != expected_surfaces:
        raise RuntimeError(
            f"Incomplete surface table: {len(chosen_surfaces)} rows, expected {expected_surfaces}"
        )
    summary_rows = aggregate_metrics(chosen_metrics)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_report(
        args.output_dir / "placement_comparison_report.md",
        scenes,
        chosen_metrics,
        summary_rows,
        chosen_surfaces,
    )
    print("\n===== CURRENT VS MAXIMUM-COVERAGE PLACEMENT =====\n")
    print(
        (args.output_dir / "placement_comparison_report.md").read_text(
            encoding="utf-8"
        )
    )
    print(f"Full report: {args.output_dir / 'placement_comparison_report.md'}")
    print(f"Depth and surface panels: {args.output_dir / 'visual_comparisons'}")
    print(f"Surface measurements: {surfaces_path}")


if __name__ == "__main__":
    main()
