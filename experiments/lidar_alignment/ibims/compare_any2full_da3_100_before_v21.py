"""Matched 100-scene comparison of Any2Full and DA3 LiDAR alignments on iBims.

Every method uses the same V2 sparse-LiDAR map. Dense ground truth is used only
for evaluation, never for fitting. Positive surplus means a DA3 method has lower
error than Any2Full.
"""

import argparse
import csv
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.optimize import least_squares, minimize
from scipy.stats import spearmanr

from depth_anything_3.alignment.poisson_alignment import poisson_align


METHOD_LABELS = {
    "any2full": "Any2Full",
    "da3_median": "DA3 + median scale",
    "da3_ls": "DA3 + affine LS",
    "da3_log_ls": "DA3 + positive log-LS",
    "da3_huber": "DA3 + Huber",
    "da3_ls_poisson": "DA3 + affine LS + Poisson",
}

REGION_LABELS = {
    "all": "All valid pixels",
    "non_anchor": "Non-anchor pixels",
    "inside_support": "Inside LiDAR depth support",
    "below_support": "Below LiDAR depth support",
    "above_support": "Above LiDAR depth support",
    "outside_support": "Outside LiDAR depth support",
    "near_0_2m": "Near field (0–2 m)",
    "anchors": "LiDAR anchor pixels",
}

METRIC_NAMES = ("rmse_m", "mae_m", "absrel_pct", "rmsrel_pct", "bias_m")


def parse_arguments():
    environment_root = os.environ.get("DA3_LIDAR_DATA_ROOT")
    repository_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(
        description=(
            "Compare Any2Full with DA3 median, affine, log-affine, Huber, "
            "and affine-plus-Poisson alignment on matched iBims V2 inputs."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(environment_root) if environment_root else None,
        required=environment_root is None,
        help="Any2Full root containing datasets/ and experiments/ibims_replication/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repository_root
            / "experiments/lidar_alignment/outputs/comparison_any2full_da3_v2_100"
        ),
    )
    parser.add_argument("--scene", help="Evaluate one scene only, for a smoke test.")
    parser.add_argument("--limit", type=int, help="Evaluate the first N matched scenes.")
    parser.add_argument(
        "--skip-poisson",
        action="store_true",
        help="Skip the slower Poisson method.",
    )
    parser.add_argument("--poisson-rtol", type=float, default=1e-6)
    parser.add_argument("--poisson-maxiter", type=int, default=1000)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check matched file counts without evaluating.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Do not generate plots and best/median/worst scene panels.",
    )
    return parser.parse_args()


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ibims_ground_truth(path):
    record = loadmat(path)["data"][0, 0]
    ground_truth = np.squeeze(record["depth"]).astype(np.float64)
    valid = np.squeeze(record["mask_invalid"]).astype(bool)
    valid &= np.squeeze(record["mask_transp"]).astype(bool)
    valid &= np.isfinite(ground_truth) & (ground_truth > 0)
    return ground_truth, valid


def load_array(path, expected_shape=None):
    array = np.squeeze(np.load(path)).astype(np.float64)
    if array.ndim != 2:
        raise ValueError(f"{path} has shape {array.shape}; expected a 2D array.")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{path} has shape {array.shape}; expected {expected_shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite values.")
    return array


def resolve_da3_path(directory, scene):
    candidates = [directory / f"{scene}_da3small.npy", directory / f"{scene}.npy"]
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one DA3 prediction for {scene}; checked {candidates}."
        )
    return existing[0]


def resolve_rgb_path(directory, scene):
    for suffix in (".png", ".jpg", ".jpeg"):
        path = directory / f"{scene}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing RGB image for {scene} in {directory}.")


def audit_inputs(data_root):
    base = data_root / "experiments/ibims_replication"
    directories = {
        "ground_truth": data_root / "datasets/ibims1/ibims1_core_mat",
        "rgb": data_root / "datasets/ibims1/ibims1_core_raw/rgb",
        "sparse_v2": base / "v2_sensor",
        "any2full_v2": base / "predictions_v2_sensor",
        "da3": base / "da3_bridge_all",
    }
    missing_directories = [str(path) for path in directories.values() if not path.is_dir()]
    if missing_directories:
        raise FileNotFoundError(
            "Required directories are missing:\n  " + "\n  ".join(missing_directories)
        )

    ground_truth_scenes = {
        path.stem for path in directories["ground_truth"].glob("*.mat")
    }
    sparse_scenes = {path.stem for path in directories["sparse_v2"].glob("*.npy")}
    any2full_scenes = {
        path.stem for path in directories["any2full_v2"].glob("*.npy")
    }
    da3_scenes = set()
    for path in directories["da3"].glob("*.npy"):
        suffix = "_da3small"
        da3_scenes.add(path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem)
    rgb_scenes = {
        path.stem
        for suffix in ("*.png", "*.jpg", "*.jpeg")
        for path in directories["rgb"].glob(suffix)
    }

    scene_sets = {
        "ground_truth": ground_truth_scenes,
        "sparse_v2": sparse_scenes,
        "any2full_v2": any2full_scenes,
        "da3": da3_scenes,
        "rgb": rgb_scenes,
    }
    matched = set.intersection(*scene_sets.values())
    union = set.union(*scene_sets.values())
    missing_by_source = {
        source: sorted(union - scenes) for source, scenes in scene_sets.items()
    }

    print("\n========== INPUT AUDIT ==========")
    for source, scenes in scene_sets.items():
        print(f"{source:16s}: {len(scenes):3d}")
    print(f"{'matched':16s}: {len(matched):3d}")

    if len(matched) != 100:
        print("\nWARNING: expected 100 matched scenes.")
        for source, missing in missing_by_source.items():
            if missing:
                print(f"Missing from {source}: {', '.join(missing[:10])}")

    return directories, sorted(matched), missing_by_source


def fit_median_scale(relative_depth, metric_depth):
    usable = np.isfinite(relative_depth) & (relative_depth > 0)
    if usable.sum() == 0:
        raise ValueError("Median scale fit has no positive DA3 anchors.")
    scale = np.median(metric_depth[usable] / relative_depth[usable])
    return float(scale), 0.0


def fit_affine_ls(relative_depth, metric_depth):
    matrix = np.column_stack((relative_depth, np.ones_like(relative_depth)))
    scale, shift = np.linalg.lstsq(matrix, metric_depth, rcond=None)[0]
    return float(scale), float(shift)


def fit_positive_log_affine(relative_depth, metric_depth, domain_min, domain_max):
    domain_size = domain_max - domain_min
    if domain_size <= 0:
        raise ValueError("DA3 prediction has no usable variation.")

    normalized = (relative_depth - domain_min) / domain_size
    initial_scale, initial_shift = fit_affine_ls(relative_depth, metric_depth)
    initial_left = max(initial_scale * domain_min + initial_shift, 1e-4)
    initial_right = max(initial_scale * domain_max + initial_shift, 1e-4)

    def objective(log_endpoints):
        left, right = np.exp(log_endpoints)
        prediction = left * (1.0 - normalized) + right * normalized
        return np.mean((np.log(prediction) - np.log(metric_depth)) ** 2)

    result = minimize(
        objective,
        np.log([initial_left, initial_right]),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-11, "fatol": 1e-13},
    )
    if not result.success:
        warnings.warn(f"Positive log-affine optimizer: {result.message}")
    left, right = np.exp(result.x)
    scale = (right - left) / domain_size
    shift = left - scale * domain_min
    return float(scale), float(shift), float(np.sqrt(result.fun))


def fit_huber(relative_depth, metric_depth, initial_scale, initial_shift, delta=0.10):
    def residual(parameters):
        scale, shift = parameters
        return scale * relative_depth + shift - metric_depth

    result = least_squares(
        residual,
        x0=np.array([initial_scale, initial_shift], dtype=np.float64),
        loss="huber",
        f_scale=delta,
        max_nfev=5000,
    )
    if not result.success:
        warnings.warn(f"Huber optimizer: {result.message}")
    return float(result.x[0]), float(result.x[1])


def calculate_metrics(prediction, ground_truth, mask):
    count = int(mask.sum())
    if count == 0:
        return None
    predicted = prediction[mask]
    actual = ground_truth[mask]
    error = predicted - actual
    relative = error / actual
    return {
        "n": count,
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
        "absrel_pct": float(100.0 * np.mean(np.abs(relative))),
        "rmsrel_pct": float(100.0 * np.sqrt(np.mean(relative**2))),
        "bias_m": float(np.mean(error)),
        "nonpositive_pct": float(100.0 * np.mean(predicted <= 0)),
    }


def make_regions(valid, anchor_mask, ground_truth, support_min, support_max):
    below = valid & (ground_truth < support_min)
    above = valid & (ground_truth > support_max)
    return {
        "all": valid,
        "non_anchor": valid & ~anchor_mask,
        "inside_support": valid & (ground_truth >= support_min) & (ground_truth <= support_max),
        "below_support": below,
        "above_support": above,
        "outside_support": below | above,
        "near_0_2m": valid & (ground_truth < 2.0),
        "anchors": valid & anchor_mask,
    }


def build_predictions(da3, sparse, anchor_mask, include_poisson, poisson_rtol, poisson_maxiter):
    relative_anchors = da3[anchor_mask]
    metric_anchors = sparse[anchor_mask]
    valid_da3 = np.isfinite(da3)
    domain_min = float(da3[valid_da3].min())
    domain_max = float(da3[valid_da3].max())

    median_scale, median_shift = fit_median_scale(relative_anchors, metric_anchors)
    ls_scale, ls_shift = fit_affine_ls(relative_anchors, metric_anchors)
    log_scale, log_shift, log_rmse = fit_positive_log_affine(
        relative_anchors, metric_anchors, domain_min, domain_max
    )
    huber_scale, huber_shift = fit_huber(
        relative_anchors, metric_anchors, ls_scale, ls_shift
    )

    predictions = {
        "da3_median": median_scale * da3 + median_shift,
        "da3_ls": ls_scale * da3 + ls_shift,
        "da3_log_ls": log_scale * da3 + log_shift,
        "da3_huber": huber_scale * da3 + huber_shift,
    }
    poisson_diagnostics = None
    if include_poisson:
        predictions["da3_ls_poisson"], poisson_diagnostics = poisson_align(
            predictions["da3_ls"],
            sparse,
            anchor_mask,
            rtol=poisson_rtol,
            maxiter=poisson_maxiter,
        )

    matrix = np.column_stack((relative_anchors, np.ones_like(relative_anchors)))
    fit_diagnostics = {
        "anchor_count": int(anchor_mask.sum()),
        "lidar_min_m": float(metric_anchors.min()),
        "lidar_max_m": float(metric_anchors.max()),
        "lidar_span_m": float(metric_anchors.max() - metric_anchors.min()),
        "da3_anchor_min": float(relative_anchors.min()),
        "da3_anchor_max": float(relative_anchors.max()),
        "da3_anchor_std": float(relative_anchors.std()),
        "affine_design_condition": float(np.linalg.cond(matrix)),
        "median_scale": median_scale,
        "median_shift": median_shift,
        "ls_scale": ls_scale,
        "ls_shift": ls_shift,
        "log_ls_scale": log_scale,
        "log_ls_shift": log_shift,
        "log_anchor_rmse": log_rmse,
        "huber_scale": huber_scale,
        "huber_shift": huber_shift,
    }
    return predictions, fit_diagnostics, poisson_diagnostics


def summarize_metrics(metric_rows):
    grouped = defaultdict(list)
    for row in metric_rows:
        grouped[(row["method"], row["region"])].append(row)

    summaries = []
    for (method, region), rows in grouped.items():
        total_n = sum(row["n"] for row in rows)
        summary = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "region": region,
            "region_label": REGION_LABELS[region],
            "scene_count": len(rows),
            "pixel_count": total_n,
        }
        for metric in METRIC_NAMES:
            values = np.array([row[metric] for row in rows], dtype=np.float64)
            summary[f"macro_mean_{metric}"] = float(values.mean())
            summary[f"macro_median_{metric}"] = float(np.median(values))

        summary["pooled_rmse_m"] = float(
            np.sqrt(sum(row["n"] * row["rmse_m"] ** 2 for row in rows) / total_n)
        )
        summary["pooled_mae_m"] = float(
            sum(row["n"] * row["mae_m"] for row in rows) / total_n
        )
        summary["pooled_absrel_pct"] = float(
            sum(row["n"] * row["absrel_pct"] for row in rows) / total_n
        )
        summary["pooled_rmsrel_pct"] = float(
            np.sqrt(
                sum(row["n"] * (row["rmsrel_pct"] / 100.0) ** 2 for row in rows)
                / total_n
            )
            * 100.0
        )
        summary["pooled_bias_m"] = float(
            sum(row["n"] * row["bias_m"] for row in rows) / total_n
        )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (row["region"], row["method"]))


def calculate_surplus(metric_rows):
    lookup = {
        (row["scene"], row["region"], row["method"]): row for row in metric_rows
    }
    scenes_regions = sorted({(row["scene"], row["region"]) for row in metric_rows})
    methods = [method for method in METHOD_LABELS if method != "any2full"]
    rows = []
    for scene, region in scenes_regions:
        baseline = lookup.get((scene, region, "any2full"))
        if baseline is None:
            continue
        for method in methods:
            candidate = lookup.get((scene, region, method))
            if candidate is None:
                continue
            rows.append(
                {
                    "scene": scene,
                    "region": region,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "rmse_surplus_m": baseline["rmse_m"] - candidate["rmse_m"],
                    "absrel_surplus_pp": (
                        baseline["absrel_pct"] - candidate["absrel_pct"]
                    ),
                    "bias_abs_surplus_m": (
                        abs(baseline["bias_m"]) - abs(candidate["bias_m"])
                    ),
                    "wins_rmse": int(candidate["rmse_m"] < baseline["rmse_m"]),
                    "wins_absrel": int(
                        candidate["absrel_pct"] < baseline["absrel_pct"]
                    ),
                }
            )
    return rows


def summarize_surplus(surplus_rows):
    grouped = defaultdict(list)
    for row in surplus_rows:
        grouped[(row["method"], row["region"])].append(row)
    summaries = []
    for (method, region), rows in grouped.items():
        absrel = np.array([row["absrel_surplus_pp"] for row in rows])
        rmse = np.array([row["rmse_surplus_m"] for row in rows])
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "region": region,
                "region_label": REGION_LABELS[region],
                "scene_count": len(rows),
                "mean_absrel_surplus_pp": float(absrel.mean()),
                "median_absrel_surplus_pp": float(np.median(absrel)),
                "absrel_win_rate_pct": float(
                    100.0 * np.mean([row["wins_absrel"] for row in rows])
                ),
                "mean_rmse_surplus_m": float(rmse.mean()),
                "median_rmse_surplus_m": float(np.median(rmse)),
                "rmse_win_rate_pct": float(
                    100.0 * np.mean([row["wins_rmse"] for row in rows])
                ),
            }
        )
    return sorted(summaries, key=lambda row: (row["region"], row["method"]))


def support_correlations(metric_rows, fit_rows):
    fit_lookup = {row["scene"]: row for row in fit_rows}
    grouped = defaultdict(list)
    for row in metric_rows:
        if row["region"] == "outside_support":
            grouped[row["method"]].append(row)

    output = []
    predictors = ("lidar_span_m", "da3_anchor_std", "anchor_count")
    for method, rows in grouped.items():
        for predictor in predictors:
            x = np.array([fit_lookup[row["scene"]][predictor] for row in rows])
            y = np.array([row["absrel_pct"] for row in rows])
            usable = np.isfinite(x) & np.isfinite(y)
            if usable.sum() < 3 or np.unique(x[usable]).size < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(x[usable], y[usable])
            output.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "predictor": predictor,
                    "region": "outside_support",
                    "scene_count": int(usable.sum()),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )
    return output


def print_main_tables(summary_rows, surplus_summary):
    print("\n========== MACRO MEAN: ALL VALID PIXELS ==========")
    selected = [row for row in summary_rows if row["region"] == "all"]
    selected.sort(key=lambda row: row["macro_mean_absrel_pct"])
    for row in selected:
        print(
            f"{row['method_label']:31s} "
            f"AbsRel={row['macro_mean_absrel_pct']:7.3f}%  "
            f"RMSE={row['macro_mean_rmse_m']:7.3f} m  "
            f"Bias={row['macro_mean_bias_m']:+7.3f} m"
        )

    print("\n========== DA3 SURPLUS OVER ANY2FULL ==========")
    selected = [row for row in surplus_summary if row["region"] == "all"]
    selected.sort(key=lambda row: row["mean_absrel_surplus_pp"], reverse=True)
    for row in selected:
        print(
            f"{row['method_label']:31s} "
            f"AbsRel surplus={row['mean_absrel_surplus_pp']:+7.3f} pp  "
            f"win rate={row['absrel_win_rate_pct']:6.1f}%  "
            f"RMSE surplus={row['mean_rmse_surplus_m']:+7.3f} m"
        )

    print("\n========== OUTSIDE LIDAR DEPTH SUPPORT ==========")
    selected = [row for row in surplus_summary if row["region"] == "outside_support"]
    selected.sort(key=lambda row: row["mean_absrel_surplus_pp"], reverse=True)
    for row in selected:
        print(
            f"{row['method_label']:31s} "
            f"AbsRel surplus={row['mean_absrel_surplus_pp']:+7.3f} pp  "
            f"win rate={row['absrel_win_rate_pct']:6.1f}%"
        )


def plot_summary(summary_rows, output_directory):
    rows = [row for row in summary_rows if row["region"] == "all"]
    rows.sort(key=lambda row: row["macro_mean_absrel_pct"])
    labels = [row["method_label"] for row in rows]
    absrel = [row["macro_mean_absrel_pct"] for row in rows]
    rmse = [row["macro_mean_rmse_m"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(labels, absrel, color="#3366cc")
    axes[0].set_xlabel("Macro mean AbsRel (%)")
    axes[0].set_title("100-scene iBims accuracy")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(labels, rmse, color="#dc3912")
    axes[1].set_xlabel("Macro mean RMSE (m)")
    axes[1].set_title("Matched V2 LiDAR input")
    axes[1].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "method_summary.png", dpi=180)
    plt.close(figure)


def plot_scatter_and_surplus(metric_rows, best_method, output_directory):
    lookup = {
        (row["scene"], row["method"]): row
        for row in metric_rows
        if row["region"] == "all"
    }
    scenes = sorted(
        scene
        for scene, method in lookup
        if method == "any2full" and (scene, best_method) in lookup
    )
    baseline = np.array([lookup[(scene, "any2full")]["absrel_pct"] for scene in scenes])
    candidate = np.array([lookup[(scene, best_method)]["absrel_pct"] for scene in scenes])
    surplus = baseline - candidate

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    maximum = float(max(baseline.max(), candidate.max()))
    axes[0].scatter(baseline, candidate, alpha=0.75, s=28)
    axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=1)
    axes[0].set_xlabel("Any2Full AbsRel (%)")
    axes[0].set_ylabel(f"{METHOD_LABELS[best_method]} AbsRel (%)")
    axes[0].set_title("Per-scene comparison")
    axes[0].grid(alpha=0.25)
    axes[1].hist(surplus, bins=20, color="#109618", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("AbsRel surplus over Any2Full (percentage points)")
    axes[1].set_ylabel("Scenes")
    axes[1].set_title(
        f"Positive means {METHOD_LABELS[best_method]} is better"
    )
    figure.tight_layout()
    figure.savefig(output_directory / "best_method_vs_any2full.png", dpi=180)
    plt.close(figure)


def plot_support_relationship(metric_rows, fit_rows, best_method, output_directory):
    fit_lookup = {row["scene"]: row for row in fit_rows}
    lookup = {
        (row["scene"], row["method"]): row
        for row in metric_rows
        if row["region"] == "outside_support"
    }
    scenes = sorted(
        scene
        for scene, method in lookup
        if method == "any2full" and (scene, best_method) in lookup
    )
    span = np.array([fit_lookup[scene]["lidar_span_m"] for scene in scenes])
    a2f = np.array([lookup[(scene, "any2full")]["absrel_pct"] for scene in scenes])
    da3 = np.array([lookup[(scene, best_method)]["absrel_pct"] for scene in scenes])
    rho_a2f = spearmanr(span, a2f).statistic
    rho_da3 = spearmanr(span, da3).statistic

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    axes[0].scatter(span, a2f, alpha=0.75, s=28, color="#dc3912")
    axes[0].set_title(f"Any2Full, Spearman ρ={rho_a2f:.3f}")
    axes[1].scatter(span, da3, alpha=0.75, s=28, color="#3366cc")
    axes[1].set_title(f"{METHOD_LABELS[best_method]}, Spearman ρ={rho_da3:.3f}")
    for axis in axes:
        axis.set_xlabel("LiDAR anchor depth span (m)")
        axis.set_ylabel("Outside-support AbsRel (%)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "support_span_vs_outside_error.png", dpi=180)
    plt.close(figure)


def overlay_rgb_lidar(rgb, sparse, anchor_mask):
    overlay = np.asarray(rgb).copy()
    return overlay, np.where(anchor_mask, sparse, np.nan)


def create_scene_visual(
    scene,
    rank,
    best_method,
    directories,
    output_directory,
    include_poisson,
    poisson_rtol,
    poisson_maxiter,
):
    ground_truth, valid = load_ibims_ground_truth(
        directories["ground_truth"] / f"{scene}.mat"
    )
    sparse = load_array(directories["sparse_v2"] / f"{scene}.npy", ground_truth.shape)
    any2full = load_array(
        directories["any2full_v2"] / f"{scene}.npy", ground_truth.shape
    )
    da3 = load_array(resolve_da3_path(directories["da3"], scene), ground_truth.shape)
    anchor_mask = np.isfinite(sparse) & (sparse > 0) & np.isfinite(da3) & (da3 > 0)
    predictions, _, _ = build_predictions(
        da3,
        sparse,
        anchor_mask,
        include_poisson,
        poisson_rtol,
        poisson_maxiter,
    )
    candidate = predictions[best_method]
    rgb = Image.open(resolve_rgb_path(directories["rgb"], scene)).convert("RGB")
    rgb = rgb.resize((ground_truth.shape[1], ground_truth.shape[0]))
    overlay, sparse_display = overlay_rgb_lidar(rgb, sparse, anchor_mask)

    gt_display = np.where(valid, ground_truth, np.nan)
    any2full_display = np.where(valid, any2full, np.nan)
    candidate_display = np.where(valid, candidate, np.nan)
    error_a2f = np.where(valid, np.abs(any2full - ground_truth), np.nan)
    error_da3 = np.where(valid, np.abs(candidate - ground_truth), np.nan)
    improvement = np.where(valid, error_a2f - error_da3, np.nan)
    depth_min, depth_max = np.nanpercentile(gt_display, [1, 99])
    error_max = np.nanpercentile(np.concatenate([error_a2f[valid], error_da3[valid]]), 95)
    advantage_max = np.nanpercentile(np.abs(improvement[valid]), 95)

    figure, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes[0, 0].imshow(overlay)
    yy, xx = np.nonzero(anchor_mask)
    scatter = axes[0, 0].scatter(
        xx, yy, c=sparse[anchor_mask], s=5, cmap="turbo",
        vmin=depth_min, vmax=depth_max
    )
    axes[0, 0].set_title(f"RGB + V2 LiDAR ({anchor_mask.sum()} anchors)")
    image_gt = axes[0, 1].imshow(gt_display, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[0, 1].set_title("Ground-truth depth")
    axes[0, 2].imshow(any2full_display, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[0, 2].set_title("Any2Full prediction")
    axes[0, 3].imshow(candidate_display, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[0, 3].set_title(METHOD_LABELS[best_method])
    axes[1, 0].imshow(sparse_display, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[1, 0].set_title("Sparse metric depth")
    image_error = axes[1, 1].imshow(error_a2f, cmap="magma", vmin=0, vmax=error_max)
    axes[1, 1].set_title("Any2Full absolute error")
    axes[1, 2].imshow(error_da3, cmap="magma", vmin=0, vmax=error_max)
    axes[1, 2].set_title("DA3 absolute error")
    image_advantage = axes[1, 3].imshow(
        improvement, cmap="RdBu", vmin=-advantage_max, vmax=advantage_max
    )
    axes[1, 3].set_title("Per-pixel advantage: A2F error − DA3 error")
    for axis in axes.ravel():
        axis.axis("off")
    figure.colorbar(scatter, ax=axes[0, 0], fraction=0.046, label="Depth (m)")
    figure.colorbar(image_gt, ax=axes[0, 1:4], fraction=0.02, label="Depth (m)")
    figure.colorbar(image_error, ax=axes[1, 1:3], fraction=0.03, label="Absolute error (m)")
    figure.colorbar(
        image_advantage, ax=axes[1, 3], fraction=0.046, label="Improvement (m)"
    )
    figure.suptitle(f"{rank.title()} surplus scene: {scene}", fontsize=16)
    figure.subplots_adjust(top=0.91, wspace=0.08, hspace=0.12)
    visual_directory = output_directory / "visuals"
    visual_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        visual_directory / f"{rank}_{scene}_{best_method}.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def select_visual_scenes(surplus_rows, best_method):
    rows = [
        row
        for row in surplus_rows
        if row["region"] == "all" and row["method"] == best_method
    ]
    rows.sort(key=lambda row: row["absrel_surplus_pp"])
    return {
        "worst": rows[0]["scene"],
        "median": rows[len(rows) // 2]["scene"],
        "best": rows[-1]["scene"],
    }


def main():
    arguments = parse_arguments()
    data_root = arguments.data_root.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    directories, scenes, missing_by_source = audit_inputs(data_root)

    if arguments.scene:
        if arguments.scene not in scenes:
            raise ValueError(f"Scene is not fully matched: {arguments.scene}")
        scenes = [arguments.scene]
    elif arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError("--limit must be positive.")
        scenes = scenes[: arguments.limit]

    if arguments.preflight_only:
        return

    if not scenes:
        raise RuntimeError("No matched scenes were found.")
    output_directory.mkdir(parents=True, exist_ok=True)
    include_poisson = not arguments.skip_poisson
    active_methods = list(METHOD_LABELS)
    if not include_poisson:
        active_methods.remove("da3_ls_poisson")

    metric_rows = []
    fit_rows = []
    poisson_rows = []

    print(f"\nEvaluating {len(scenes)} scene(s)...")
    for index, scene in enumerate(scenes, start=1):
        ground_truth, valid = load_ibims_ground_truth(
            directories["ground_truth"] / f"{scene}.mat"
        )
        sparse = load_array(
            directories["sparse_v2"] / f"{scene}.npy", ground_truth.shape
        )
        any2full = load_array(
            directories["any2full_v2"] / f"{scene}.npy", ground_truth.shape
        )
        da3 = load_array(resolve_da3_path(directories["da3"], scene), ground_truth.shape)

        anchor_mask = (
            np.isfinite(sparse)
            & (sparse > 0)
            & np.isfinite(da3)
            & (da3 > 0)
        )
        if anchor_mask.sum() < 3:
            raise ValueError(f"{scene} has only {anchor_mask.sum()} usable anchors.")

        predictions, fit_diagnostics, poisson_diagnostics = build_predictions(
            da3,
            sparse,
            anchor_mask,
            include_poisson,
            arguments.poisson_rtol,
            arguments.poisson_maxiter,
        )
        predictions["any2full"] = any2full
        fit_diagnostics["scene"] = scene
        fit_diagnostics["valid_pixel_count"] = int(valid.sum())
        fit_rows.append(fit_diagnostics)

        if poisson_diagnostics is not None:
            poisson_diagnostics["scene"] = scene
            poisson_rows.append(poisson_diagnostics)

        regions = make_regions(
            valid,
            anchor_mask,
            ground_truth,
            fit_diagnostics["lidar_min_m"],
            fit_diagnostics["lidar_max_m"],
        )
        for method in active_methods:
            prediction = predictions[method]
            if not np.isfinite(prediction).all():
                raise ValueError(f"{scene}/{method} contains non-finite values.")
            for region, mask in regions.items():
                metrics = calculate_metrics(prediction, ground_truth, mask)
                if metrics is None:
                    continue
                metric_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "region": region,
                        "region_label": REGION_LABELS[region],
                        **metrics,
                    }
                )
        print(
            f"[{index:3d}/{len(scenes):3d}] {scene:24s} "
            f"anchors={fit_diagnostics['anchor_count']:4d} "
            f"span={fit_diagnostics['lidar_span_m']:.3f} m"
        )

    summary_rows = summarize_metrics(metric_rows)
    surplus_rows = calculate_surplus(metric_rows)
    surplus_summary = summarize_surplus(surplus_rows)
    correlation_rows = support_correlations(metric_rows, fit_rows)

    write_csv(output_directory / "per_scene_metrics.csv", metric_rows)
    write_csv(output_directory / "per_scene_fits.csv", fit_rows)
    if poisson_rows:
        write_csv(output_directory / "poisson_diagnostics.csv", poisson_rows)
    write_csv(output_directory / "summary_metrics.csv", summary_rows)
    write_csv(output_directory / "per_scene_surplus.csv", surplus_rows)
    write_csv(output_directory / "summary_surplus.csv", surplus_summary)
    write_csv(output_directory / "support_correlations.csv", correlation_rows)

    configuration = {
        "data_root": str(data_root),
        "scene_count": len(scenes),
        "include_poisson": include_poisson,
        "poisson_rtol": arguments.poisson_rtol,
        "poisson_maxiter": arguments.poisson_maxiter,
        "methods": active_methods,
        "missing_by_source": missing_by_source,
    }
    with (output_directory / "run_configuration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, indent=2)

    print_main_tables(summary_rows, surplus_summary)

    best_row = min(
        (
            row
            for row in summary_rows
            if row["region"] == "all" and row["method"] != "any2full"
        ),
        key=lambda row: row["macro_mean_absrel_pct"],
    )
    best_method = best_row["method"]
    print(f"\nBest DA3 method by macro all-pixel AbsRel: {METHOD_LABELS[best_method]}")

    if not arguments.no_visuals:
        plot_summary(summary_rows, output_directory)
        plot_scatter_and_surplus(metric_rows, best_method, output_directory)
        plot_support_relationship(metric_rows, fit_rows, best_method, output_directory)
        selected_scenes = select_visual_scenes(surplus_rows, best_method)
        for rank, scene in selected_scenes.items():
            create_scene_visual(
                scene,
                rank,
                best_method,
                directories,
                output_directory,
                include_poisson,
                arguments.poisson_rtol,
                arguments.poisson_maxiter,
            )
        print(f"Visual scenes: {selected_scenes}")

    print(f"\nSaved comparison results to: {output_directory}")


if __name__ == "__main__":
    main()
