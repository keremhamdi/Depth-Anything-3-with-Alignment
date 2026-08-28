"""Diagnose catastrophic outside-support failures for Any2Full and DA3+median.

The script uses the exact V2.1 sparse depth map for both methods. It selects
catastrophic scenes from the benchmark CSV and writes a spatial diagnostic
panel for every selected scene, plus a paired summary figure and CSV.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat


A2F_LABEL = "Any2Full"
DA3_LABEL = "DA3 + median scale"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path(os.environ["DA3_LIDAR_DATA_ROOT"])
            if os.environ.get("DA3_LIDAR_DATA_ROOT")
            else None
        ),
        required=not bool(os.environ.get("DA3_LIDAR_DATA_ROOT")),
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold-absrel",
        type=float,
        default=40.0,
        help="A scene is catastrophic when outside-support AbsRel reaches this value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Also include the top K worst scenes and largest method disagreements.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Always include this scene. May be passed more than once.",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_gt(path):
    record = loadmat(path)["data"][0, 0]
    gt = np.squeeze(record["depth"]).astype(np.float64)
    valid = np.squeeze(record["mask_invalid"]).astype(bool)
    if "mask_transp" in record.dtype.names:
        valid &= np.squeeze(record["mask_transp"]).astype(bool)
    valid &= np.isfinite(gt) & (gt > 0)
    return gt, valid


def load_array(path, shape):
    array = np.squeeze(np.load(path)).astype(np.float64)
    if array.shape != shape:
        raise ValueError(f"{path}: found shape {array.shape}, expected {shape}")
    return array


def resolve_da3(directory, scene):
    candidates = [
        directory / f"{scene}_da3small.npy",
        directory / f"{scene}.npy",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one DA3 file for {scene}; checked {candidates}")
    return found[0]


def resolve_rgb(directory, scene):
    for suffix in (".png", ".jpg", ".jpeg"):
        path = directory / f"{scene}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"RGB not found for {scene} in {directory}")


def directories(data_root):
    base = data_root / "experiments/ibims_replication"
    result = {
        "gt": data_root / "datasets/ibims1/ibims1_core_mat",
        "rgb": data_root / "datasets/ibims1/ibims1_core_raw/rgb",
        "sparse": base / "v2_1_sensor",
        "a2f": base / "predictions_v2_1_sensor",
        "da3": base / "da3_bridge_all",
    }
    missing = [str(path) for path in result.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing input directories:\n  " + "\n  ".join(missing))
    return result


def metric_values(prediction, gt, mask):
    if not np.any(mask):
        return {
            "n": 0,
            "absrel_pct": np.nan,
            "rmse_m": np.nan,
            "mae_m": np.nan,
            "bias_m": np.nan,
            "p90_abs_m": np.nan,
            "bad_025_pct": np.nan,
            "bad_050_pct": np.nan,
            "bad_100_pct": np.nan,
            "delta1_pct": np.nan,
        }
    actual = gt[mask]
    predicted = prediction[mask]
    error = predicted - actual
    absolute = np.abs(error)
    ratio = np.maximum(
        np.maximum(predicted, 1e-8) / actual,
        actual / np.maximum(predicted, 1e-8),
    )
    return {
        "n": int(mask.sum()),
        "absrel_pct": float(100 * np.mean(absolute / actual)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "mae_m": float(np.mean(absolute)),
        "bias_m": float(np.mean(error)),
        "p90_abs_m": float(np.percentile(absolute, 90)),
        "bad_025_pct": float(100 * np.mean(absolute > 0.25)),
        "bad_050_pct": float(100 * np.mean(absolute > 0.50)),
        "bad_100_pct": float(100 * np.mean(absolute > 1.00)),
        "delta1_pct": float(100 * np.mean(ratio < 1.25)),
    }


def paired_rows(metrics_csv):
    selected = {}
    for row in read_csv(metrics_csv):
        if row.get("region") != "outside_support":
            continue
        label = row.get("method_label", row.get("method", ""))
        if label not in (A2F_LABEL, DA3_LABEL):
            continue
        selected.setdefault(row["scene"], {})[label] = row
    paired = {}
    for scene, methods in selected.items():
        if A2F_LABEL in methods and DA3_LABEL in methods:
            paired[scene] = {
                "a2f": float(methods[A2F_LABEL]["absrel_pct"]),
                "da3": float(methods[DA3_LABEL]["absrel_pct"]),
            }
    if not paired:
        raise ValueError("No paired outside_support rows for Any2Full and DA3 + median scale")
    return paired


def classify(a2f, da3, threshold):
    if a2f >= threshold and da3 >= threshold:
        return "both_catastrophic"
    if a2f >= threshold:
        return "any2full_only"
    if da3 >= threshold:
        return "da3_only"
    return "large_disagreement"


def select_scenes(paired, threshold, top_k, forced):
    chosen = set(forced)
    for scene, values in paired.items():
        if max(values["a2f"], values["da3"]) >= threshold:
            chosen.add(scene)

    by_a2f = sorted(paired, key=lambda s: paired[s]["a2f"], reverse=True)
    by_da3 = sorted(paired, key=lambda s: paired[s]["da3"], reverse=True)
    by_a2f_gap = sorted(
        paired, key=lambda s: paired[s]["a2f"] - paired[s]["da3"], reverse=True
    )
    by_da3_gap = sorted(
        paired, key=lambda s: paired[s]["da3"] - paired[s]["a2f"], reverse=True
    )
    for ranking in (by_a2f, by_da3, by_a2f_gap, by_da3_gap):
        chosen.update(ranking[:top_k])
    return sorted(chosen)


def transparent_mask(mask):
    rgba = np.zeros(mask.shape + (4,), dtype=np.float64)
    rgba[..., 0] = 0.98
    rgba[..., 1] = 0.16
    rgba[..., 2] = 0.12
    rgba[..., 3] = mask.astype(np.float64) * 0.43
    return rgba


def quantile_bins(gt_values, count=8):
    edges = np.unique(np.quantile(gt_values, np.linspace(0, 1, count + 1)))
    if len(edges) < 3:
        edges = np.linspace(float(gt_values.min()), float(gt_values.max()) + 1e-6, 3)
    return edges


def make_panel(scene, dirs, output_dir, benchmark_values, threshold):
    gt, valid = load_gt(dirs["gt"] / f"{scene}.mat")
    sparse = load_array(dirs["sparse"] / f"{scene}.npy", gt.shape)
    a2f = load_array(dirs["a2f"] / f"{scene}.npy", gt.shape)
    da3_relative = load_array(resolve_da3(dirs["da3"], scene), gt.shape)

    anchor = (
        np.isfinite(sparse)
        & (sparse > 0)
        & np.isfinite(da3_relative)
        & (da3_relative > 0)
    )
    if not np.any(anchor):
        raise ValueError(f"{scene}: no usable V2.1 anchors")
    scale = float(np.median(sparse[anchor] / da3_relative[anchor]))
    da3 = scale * da3_relative
    support_min = float(sparse[anchor].min())
    support_max = float(sparse[anchor].max())
    below = valid & (gt < support_min)
    above = valid & (gt > support_max)
    outside = below | above

    a2f_metrics = metric_values(a2f, gt, outside)
    da3_metrics = metric_values(da3, gt, outside)
    group = classify(
        benchmark_values["a2f"], benchmark_values["da3"], threshold
    )

    rgb = Image.open(resolve_rgb(dirs["rgb"], scene)).convert("RGB")
    rgb = rgb.resize((gt.shape[1], gt.shape[0]))
    rgb_array = np.asarray(rgb)

    gt_show = np.where(valid, gt, np.nan)
    a2f_show = np.where(valid, a2f, np.nan)
    da3_show = np.where(valid, da3, np.nan)
    a2f_error = np.where(outside, np.abs(a2f - gt), np.nan)
    da3_error = np.where(outside, np.abs(da3 - gt), np.nan)
    advantage = np.where(outside, np.abs(a2f - gt) - np.abs(da3 - gt), np.nan)

    depth_min, depth_max = np.nanpercentile(gt_show, [1, 99])
    combined_error = np.concatenate([a2f_error[outside], da3_error[outside]])
    error_max = max(float(np.percentile(combined_error, 95)), 0.10)
    advantage_max = max(float(np.percentile(np.abs(advantage[outside]), 95)), 0.10)

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    axes[0, 0].imshow(rgb_array)
    yy, xx = np.nonzero(anchor)
    anchors = axes[0, 0].scatter(
        xx,
        yy,
        c=sparse[anchor],
        s=7,
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
    )
    axes[0, 0].set_title(f"RGB + V2.1 anchors (n={anchor.sum()})")
    depth_image = axes[0, 1].imshow(gt_show, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[0, 1].set_title("Dense ground truth")
    axes[0, 2].imshow(a2f_show, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[0, 2].set_title("Any2Full prediction")
    axes[0, 3].imshow(da3_show, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[0, 3].set_title("DA3-Small + median scale")

    axes[1, 0].imshow(rgb_array)
    axes[1, 0].imshow(transparent_mask(outside))
    axes[1, 0].set_title(
        f"Outside support (red): {100 * outside.sum() / valid.sum():.1f}% valid pixels\n"
        f"support = [{support_min:.2f}, {support_max:.2f}] m"
    )
    error_image = axes[1, 1].imshow(
        a2f_error, cmap="magma", vmin=0, vmax=error_max
    )
    axes[1, 1].set_title(
        f"Any2Full |error| outside\n"
        f"AbsRel={a2f_metrics['absrel_pct']:.1f}%  RMSE={a2f_metrics['rmse_m']:.2f} m"
    )
    axes[1, 2].imshow(da3_error, cmap="magma", vmin=0, vmax=error_max)
    axes[1, 2].set_title(
        f"DA3 + median |error| outside\n"
        f"AbsRel={da3_metrics['absrel_pct']:.1f}%  RMSE={da3_metrics['rmse_m']:.2f} m"
    )
    advantage_image = axes[1, 3].imshow(
        advantage, cmap="RdBu_r", vmin=-advantage_max, vmax=advantage_max
    )
    axes[1, 3].set_title("A2F |error| − DA3 |error|\nblue: DA3 worse, red: DA3 better")

    actual = gt[outside]
    sample_count = min(actual.size, 25000)
    rng = np.random.default_rng(0)
    sample = rng.choice(actual.size, sample_count, replace=False)
    axes[2, 0].scatter(actual[sample], a2f[outside][sample], s=2, alpha=0.18, label="A2F")
    axes[2, 0].scatter(actual[sample], da3[outside][sample], s=2, alpha=0.18, label="DA3")
    scatter_min = float(min(actual.min(), a2f[outside].min(), da3[outside].min()))
    scatter_max = float(np.percentile(np.concatenate([actual, a2f[outside], da3[outside]]), 99))
    axes[2, 0].plot([scatter_min, scatter_max], [scatter_min, scatter_max], "k--", lw=1)
    axes[2, 0].set_xlim(scatter_min, scatter_max)
    axes[2, 0].set_ylim(scatter_min, scatter_max)
    axes[2, 0].set_xlabel("Ground-truth depth (m)")
    axes[2, 0].set_ylabel("Predicted depth (m)")
    axes[2, 0].set_title("Outside-support calibration")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.25)

    edges = quantile_bins(actual)
    centers = 0.5 * (edges[:-1] + edges[1:])
    a2f_bins = []
    da3_bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        bucket = outside & (gt >= left) & (gt <= right)
        a2f_bins.append(metric_values(a2f, gt, bucket)["absrel_pct"])
        da3_bins.append(metric_values(da3, gt, bucket)["absrel_pct"])
    axes[2, 1].plot(centers, a2f_bins, "o-", label="A2F")
    axes[2, 1].plot(centers, da3_bins, "o-", label="DA3")
    axes[2, 1].axvspan(support_min, support_max, color="grey", alpha=0.16, label="anchor range")
    axes[2, 1].set_xlabel("Ground-truth depth-bin center (m)")
    axes[2, 1].set_ylabel("AbsRel (%)")
    axes[2, 1].set_title("Where along depth the failure occurs")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.25)

    for prediction, label in ((a2f, "A2F"), (da3, "DA3")):
        values = np.sort(np.abs(prediction[outside] - actual))
        cdf = 100 * np.arange(1, len(values) + 1) / len(values)
        axes[2, 2].plot(values, cdf, label=label)
    axes[2, 2].set_xlim(0, max(float(np.percentile(combined_error, 99)), 0.2))
    axes[2, 2].set_xlabel("Absolute error (m)")
    axes[2, 2].set_ylabel("Outside pixels at or below error (%)")
    axes[2, 2].set_title("Absolute-error CDF")
    axes[2, 2].legend()
    axes[2, 2].grid(alpha=0.25)

    shallow_fraction = 100 * np.mean(actual < 2.0)
    text = (
        f"Failure group: {group}\n\n"
        f"Support: {support_min:.3f}–{support_max:.3f} m\n"
        f"DA3 median scale: {scale:.6g}\n"
        f"Outside pixels: {outside.sum():,}\n"
        f"  below: {below.sum():,}   above: {above.sum():,}\n"
        f"  outside GT <2 m: {shallow_fraction:.1f}%\n\n"
        f"Any2Full\n"
        f"  MAE / bias: {a2f_metrics['mae_m']:.3f} / {a2f_metrics['bias_m']:+.3f} m\n"
        f"  P90 |error|: {a2f_metrics['p90_abs_m']:.3f} m\n"
        f"  >0.5 / >1 m: {a2f_metrics['bad_050_pct']:.1f}% / {a2f_metrics['bad_100_pct']:.1f}%\n"
        f"  delta1: {a2f_metrics['delta1_pct']:.1f}%\n\n"
        f"DA3 + median\n"
        f"  MAE / bias: {da3_metrics['mae_m']:.3f} / {da3_metrics['bias_m']:+.3f} m\n"
        f"  P90 |error|: {da3_metrics['p90_abs_m']:.3f} m\n"
        f"  >0.5 / >1 m: {da3_metrics['bad_050_pct']:.1f}% / {da3_metrics['bad_100_pct']:.1f}%\n"
        f"  delta1: {da3_metrics['delta1_pct']:.1f}%"
    )
    axes[2, 3].axis("off")
    axes[2, 3].text(0.02, 0.98, text, va="top", family="monospace", fontsize=10.5)

    for axis in axes[:2].ravel():
        axis.axis("off")
    fig.colorbar(anchors, ax=axes[0, 0], fraction=0.046, label="Depth (m)")
    fig.colorbar(depth_image, ax=axes[0, 1:4], fraction=0.018, label="Depth (m)")
    fig.colorbar(error_image, ax=axes[1, 1:3], fraction=0.025, label="|error| (m)")
    fig.colorbar(advantage_image, ax=axes[1, 3], fraction=0.046, label="metres")
    fig.suptitle(
        f"Outside-support catastrophic-failure diagnosis: {scene} — {group}",
        fontsize=17,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.93, wspace=0.18, hspace=0.20)
    path = output_dir / "scene_panels" / f"{group}__{scene}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    row = {
        "scene": scene,
        "failure_group": group,
        "anchor_count": int(anchor.sum()),
        "support_min_m": support_min,
        "support_max_m": support_max,
        "support_span_m": support_max - support_min,
        "outside_pixels": int(outside.sum()),
        "below_pixels": int(below.sum()),
        "above_pixels": int(above.sum()),
        "outside_shallow_lt2_pct": shallow_fraction,
        "da3_median_scale": scale,
    }
    for prefix, metrics in (("a2f", a2f_metrics), ("da3_median", da3_metrics)):
        for name, value in metrics.items():
            row[f"{prefix}_{name}"] = value
    row["absrel_advantage_da3_pp"] = a2f_metrics["absrel_pct"] - da3_metrics["absrel_pct"]
    row["rmse_advantage_da3_m"] = a2f_metrics["rmse_m"] - da3_metrics["rmse_m"]
    row["panel"] = str(path)
    return row


def summary_figure(rows, output_dir, threshold):
    ordered = sorted(rows, key=lambda row: max(row["a2f_absrel_pct"], row["da3_median_absrel_pct"]))
    scenes = [row["scene"] for row in ordered]
    y = np.arange(len(scenes))
    height = max(7, 0.45 * len(scenes) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, height))
    axes[0].barh(y - 0.19, [row["a2f_absrel_pct"] for row in ordered], 0.38, label="Any2Full")
    axes[0].barh(y + 0.19, [row["da3_median_absrel_pct"] for row in ordered], 0.38, label="DA3 + median")
    axes[0].axvline(threshold, color="red", linestyle="--", label=f"catastrophic = {threshold:g}%")
    axes[0].set_yticks(y, scenes)
    axes[0].set_xlabel("Outside-support AbsRel (%)")
    axes[0].set_title("Relative error")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y - 0.19, [row["a2f_rmse_m"] for row in ordered], 0.38, label="Any2Full")
    axes[1].barh(y + 0.19, [row["da3_median_rmse_m"] for row in ordered], 0.38, label="DA3 + median")
    axes[1].set_yticks(y, [row["failure_group"] for row in ordered])
    axes[1].set_xlabel("Outside-support RMSE (m)")
    axes[1].set_title("Absolute metric error")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Selected outside-support failures: relative and metric severity", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "00_catastrophic_failure_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = directories(args.data_root.resolve())
    paired = paired_rows(args.metrics_csv.resolve())
    selected = select_scenes(
        paired,
        args.threshold_absrel,
        args.top_k,
        args.scene,
    )
    print(f"Selected {len(selected)} scene(s) for spatial diagnosis")
    rows = []
    for index, scene in enumerate(selected, 1):
        if scene not in paired:
            print(f"[{index:2d}/{len(selected)}] SKIP {scene}: absent from paired CSV")
            continue
        print(f"[{index:2d}/{len(selected)}] {scene}")
        rows.append(
            make_panel(
                scene,
                dirs,
                args.out_dir,
                paired[scene],
                args.threshold_absrel,
            )
        )
    write_csv(args.out_dir / "catastrophic_scene_diagnostics.csv", rows)
    summary_figure(rows, args.out_dir, args.threshold_absrel)
    print(f"\nWrote diagnostics to: {args.out_dir}")
    print("Open 00_catastrophic_failure_summary.png first, then scene_panels/*.png")


if __name__ == "__main__":
    main()
