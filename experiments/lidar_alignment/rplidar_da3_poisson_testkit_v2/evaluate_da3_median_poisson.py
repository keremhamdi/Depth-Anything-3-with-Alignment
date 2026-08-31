#!/usr/bin/env python3
"""Evaluate DA3+median and DA3+median+Poisson at held-out RPLidar points.

This is four-fold blocked sparse scan-line evaluation, not dense-GT evaluation.
DA3 relative predictions must already exist as .npy files.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=5000)
    return parser.parse_args()


def load_poisson(da3_root: Path):
    source = da3_root / "experiments/lidar_alignment/ibims/compare_median_poisson_oasis_100.py"
    if not source.is_file():
        raise FileNotFoundError(f"Validated existing_poisson source not found: {source}")
    spec = importlib.util.spec_from_file_location("validated_existing_poisson", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "existing_poisson", None)
    if not callable(function):
        raise AttributeError(f"{source} does not define existing_poisson")
    print(f"Using validated existing_poisson{inspect.signature(function)} from {source}")
    return function


def call_poisson(function, base, sparse, anchors, rtol, maxiter):
    aliases = {
        "base": base, "base_depth": base, "depth": base, "initial": base,
        "initial_depth": base, "aligned": base, "aligned_depth": base,
        "prediction": base, "pred": base, "sparse": sparse,
        "sparse_depth": sparse, "lidar": sparse, "lidar_depth": sparse,
        "metric_depth": sparse, "anchors": anchors, "anchor_mask": anchors,
        "sparse_mask": anchors, "valid_mask": anchors, "rtol": rtol,
        "tol": rtol, "maxiter": maxiter, "max_iter": maxiter,
    }
    signature = inspect.signature(function)
    kwargs = {}
    unknown = []
    for name, parameter in signature.parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            unknown.append(name)
    result = (function(base, sparse, anchors, rtol, maxiter)
              if unknown else function(**kwargs))
    prediction = result[0] if isinstance(result, tuple) else result
    prediction = np.squeeze(np.asarray(prediction, dtype=np.float32))
    if prediction.shape != base.shape:
        raise ValueError(f"Poisson returned {prediction.shape}; expected {base.shape}")
    return prediction


def npy(directory: Path, stem: str) -> Path:
    names = [f"{stem}.npy", f"{stem}_da3small.npy", f"{stem}_da3.npy"]
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    matches = sorted(directory.glob(f"{stem}*.npy"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve {stem} in {directory}: {matches}")


def load_map(path: Path, shape: tuple[int, int]) -> np.ndarray:
    array = np.squeeze(np.load(path)).astype(np.float32)
    if array.shape == shape:
        return array
    source = Image.fromarray(array, mode="F")
    resized = source.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def median_align(relative: np.ndarray, sparse: np.ndarray) -> tuple[np.ndarray, float]:
    anchors = np.isfinite(sparse) & (sparse > 0) & np.isfinite(relative) & (relative > 0)
    ratios = sparse[anchors] / relative[anchors]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 8:
        raise RuntimeError(f"Only {len(ratios)} valid alignment anchors")
    scale = float(np.median(ratios))
    return (relative * scale).astype(np.float32), scale


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    pred = np.array([float(row["prediction_m"]) for row in rows])
    target = np.array([float(row["gt_z_m"]) for row in rows])
    error = pred - target
    absolute = np.abs(error)
    ratio = np.maximum(pred / target, target / pred)
    return {
        "points": len(rows),
        "absrel_pct": float(100 * np.mean(absolute / target)),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mae_m": float(np.mean(absolute)),
        "bias_m": float(np.mean(error)),
        "p90_abs_m": float(np.percentile(absolute, 90)),
        "delta1_pct": float(100 * np.mean(ratio < 1.25)),
        "bad_010_pct": float(100 * np.mean(absolute > 0.10)),
        "bad_025_pct": float(100 * np.mean(absolute > 0.25)),
    }


def evaluate_points(prediction: np.ndarray, heldout: list[dict[str, str]],
                    method: str, fold: int) -> list[dict]:
    rows = []
    for point in heldout:
        u, v = int(point["u"]), int(point["v"])
        value = float(prediction[v, u])
        target = float(point["z_m"])
        if not math.isfinite(value) or value <= 0:
            continue
        rows.append({
            "method": method,
            "stem": point["stem"],
            "fold": fold,
            "sector": point["sector"],
            "u": u,
            "v": v,
            "gt_z_m": target,
            "prediction_m": value,
            "abs_error_m": abs(value - target),
            "absrel_pct": 100 * abs(value - target) / target,
        })
    return rows


def render_panel(stem: str, rgb_path: Path, sparse: np.ndarray,
                 median_prediction: np.ndarray, poisson_prediction: np.ndarray,
                 point_rows: list[dict], output: Path) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    positive = poisson_prediction[np.isfinite(poisson_prediction) & (poisson_prediction > 0)]
    vmax = float(np.clip(np.percentile(positive, 99), 3.0, 12.0))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    y, x = np.where(sparse > 0)
    axes[0, 0].scatter(x, y, s=5, c="cyan")
    axes[0, 0].set_title("RGB + all real RPLidar anchors")
    im = axes[0, 1].imshow(median_prediction, cmap="turbo", vmin=0, vmax=vmax)
    axes[0, 1].set_title("DA3 + median (full-input qualitative)")
    axes[1, 0].imshow(poisson_prediction, cmap="turbo", vmin=0, vmax=vmax)
    axes[1, 0].set_title("DA3 + median + existing Poisson")
    axes[1, 1].imshow(rgb)
    errors = [row for row in point_rows if row["method"] == "da3_median_poisson"]
    for row in errors:
        rel = float(row["absrel_pct"])
        color = "lime" if rel < 10 else "gold" if rel < 25 else "red"
        axes[1, 1].scatter([int(row["u"])], [int(row["v"])], s=34,
                           facecolors="none", edgecolors=color, linewidths=1.5)
    axes[1, 1].set_title("Four-fold held-out LiDAR error: green <10%, yellow <25%, red >=25%")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.colorbar(im, ax=[axes[0, 1], axes[1, 0]], shrink=0.8, label="Metric depth (m)")
    fig.suptitle(stem + "\nNumerical dots were never used by that fold's alignment/Poisson solve")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)


def main() -> None:
    args = arguments()
    da3_root = args.da3_root.expanduser().resolve()
    prepared = args.prepared_root.expanduser().resolve()
    da3_dir = args.da3_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    poisson = load_poisson(da3_root)

    manifest = read_rows(prepared / "manifest.csv")
    stems = [row["stem"] for row in manifest]
    point_metrics: list[dict] = []
    per_scene_scales: list[dict] = []
    for fold in range(args.folds):
        heldout_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_rows(prepared / f"fold_{fold}" / "heldout_points.csv"):
            heldout_by_stem[row["stem"]].append(row)
        for index, row in enumerate(manifest, 1):
            stem = row["stem"]
            shape = (int(row["height"]), int(row["width"]))
            relative = load_map(npy(da3_dir, stem), shape)
            sparse = load_map(prepared / f"fold_{fold}" / "depth_fit_points" / f"{stem}.npy", shape)
            median_prediction, scale = median_align(relative, sparse)
            anchors = sparse > 0
            poisson_prediction = call_poisson(
                poisson, median_prediction, sparse, anchors, args.rtol, args.maxiter
            )
            heldout = heldout_by_stem[stem]
            point_metrics.extend(evaluate_points(median_prediction, heldout, "da3_median", fold))
            point_metrics.extend(evaluate_points(
                poisson_prediction, heldout, "da3_median_poisson", fold
            ))
            per_scene_scales.append({"stem": stem, "fold": fold,
                                     "fit_anchor_count": int(anchors.sum()),
                                     "median_scale": scale})
            print(f"fold {fold} [{index:02d}/{len(stems)}] {stem} "
                  f"fit={anchors.sum()} heldout={len(heldout)}", flush=True)

    scene_rows: list[dict] = []
    for method in ("da3_median", "da3_median_poisson"):
        for stem in stems:
            rows = [row for row in point_metrics
                    if row["method"] == method and row["stem"] == stem]
            scene_rows.append({"method": method, "stem": stem, **aggregate(rows)})
    summary_rows = []
    for method in ("da3_median", "da3_median_poisson"):
        rows = [row for row in point_metrics if row["method"] == method]
        summary_rows.append({"method": method, "evaluation": "4-fold blocked held-out LiDAR",
                             **aggregate(rows)})

    full_dir = output / "full_predictions_m"
    panel_dir = output / "panels"
    full_dir.mkdir(exist_ok=True); panel_dir.mkdir(exist_ok=True)
    for index, row in enumerate(manifest, 1):
        stem = row["stem"]
        shape = (int(row["height"]), int(row["width"]))
        relative = load_map(npy(da3_dir, stem), shape)
        sparse = load_map(prepared / "depth_full_points" / f"{stem}.npy", shape)
        median_prediction, _ = median_align(relative, sparse)
        poisson_prediction = call_poisson(
            poisson, median_prediction, sparse, sparse > 0, args.rtol, args.maxiter
        )
        np.save(full_dir / f"{stem}.npy", poisson_prediction.astype(np.float32))
        render_panel(
            stem, prepared / "rgb" / f"{stem}.png", sparse,
            median_prediction, poisson_prediction,
            [metric for metric in point_metrics if metric["stem"] == stem],
            panel_dir / f"{stem}__real_lidar_heldout_panel.png",
        )
        print(f"full [{index:02d}/{len(stems)}] {stem}", flush=True)

    write_csv(output / "per_point_metrics.csv", point_metrics)
    write_csv(output / "per_scene_metrics.csv", scene_rows)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "alignment_scales.csv", per_scene_scales)
    print("\n===== SPARSE HELD-OUT RESULT =====")
    for row in summary_rows:
        print(f"{row['method']:22s} points={row['points']:4d} "
              f"AbsRel={row['absrel_pct']:7.3f}% RMSE={row['rmse_m']:.4f} m "
              f"MAE={row['mae_m']:.4f} m delta1={row['delta1_pct']:.2f}%")
    print(f"Output: {output}")
    print("These are sparse held-out scan-line metrics, not full-image GT metrics.")


if __name__ == "__main__":
    main()
