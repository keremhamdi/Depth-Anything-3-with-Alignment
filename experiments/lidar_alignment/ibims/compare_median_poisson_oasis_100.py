"""Compare DA3+median with two training-free Poisson refinements on iBims V2.1.

Methods:
  - Any2Full (reference)
  - DA3-Small + median scale
  - DA3-Small + median scale + the repository's existing soft Poisson correction
  - DA3-Small + median scale + OASIS-style hard-anchor Poisson pseudo-depth

The primary region is outside_support. The script intentionally produces only
one summary dashboard and four simple qualitative examples.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator, cg


METHODS = {
    "any2full": "Any2Full",
    "da3_median": "DA3 + median",
    "da3_median_poisson": "DA3 + median + existing Poisson",
    "da3_median_oasis": "DA3 + median + OASIS Poisson prior",
}


def parse_args():
    env_root = os.environ.get("DA3_LIDAR_DATA_ROOT")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(env_root) if env_root else None,
        required=env_root is None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--skip-existing-poisson", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


def paths(data_root):
    base = data_root / "experiments/ibims_replication"
    result = {
        "gt": data_root / "datasets/ibims1/ibims1_core_mat",
        "rgb": data_root / "datasets/ibims1/ibims1_core_raw/rgb",
        "sparse": base / "v2_1_sensor",
        "a2f": base / "predictions_v2_1_sensor",
        "da3": base / "da3_bridge_all",
    }
    missing = [str(value) for value in result.values() if not value.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required directories:\n  " + "\n  ".join(missing))
    return result


def scene_names(directories):
    sets = [
        {p.stem for p in directories["gt"].glob("*.mat")},
        {p.stem for p in directories["sparse"].glob("*.npy")},
        {
            p.stem
            for p in directories["a2f"].glob("*.npy")
            if not p.stem.endswith("_rel")
        },
    ]
    da3 = set()
    for path in directories["da3"].glob("*.npy"):
        stem = path.stem
        da3.add(stem[:-9] if stem.endswith("_da3small") else stem)
    sets.append(da3)
    matched = sorted(set.intersection(*sets))
    print("\n========== INPUT AUDIT ==========")
    print(f"Matched V2.1 scenes: {len(matched)}")
    return matched


def load_gt(path):
    record = loadmat(path)["data"][0, 0]
    depth = np.squeeze(record["depth"]).astype(np.float64)
    valid = np.squeeze(record["mask_invalid"]).astype(bool)
    if record.dtype.names and "mask_transp" in record.dtype.names:
        valid &= np.squeeze(record["mask_transp"]).astype(bool)
    valid &= np.isfinite(depth) & (depth > 0)
    return depth, valid


def load_array(path, shape):
    value = np.squeeze(np.load(path)).astype(np.float64)
    if value.shape != shape:
        raise ValueError(f"{path}: shape {value.shape}; expected {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{path}: contains non-finite values")
    return value


def resolve_da3(directory, scene):
    choices = [directory / f"{scene}_da3small.npy", directory / f"{scene}.npy"]
    found = [path for path in choices if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one DA3 file for {scene}: {choices}")
    return found[0]


def resolve_rgb(directory, scene):
    for suffix in (".png", ".jpg", ".jpeg"):
        path = directory / f"{scene}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"RGB not found for {scene}")


def median_align(relative, sparse, anchors):
    scale = float(np.median(sparse[anchors] / relative[anchors]))
    return scale * relative, scale


def laplacian4(image):
    result = 4.0 * image.copy()
    result[1:, :] -= image[:-1, :]
    result[:-1, :] -= image[1:, :]
    result[:, 1:] -= image[:, :-1]
    result[:, :-1] -= image[:, 1:]
    return result


def oasis_hard_poisson(prior, sparse, anchors, rtol=1e-6, maxiter=1000):
    """OASIS pseudo-depth equations (1)-(6), without its learned network."""
    height, width = prior.shape
    boundary = np.zeros_like(anchors, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    known = anchors | boundary
    unknown = ~known

    known_values = np.zeros_like(prior)
    known_values[boundary] = prior[boundary]
    known_values[anchors] = sparse[anchors]

    if not np.any(unknown):
        return known_values, {
            "iterations_status": 0,
            "unknown_count": 0,
            "anchor_rmse_m": 0.0,
        }

    def matvec(vector):
        image = np.zeros_like(prior)
        image[unknown] = vector
        return laplacian4(image)[unknown]

    operator = LinearOperator(
        (int(unknown.sum()), int(unknown.sum())),
        matvec=matvec,
        dtype=np.float64,
    )

    known_neighbor_sum = np.zeros_like(prior)
    known_neighbor_sum[1:, :] += known_values[:-1, :]
    known_neighbor_sum[:-1, :] += known_values[1:, :]
    known_neighbor_sum[:, 1:] += known_values[:, :-1]
    known_neighbor_sum[:, :-1] += known_values[:, 1:]
    rhs = (laplacian4(prior) + known_neighbor_sum)[unknown]
    x0 = prior[unknown]

    try:
        solution, status = cg(
            operator,
            rhs,
            x0=x0,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
        )
    except TypeError:
        solution, status = cg(operator, rhs, x0=x0, tol=rtol, maxiter=maxiter)

    output = known_values.copy()
    output[unknown] = solution
    anchor_rmse = float(
        np.sqrt(np.mean((output[anchors] - sparse[anchors]) ** 2))
    )
    return output, {
        "iterations_status": int(status),
        "unknown_count": int(unknown.sum()),
        "anchor_rmse_m": anchor_rmse,
    }


def existing_poisson(prior, sparse, anchors, rtol, maxiter):
    try:
        from depth_anything_3.alignment.poisson_alignment import poisson_align
    except ImportError as error:
        raise ImportError(
            "Could not import the repository Poisson solver. Run from the DA3 "
            "repository with PYTHONPATH=src."
        ) from error
    result = poisson_align(
        prior,
        sparse,
        anchors,
        rtol=rtol,
        maxiter=maxiter,
    )
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, {}


def metrics(prediction, gt, mask):
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
        "delta1_pct": float(100 * np.mean(ratio < 1.25)),
        "bad_050_pct": float(100 * np.mean(absolute > 0.50)),
        "bad_100_pct": float(100 * np.mean(absolute > 1.00)),
    }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(differences, seed=20260827, repeats=10000):
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for start in range(0, repeats, 500):
        count = min(500, repeats - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        means[start : start + count] = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize(rows):
    outside = [row for row in rows if row["region"] == "outside_support"]
    methods = sorted({row["method"] for row in outside})
    summary = []
    for method in methods:
        group = [row for row in outside if row["method"] == method]
        counts = np.array([row["n"] for row in group], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "method_label": METHODS[method],
                "scene_count": len(group),
                "macro_mean_absrel_pct": float(np.mean([r["absrel_pct"] for r in group])),
                "macro_median_absrel_pct": float(np.median([r["absrel_pct"] for r in group])),
                "macro_mean_rmse_m": float(np.mean([r["rmse_m"] for r in group])),
                "macro_mean_mae_m": float(np.mean([r["mae_m"] for r in group])),
                "p90_scene_absrel_pct": float(np.percentile([r["absrel_pct"] for r in group], 90)),
                "failure_ge20_pct": float(100 * np.mean([r["absrel_pct"] >= 20 for r in group])),
                "failure_ge40_pct": float(100 * np.mean([r["absrel_pct"] >= 40 for r in group])),
                "pooled_absrel_pct": float(np.average([r["absrel_pct"] for r in group], weights=counts)),
            }
        )
    return summary


def paired_report(rows):
    outside = [row for row in rows if row["region"] == "outside_support"]
    lookup = {(row["scene"], row["method"]): row for row in outside}
    scenes = sorted({row["scene"] for row in outside})
    baseline = "da3_median"
    report = []
    for candidate in ("da3_median_poisson", "da3_median_oasis"):
        if not all((scene, candidate) in lookup for scene in scenes):
            continue
        differences = np.array(
            [
                lookup[(scene, baseline)]["absrel_pct"]
                - lookup[(scene, candidate)]["absrel_pct"]
                for scene in scenes
            ]
        )
        rmse_differences = np.array(
            [
                lookup[(scene, baseline)]["rmse_m"]
                - lookup[(scene, candidate)]["rmse_m"]
                for scene in scenes
            ]
        )
        low, high = bootstrap_mean_ci(differences)
        report.append(
            {
                "candidate": candidate,
                "candidate_label": METHODS[candidate],
                "mean_absrel_improvement_pp": float(differences.mean()),
                "median_absrel_improvement_pp": float(np.median(differences)),
                "absrel_win_rate_pct": float(100 * np.mean(differences > 0)),
                "mean_rmse_improvement_m": float(rmse_differences.mean()),
                "rmse_win_rate_pct": float(100 * np.mean(rmse_differences > 0)),
                "bootstrap_ci_low_pp": low,
                "bootstrap_ci_high_pp": high,
            }
        )
    return report


def summary_figure(summary, paired, output_dir):
    order = [
        method
        for method in (
            "any2full",
            "da3_median",
            "da3_median_poisson",
            "da3_median_oasis",
        )
        if any(row["method"] == method for row in summary)
    ]
    lookup = {row["method"]: row for row in summary}
    labels = [METHODS[method].replace("DA3 + median + ", "+ ") for method in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plots = [
        ("macro_mean_absrel_pct", "Mean outside-support AbsRel", "%"),
        ("macro_median_absrel_pct", "Median outside-support AbsRel", "%"),
        ("macro_mean_rmse_m", "Mean outside-support RMSE", "m"),
        ("failure_ge40_pct", "Scenes with at least 40% AbsRel", "% of scenes"),
    ]
    for axis, (key, title, unit) in zip(axes.ravel(), plots):
        values = [lookup[method][key] for method in order]
        bars = axis.bar(x, values)
        best = int(np.argmin(values))
        for index, bar in enumerate(bars):
            bar.set_color("#2ca25f" if index == best else "#9ecae1")
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{values[index]:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel(unit)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "What Poisson and OASIS add to DA3 + median - outside support",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Lower is better. Green marks the best method in each panel.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output_dir / "01_outside_support_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_scene_predictions(scene, directories, rtol, maxiter, include_existing):
    gt, valid = load_gt(directories["gt"] / f"{scene}.mat")
    sparse = load_array(directories["sparse"] / f"{scene}.npy", gt.shape)
    a2f = load_array(directories["a2f"] / f"{scene}.npy", gt.shape)
    relative = load_array(resolve_da3(directories["da3"], scene), gt.shape)
    anchors = (
        np.isfinite(sparse)
        & (sparse > 0)
        & np.isfinite(relative)
        & (relative > 0)
    )
    median, scale = median_align(relative, sparse, anchors)
    predictions = {
        "any2full": a2f,
        "da3_median": median,
    }
    diagnostics = {}
    if include_existing:
        predictions["da3_median_poisson"], diagnostics["existing"] = existing_poisson(
            median, sparse, anchors, rtol, maxiter
        )
    predictions["da3_median_oasis"], diagnostics["oasis"] = oasis_hard_poisson(
        median, sparse, anchors, rtol, maxiter
    )
    support_min = float(sparse[anchors].min())
    support_max = float(sparse[anchors].max())
    outside = valid & ((gt < support_min) | (gt > support_max))
    return gt, valid, sparse, anchors, predictions, diagnostics, outside, scale


def simple_scene_visual(scene, tag, directories, output_dir, rtol, maxiter, include_existing):
    gt, valid, sparse, anchors, predictions, _, outside, _ = load_scene_predictions(
        scene, directories, rtol, maxiter, include_existing
    )
    rgb = Image.open(resolve_rgb(directories["rgb"], scene)).convert("RGB")
    rgb = rgb.resize((gt.shape[1], gt.shape[0]))
    methods = ["da3_median"]
    if include_existing:
        methods.append("da3_median_poisson")
    methods.append("da3_median_oasis")
    values = {method: metrics(predictions[method], gt, outside) for method in methods}
    winner = min(methods, key=lambda method: values[method]["absrel_pct"])

    columns = 2 + len(methods)
    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.5))
    axes[0].imshow(rgb)
    yy, xx = np.nonzero(anchors)
    axes[0].scatter(xx, yy, c=sparse[anchors], s=5, cmap="turbo")
    axes[0].set_title(f"RGB + V2.1 anchors\nn={anchors.sum()}")
    gt_show = np.where(valid, gt, np.nan)
    depth_min, depth_max = np.nanpercentile(gt_show, [1, 99])
    axes[1].imshow(gt_show, cmap="turbo", vmin=depth_min, vmax=depth_max)
    axes[1].set_title("Dense ground truth")

    for index, method in enumerate(methods, start=2):
        axis = axes[index]
        axis.imshow(
            np.where(valid, predictions[method], np.nan),
            cmap="turbo",
            vmin=depth_min,
            vmax=depth_max,
        )
        score = values[method]
        status = "WINNER" if method == winner else ""
        axis.set_title(
            f"{METHODS[method]}\n"
            f"AbsRel {score['absrel_pct']:.1f}% | RMSE {score['rmse_m']:.2f} m\n{status}"
        )
        color = "#1a9850" if method == winner else "#d73027"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(5 if method == winner else 2)
            spine.set_edgecolor(color)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"{tag}: {scene} - outside-support comparison",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    folder = output_dir / "simple_scene_examples"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{tag}__{scene}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def choose_visual_scenes(rows, include_existing):
    outside = [row for row in rows if row["region"] == "outside_support"]
    lookup = {(row["scene"], row["method"]): row for row in outside}
    scenes = sorted({row["scene"] for row in outside})
    candidates = ["da3_median_oasis"]
    if include_existing:
        candidates.insert(0, "da3_median_poisson")
    gains = {
        candidate: {
            scene: lookup[(scene, "da3_median")]["absrel_pct"]
            - lookup[(scene, candidate)]["absrel_pct"]
            for scene in scenes
        }
        for candidate in candidates
    }
    selected = []
    used = set()
    for candidate in candidates:
        scene = max(gains[candidate], key=gains[candidate].get)
        if scene not in used:
            selected.append((f"best_{candidate}", scene))
            used.add(scene)
    combined_worst = min(
        scenes,
        key=lambda scene: min(gains[candidate][scene] for candidate in candidates),
    )
    if combined_worst not in used:
        selected.append(("worst_refinement", combined_worst))
        used.add(combined_worst)
    combined = np.array(
        [np.mean([gains[candidate][scene] for candidate in candidates]) for scene in scenes]
    )
    typical_order = np.argsort(np.abs(combined - np.median(combined)))
    for index in typical_order:
        scene = scenes[int(index)]
        if scene not in used:
            selected.append(("typical", scene))
            break
    return selected[:4]


def write_report(path, summary, paired):
    lines = [
        "DA3 + median Poisson/OASIS comparison",
        "Primary region: outside_support",
        "",
    ]
    for row in sorted(summary, key=lambda value: value["macro_mean_absrel_pct"]):
        lines.append(
            f"{row['method_label']}: mean/median AbsRel "
            f"{row['macro_mean_absrel_pct']:.3f}% / "
            f"{row['macro_median_absrel_pct']:.3f}%; mean RMSE "
            f"{row['macro_mean_rmse_m']:.3f} m; >=40% failures "
            f"{row['failure_ge40_pct']:.1f}%"
        )
    lines.append("")
    lines.append("Directly against DA3 + median:")
    for row in paired:
        lines.append(
            f"{row['candidate_label']}: mean AbsRel improvement "
            f"{row['mean_absrel_improvement_pp']:+.3f} pp; wins "
            f"{row['absrel_win_rate_pct']:.1f}%; paired bootstrap CI "
            f"[{row['bootstrap_ci_low_pp']:+.3f}, "
            f"{row['bootstrap_ci_high_pp']:+.3f}] pp; mean RMSE improvement "
            f"{row['mean_rmse_improvement_m']:+.3f} m"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "A refinement is useful only if it improves DA3 + median directly, not merely Any2Full.",
            "The OASIS row is its training-free hard-anchor pseudo-depth stage, not the full trained OASIS-DC network.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    directories = paths(data_root)
    scenes = scene_names(directories)
    if args.scene:
        if args.scene not in scenes:
            raise ValueError(f"Scene not matched: {args.scene}")
        scenes = [args.scene]
    if args.limit:
        scenes = scenes[: args.limit]
    include_existing = not args.skip_existing_poisson

    print(f"Evaluating {len(scenes)} scene(s)")
    metric_rows = []
    diagnostic_rows = []
    for index, scene in enumerate(scenes, 1):
        (
            gt,
            valid,
            sparse,
            anchors,
            predictions,
            diagnostics,
            outside,
            scale,
        ) = load_scene_predictions(
            scene,
            directories,
            args.rtol,
            args.maxiter,
            include_existing,
        )
        print(
            f"[{index:3d}/{len(scenes)}] {scene:25s} "
            f"anchors={anchors.sum():3d} outside={outside.sum():7d}"
        )
        for method, prediction in predictions.items():
            for region, mask in (("outside_support", outside), ("all_valid", valid)):
                row = {
                    "scene": scene,
                    "method": method,
                    "method_label": METHODS[method],
                    "region": region,
                }
                row.update(metrics(prediction, gt, mask))
                metric_rows.append(row)
        diagnostic_rows.append(
            {
                "scene": scene,
                "anchor_count": int(anchors.sum()),
                "support_min_m": float(sparse[anchors].min()),
                "support_max_m": float(sparse[anchors].max()),
                "median_scale": scale,
                "existing_poisson": json.dumps(diagnostics.get("existing", {}), default=str),
                "oasis_poisson": json.dumps(diagnostics.get("oasis", {}), default=str),
            }
        )

    write_csv(output_dir / "per_scene_metrics.csv", metric_rows)
    write_csv(output_dir / "solver_diagnostics.csv", diagnostic_rows)
    summary = summarize(metric_rows)
    paired = paired_report(metric_rows)
    write_csv(output_dir / "summary_outside_support.csv", summary)
    write_csv(output_dir / "paired_vs_da3_median.csv", paired)
    write_report(output_dir / "comparison_report.txt", summary, paired)
    summary_figure(summary, paired, output_dir)

    if not args.no_visuals:
        for tag, scene in choose_visual_scenes(metric_rows, include_existing):
            print(f"Writing simple visual: {tag} / {scene}")
            simple_scene_visual(
                scene,
                tag,
                directories,
                output_dir,
                args.rtol,
                args.maxiter,
                include_existing,
            )

    print(f"\nWrote comparison to: {output_dir}")
    print("Open 01_outside_support_summary.png and comparison_report.txt first.")


if __name__ == "__main__":
    main()
