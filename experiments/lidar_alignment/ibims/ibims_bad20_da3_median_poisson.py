#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ibims_visual_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_metrics(root: Path) -> tuple[Path, list[dict[str, str]]]:
    candidates = []
    for path in root.rglob("per_scene_metrics.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        wanted = [r for r in rows if r.get("method") == "da3_median_poisson"]
        scenes = {r.get("scene") for r in wanted if r.get("scene")}
        regions = {r.get("region") for r in wanted}
        if len(scenes) >= 20 and {"all_valid", "outside_support"} <= regions:
            candidates.append((len(scenes), path.stat().st_mtime, path, rows))
    if not candidates:
        raise FileNotFoundError(
            "No per_scene_metrics.csv containing da3_median_poisson for "
            "all_valid and outside_support was found under " + str(root)
        )
    _, _, path, rows = max(candidates, key=lambda item: (item[0], item[1]))
    return path, rows


def percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda scene: values[scene])
    denom = max(1, len(ordered) - 1)
    return {scene: rank / denom for rank, scene in enumerate(ordered)}


def choose(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    by_scene: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("method") == "da3_median_poisson":
            by_scene.setdefault(row["scene"], {})[row["region"]] = row
    usable = {
        scene: regions for scene, regions in by_scene.items()
        if "all_valid" in regions and "outside_support" in regions
    }
    fields = [
        ("all_valid", "absrel_pct"),
        ("all_valid", "rmse_m"),
        ("outside_support", "absrel_pct"),
        ("outside_support", "rmse_m"),
    ]
    rank_maps = []
    for region, field in fields:
        values = {scene: float(rs[region][field]) for scene, rs in usable.items()}
        rank_maps.append(percentile_ranks(values))
    selected = []
    for scene, rs in usable.items():
        selected.append({
            "scene": scene,
            # Higher quality_score = worse performance (higher percentile rank on error metrics).
            "quality_score": float(np.mean([m[scene] for m in rank_maps])),
            "all_valid_absrel_pct": float(rs["all_valid"]["absrel_pct"]),
            "all_valid_rmse_m": float(rs["all_valid"]["rmse_m"]),
            "outside_support_absrel_pct": float(rs["outside_support"]["absrel_pct"]),
            "outside_support_rmse_m": float(rs["outside_support"]["rmse_m"]),
        })
    # Sort DESCENDING: worst scenes first.
    selected.sort(
        key=lambda row: (float(row["quality_score"]), float(row["all_valid_absrel_pct"])),
        reverse=True,
    )
    return selected[:20]


def save_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_panel(rank, scene, rgb, gt, pred, valid, sparse, probes, selected_row, output):
    gt_show = np.where(valid, gt, np.nan)
    pred_show = np.where(valid & np.isfinite(pred) & (pred > 0), pred, np.nan)
    err_show = np.abs(pred_show - gt_show)
    finite_gt = gt[valid & np.isfinite(gt)]
    depth_max = float(np.clip(np.percentile(finite_gt, 99.5), 6.0, 40.0))
    # Wider error range for bad scenes so large errors remain visible.
    finite_err = err_show[np.isfinite(err_show)]
    error_max = float(np.clip(np.percentile(finite_err, 99.0), 1.0, 15.0)) \
        if finite_err.size else 5.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    ay, ax = np.where(sparse > 0)
    axes[0, 0].scatter(ax, ay, s=4, c="cyan")
    axes[0, 0].set_title("RGB + simulated one-line sparse depth")
    im_depth = axes[0, 1].imshow(gt_show, cmap="turbo", vmin=0, vmax=depth_max)
    axes[0, 1].set_title("iBims GT metric depth")
    axes[1, 0].imshow(pred_show, cmap="turbo", vmin=0, vmax=depth_max)
    axes[1, 0].set_title("DA3 + median + existing Poisson")
    im_error = axes[1, 1].imshow(err_show, cmap="inferno", vmin=0, vmax=error_max)
    axes[1, 1].set_title("Absolute metric-depth error")

    colors = {"N": "lime", "F": "dodgerblue", "W": "red"}
    for probe in probes:
        x, y, code = probe["x"], probe["y"], probe["code"]
        color = colors[code]
        for axis in axes.flat:
            axis.scatter([x], [y], s=70, facecolors="none", edgecolors=color, lw=2)
            axis.text(x, y, code, color=color, weight="bold", ha="center", va="center", fontsize=8)
        box = dict(facecolor="black", alpha=0.72, edgecolor="none", pad=2)
        axes[0, 1].text(x + 7, y - 7, f"{code} GT {probe['gt_m']:.2f} m",
                        color=color, fontsize=8, weight="bold", bbox=box)
        axes[1, 0].text(x + 7, y - 7,
                        f"{code} Pred {probe['pred_m']:.2f} m\n"
                        f"|e| {probe['abs_error_m']:.2f} m ({probe['absrel_pct']:.1f}%)",
                        color=color, fontsize=8, weight="bold", bbox=box)
        axes[1, 1].text(x + 7, y - 7,
                        f"{code} d(anchor) {probe['distance_to_anchor_px']:.1f}px",
                        color=color, fontsize=8, weight="bold", bbox=box)

    for axis in axes.flat:
        axis.set_axis_off()
    fig.colorbar(im_depth, ax=[axes[0, 1], axes[1, 0]], shrink=0.78, label="Depth (m)")
    fig.colorbar(im_error, ax=axes[1, 1], shrink=0.78, label="Absolute error (m)")
    fig.suptitle(
        f"BAD #{rank:02d} — {scene}\n"
        f"all-valid: AbsRel {selected_row['all_valid_absrel_pct']:.3f}% | "
        f"RMSE {selected_row['all_valid_rmse_m']:.3f} m    "
        f"outside-support: AbsRel {selected_row['outside_support_absrel_pct']:.3f}% | "
        f"RMSE {selected_row['outside_support_rmse_m']:.3f} m"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def contact_sheet(paths: list[Path], output: Path):
    thumbs = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((480, 320))
            canvas = Image.new("RGB", (490, 350), "white")
            canvas.paste(thumb, ((490 - thumb.width) // 2, 5))
            ImageDraw.Draw(canvas).text((8, 328), path.stem[:70], fill="black")
            thumbs.append(canvas)
    sheet = Image.new("RGB", (490 * 4, 350 * 5), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 4) * 490, (index // 4) * 350))
    sheet.save(output)


def main():
    da3_root = Path(os.environ["DA3_ROOT"]).expanduser().resolve()
    a2f_root = Path(os.environ["A2F_ROOT"]).expanduser().resolve()
    output_root = da3_root / "experiments/lidar_alignment/outputs"
    metrics_path, rows = find_metrics(output_root)
    selected = choose(rows)
    out_dir = metrics_path.parent / "examples_bad20_da3_median_poisson"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(out_dir / "selected_bad20.csv", selected)

    helper_path = da3_root / "experiments/lidar_alignment/ibims/ibims_4line_da3_median_poisson_eval.py"
    helper = load_module(helper_path)
    search_roots = [a2f_root]
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        search_roots.append(Path(data_root).expanduser().resolve())
    gt_dir = helper.resolve_gt(None, search_roots)
    sparse_dir = helper.resolve_npy_dir(None, search_roots, ["v2_1_sensor", "v2_sensor"], "one-line maps")
    da3_dir = helper.resolve_npy_dir(None, search_roots, ["da3_bridge_all"], "DA3 maps")
    poisson = helper.load_poisson(da3_root)

    panel_paths = []
    for rank, selected_row in enumerate(selected, 1):
        scene = str(selected_row["scene"])
        gt, valid, rgb = helper.load_ibims(gt_dir / f"{scene}.mat")
        sparse = helper.load_npy(helper.npy_path(sparse_dir, scene), gt.shape)
        anchors = valid & np.isfinite(sparse) & (sparse > 0)
        relative = helper.load_npy(helper.npy_path(da3_dir, scene, True), gt.shape)
        base, _ = helper.median_align(relative, sparse, anchors)
        pred, _ = helper.call_poisson(poisson, base, sparse, anchors, 1e-6, 5000)
        probe_rows = helper.probes(gt, pred, valid, anchors)
        panel_path = out_dir / f"{rank:02d}__{scene}__metric_panel.png"
        make_panel(rank, scene, rgb, gt, pred, valid, sparse, probe_rows, selected_row, panel_path)
        panel_paths.append(panel_path)
        print(f"[{rank:02d}/20] {scene} -> {panel_path.name}", flush=True)

    contact_sheet(panel_paths, out_dir / "00_bad20_contact_sheet.png")
    print(f"\nMetrics source: {metrics_path}")
    print(f"Open: {out_dir / '00_bad20_contact_sheet.png'}")
    print(f"Individual full-resolution panels: {out_dir}")


if __name__ == "__main__":
    main()
