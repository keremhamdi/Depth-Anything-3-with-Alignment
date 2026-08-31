#!/usr/bin/env python3
"""Annotate exact-pixel metric depth from DA3+median and DA3+median+Poisson.

This is a qualitative measurement tool for the prepared real RPLidar dataset.
It does not use a 7x7 patch and it does not convert predictions to radial laser
range.  Every displayed value is the camera-axis Z depth at the selected pixel.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--scene",
        help="Frame stem or its 1-based index in prepared/manifest.csv.",
    )
    parser.add_argument(
        "--points",
        nargs="*",
        metavar="X,Y[,LABEL]",
        help="Optional non-interactive exact pixels; otherwise click the image.",
    )
    parser.add_argument(
        "--display-max-m",
        type=float,
        default=None,
        help="Shared maximum of the two depth-map color scales.",
    )
    return parser.parse_args()


def read_manifest(prepared_root: Path) -> list[str]:
    manifest_path = prepared_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    candidate_keys = ("stem", "scene", "frame", "frame_id", "id")
    key = next((name for name in candidate_keys if name in rows[0]), None)
    if key is None:
        raise RuntimeError(
            f"Cannot find a scene/stem column in {manifest_path}; "
            f"columns are {list(rows[0])}"
        )
    return [row[key] for row in rows]


def choose_scene(stems: list[str], requested: str | None) -> str:
    if requested is None:
        print("\nAvailable frames:")
        for index, stem in enumerate(stems, start=1):
            print(f"  {index:2d}: {stem}")
        requested = input("Choose a frame number or exact stem: ").strip()

    if requested in stems:
        return requested
    try:
        index = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown scene: {requested!r}") from exc
    if not 1 <= index <= len(stems):
        raise ValueError(f"Scene index must be 1..{len(stems)}, got {index}")
    return stems[index - 1]


def resolve_existing(directory: Path, names: list[str], kind: str) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    tried = ", ".join(str(directory / name) for name in names)
    raise FileNotFoundError(f"Cannot find {kind}. Tried: {tried}")


def load_rgb(prepared_root: Path, stem: str) -> tuple[Path, np.ndarray]:
    path = resolve_existing(
        prepared_root / "rgb",
        [f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg"],
        "RGB image",
    )
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return path, rgb


def load_relative(da3_dir: Path, stem: str, shape: tuple[int, int]) -> np.ndarray:
    path = resolve_existing(
        da3_dir,
        [f"{stem}.npy", f"{stem}_da3small.npy", f"{stem}_da3.npy"],
        "DA3 prediction",
    )
    relative = np.load(path).astype(np.float32)
    if relative.shape != shape:
        image = Image.fromarray(relative, mode="F")
        relative = np.asarray(
            image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    return relative


def median_align(relative: np.ndarray, sparse_z: np.ndarray) -> tuple[np.ndarray, float, int]:
    anchors = (
        np.isfinite(relative)
        & np.isfinite(sparse_z)
        & (relative > 0)
        & (sparse_z > 0)
    )
    count = int(anchors.sum())
    if count == 0:
        raise RuntimeError("This frame has no valid LiDAR anchors for median alignment")
    scale = float(np.median(sparse_z[anchors] / relative[anchors]))
    return (relative * scale).astype(np.float32), scale, count


def parse_points(specs: list[str], width: int, height: int) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    for index, spec in enumerate(specs, start=1):
        fields = [part.strip() for part in spec.split(",", maxsplit=2)]
        if len(fields) < 2:
            raise ValueError(f"Invalid point {spec!r}; use X,Y or X,Y,LABEL")
        x = int(round(float(fields[0])))
        y = int(round(float(fields[1])))
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Point ({x}, {y}) is outside image bounds "
                f"x=0..{width - 1}, y=0..{height - 1}"
            )
        label = fields[2] if len(fields) == 3 and fields[2] else f"P{index}"
        result.append((x, y, label))
    return result


def select_points(
    rgb: np.ndarray,
    median_depth: np.ndarray,
    poisson_depth: np.ndarray,
    vmax: float,
) -> list[tuple[int, int, str]]:
    height, width = rgb.shape[:2]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Click object points here")
    axes[1].imshow(median_depth, cmap="turbo", vmin=0, vmax=vmax)
    axes[1].set_title("DA3 + median")
    axes[2].imshow(poisson_depth, cmap="turbo", vmin=0, vmax=vmax)
    axes[2].set_title("DA3 + median + Poisson")
    for axis in axes:
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)
        axis.set_axis_off()

    print("\nSelection controls:")
    print("  Left-click object points on the RGB image.")
    print("  Right-click removes the last point.")
    print("  Middle-click or Enter finishes.")
    clicked = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(figure)

    points: list[tuple[int, int, str]] = []
    for index, (x_float, y_float) in enumerate(clicked, start=1):
        x = min(max(int(round(x_float)), 0), width - 1)
        y = min(max(int(round(y_float)), 0), height - 1)
        default = f"P{index}"
        label = input(f"Label for point {index} at ({x}, {y}) [{default}]: ").strip()
        points.append((x, y, label or default))
    return points


def annotate_axis(axis, x: int, y: int, text: str, color: str) -> None:
    axis.scatter([x], [y], s=65, facecolors="none", edgecolors=color, linewidths=2.2)
    axis.annotate(
        text,
        xy=(x, y),
        xytext=(8, -12),
        textcoords="offset points",
        color="white",
        fontsize=9,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.78},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.4},
    )


def render_and_save(
    rgb: np.ndarray,
    median_depth: np.ndarray,
    poisson_depth: np.ndarray,
    points: list[tuple[int, int, str]],
    stem: str,
    scale: float,
    anchor_count: int,
    vmax: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB — exact selected pixels")
    median_image = axes[1].imshow(median_depth, cmap="turbo", vmin=0, vmax=vmax)
    axes[1].set_title("DA3 + median (camera-Z metres)")
    axes[2].imshow(poisson_depth, cmap="turbo", vmin=0, vmax=vmax)
    axes[2].set_title("DA3 + median + Poisson (camera-Z metres)")

    colors = ("lime", "cyan", "yellow", "magenta", "orange", "white")
    for index, (x, y, label) in enumerate(points):
        color = colors[index % len(colors)]
        median_value = float(median_depth[y, x])
        poisson_value = float(poisson_depth[y, x])
        annotate_axis(axes[0], x, y, label, color)
        annotate_axis(axes[1], x, y, f"{label}: {median_value:.2f} m", color)
        annotate_axis(axes[2], x, y, f"{label}: {poisson_value:.2f} m", color)

    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(
        f"{stem} | exact-pixel predictions | median scale={scale:.5f} | "
        f"anchors={anchor_count}",
        fontsize=14,
    )
    colorbar = figure.colorbar(median_image, ax=axes[1:], shrink=0.82, pad=0.015)
    colorbar.set_label("Camera-axis Z depth (m)")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(
    path: Path,
    stem: str,
    points: list[tuple[int, int, str]],
    median_depth: np.ndarray,
    poisson_depth: np.ndarray,
) -> None:
    fieldnames = [
        "scene",
        "point",
        "label",
        "x",
        "y",
        "da3_median_pred_camera_z_m",
        "da3_median_poisson_pred_camera_z_m",
        "poisson_minus_median_m",
        "rangefinder_m",
        "median_abs_error_m",
        "poisson_abs_error_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (x, y, label) in enumerate(points, start=1):
            median_value = float(median_depth[y, x])
            poisson_value = float(poisson_depth[y, x])
            writer.writerow(
                {
                    "scene": stem,
                    "point": f"P{index}",
                    "label": label,
                    "x": x,
                    "y": y,
                    "da3_median_pred_camera_z_m": f"{median_value:.6f}",
                    "da3_median_poisson_pred_camera_z_m": f"{poisson_value:.6f}",
                    "poisson_minus_median_m": f"{poisson_value - median_value:.6f}",
                    "rangefinder_m": "",
                    "median_abs_error_m": "",
                    "poisson_abs_error_m": "",
                }
            )


def main() -> None:
    args = parse_args()
    prepared_root = args.prepared_root.expanduser().resolve()
    da3_dir = args.da3_dir.expanduser().resolve()
    eval_dir = args.eval_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else eval_dir / "object_metric_probes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = read_manifest(prepared_root)
    stem = choose_scene(stems, args.scene)
    _, rgb = load_rgb(prepared_root, stem)
    shape = rgb.shape[:2]

    sparse_path = prepared_root / "depth_full_points" / f"{stem}.npy"
    if not sparse_path.is_file():
        raise FileNotFoundError(f"Missing sparse camera-Z map: {sparse_path}")
    sparse_z = np.load(sparse_path).astype(np.float32)
    if sparse_z.shape != shape:
        raise RuntimeError(f"Sparse shape {sparse_z.shape} does not match RGB {shape}")

    relative = load_relative(da3_dir, stem, shape)
    median_depth, scale, anchor_count = median_align(relative, sparse_z)

    poisson_path = eval_dir / "full_predictions_m" / f"{stem}.npy"
    if not poisson_path.is_file():
        raise FileNotFoundError(
            f"Missing Poisson full map: {poisson_path}\n"
            "Run the real-data evaluator first."
        )
    poisson_depth = np.load(poisson_path).astype(np.float32)
    if poisson_depth.shape != shape:
        raise RuntimeError(f"Poisson shape {poisson_depth.shape} does not match RGB {shape}")

    median_save_dir = output_dir / "full_predictions_median_m"
    median_save_dir.mkdir(parents=True, exist_ok=True)
    np.save(median_save_dir / f"{stem}.npy", median_depth)

    positive = np.concatenate(
        [median_depth[median_depth > 0], poisson_depth[poisson_depth > 0]]
    )
    automatic_vmax = float(np.percentile(positive, 99.0)) if positive.size else 3.0
    vmax = args.display_max_m if args.display_max_m is not None else automatic_vmax
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError(f"Invalid display maximum: {vmax}")

    if args.points:
        points = parse_points(args.points, shape[1], shape[0])
    else:
        points = select_points(rgb, median_depth, poisson_depth, vmax)
    if not points:
        print("No points selected; nothing was written.")
        return

    figure_path = output_dir / f"{stem}__exact_metric_probes.png"
    csv_path = output_dir / f"{stem}__exact_metric_probes.csv"
    render_and_save(
        rgb,
        median_depth,
        poisson_depth,
        points,
        stem,
        scale,
        anchor_count,
        vmax,
        figure_path,
    )
    write_csv(csv_path, stem, points, median_depth, poisson_depth)

    print(f"\nScene: {stem}")
    print(f"Median alignment scale: {scale:.8f} from {anchor_count} LiDAR anchors")
    print("Exact-pixel camera-Z predictions:")
    for index, (x, y, label) in enumerate(points, start=1):
        median_value = float(median_depth[y, x])
        poisson_value = float(poisson_depth[y, x])
        print(
            f"  P{index} {label!r} ({x}, {y}): "
            f"median={median_value:.3f} m, "
            f"Poisson={poisson_value:.3f} m, "
            f"difference={poisson_value - median_value:+.3f} m"
        )
    print(f"Annotated image: {figure_path}")
    print(f"Measurement sheet: {csv_path}")
    print("Fill rangefinder_m later; predictions are camera-axis Z, not radial range.")


if __name__ == "__main__":
    main()
