#!/usr/bin/env python3
"""Prepare synchronized RPLidar/RGB captures for honest DA3 alignment tests.

The CSV range is converted to camera-axis Z depth. Four spatially blocked
cross-validation folds are created so every LiDAR point is evaluated once
without being used for median alignment or Poisson correction in that fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Point:
    source_index: int
    scan_id: int
    angle_deg: float
    range_m: float
    z_m: float
    u: int
    v: int
    t_ms: float


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--sectors", type=int, default=16)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument(
        "--include-margin-beams",
        action="store_true",
        help="Keep beams in the configured +/-4 ms margin outside exposure.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_points(csv_path: Path, metadata: dict, width: int, height: int,
                include_margin: bool) -> tuple[list[Point], dict[str, int]]:
    exposure_ms = float(metadata["exposure_us"]) / 1000.0
    candidates: list[Point] = []
    counts = {"csv_rows": 0, "valid_rows": 0, "projected_rows": 0,
              "outside_exposure": 0, "deduplicated": 0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            counts["csv_rows"] += 1
            if int(row["valid"]) != 1:
                continue
            range_mm = float(row["range_mm"])
            if not math.isfinite(range_mm) or range_mm <= 0:
                continue
            counts["valid_rows"] += 1
            if not row.get("u_px") or not row.get("v_px"):
                continue
            u = int(round(float(row["u_px"])))
            v = int(round(float(row["v_px"])))
            if not (0 <= u < width and 0 <= v < height):
                continue
            counts["projected_rows"] += 1
            t_ms = float(row["t_rel_exposure_ms"])
            if t_ms < 0 or t_ms > exposure_ms:
                counts["outside_exposure"] += 1
                if not include_margin:
                    continue
            angle_deg = float(row["angle_deg"])
            angle_rad = math.radians(angle_deg)
            range_m = range_mm / 1000.0
            # The current session has zero yaw/pitch/roll and zero camera-Z
            # translation. Metric image depth is optical-axis Z, not slant range.
            z_m = range_m * math.cos(angle_rad)
            if not math.isfinite(z_m) or z_m <= 0:
                continue
            candidates.append(Point(
                source_index=index,
                scan_id=int(row["scan_id"]),
                angle_deg=angle_deg,
                range_m=range_m,
                z_m=z_m,
                u=u,
                v=v,
                t_ms=t_ms,
            ))

    # Long exposure can include repeated returns from adjacent revolutions.
    # At an identical projected pixel, retain the nearest camera-Z surface.
    by_pixel: dict[tuple[int, int], Point] = {}
    for point in candidates:
        old = by_pixel.get((point.u, point.v))
        if old is None or point.z_m < old.z_m:
            by_pixel[(point.u, point.v)] = point
    counts["deduplicated"] = len(candidates) - len(by_pixel)
    return sorted(by_pixel.values(), key=lambda p: (p.u, p.v, p.source_index)), counts


def rasterize(points: list[Point], shape: tuple[int, int], radius: int) -> np.ndarray:
    height, width = shape
    depth = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x0, x1 = max(0, point.u - radius), min(width, point.u + radius + 1)
        y0, y1 = max(0, point.v - radius), min(height, point.v + radius + 1)
        patch = depth[y0:y1, x0:x1]
        replace = (patch == 0) | (point.z_m < patch)
        patch[replace] = point.z_m
    return depth


def point_row(stem: str, point: Point, sector: int, fold: int | str) -> dict:
    return {
        "stem": stem,
        "fold": fold,
        "source_index": point.source_index,
        "scan_id": point.scan_id,
        "sector": sector,
        "u": point.u,
        "v": point.v,
        "angle_deg": f"{point.angle_deg:.6f}",
        "range_m": f"{point.range_m:.6f}",
        "z_m": f"{point.z_m:.6f}",
        "t_rel_exposure_ms": f"{point.t_ms:.6f}",
    }


def main() -> None:
    args = arguments()
    if args.folds < 2 or args.sectors < args.folds or args.sectors % args.folds:
        raise ValueError("--sectors must be a multiple of --folds, with folds >= 2")
    if args.splat_radius < 0:
        raise ValueError("--splat-radius must be non-negative")

    root = args.dataset_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = json.loads((root / "metadata/session.json").read_text(encoding="utf-8"))
    calibration = session["final_calibration"]
    rotations = [abs(float(calibration[k])) for k in ("yaw_deg", "pitch_deg", "roll_deg")]
    if any(value > 1e-9 for value in rotations) or abs(float(calibration["t_z_mm"])) > 1e-9:
        raise RuntimeError(
            "This adapter's Z conversion is validated for this recording's zero "
            "yaw/pitch/roll/t_z calibration. Add the calibrated rigid transform "
            "before using a different session."
        )

    rgb_out = output / "rgb"
    full_points_out = output / "depth_full_points"
    full_splat_out = output / "depth_full_splat"
    for directory in (rgb_out, full_points_out, full_splat_out):
        directory.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        (output / f"fold_{fold}" / "depth_fit_points").mkdir(parents=True, exist_ok=True)
        (output / f"fold_{fold}" / "depth_fit_splat").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    anchor_rows: list[dict] = []
    heldout_rows: dict[int, list[dict]] = {fold: [] for fold in range(args.folds)}
    csv_files = sorted((root / "lidar_csv").glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {root / 'lidar_csv'}")

    for csv_path in csv_files:
        stem = csv_path.stem
        rgb_path = root / "cam_rgb" / f"{stem}.png"
        metadata_path = root / "metadata" / f"{stem}.json"
        if not rgb_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete RGB/metadata pair for {stem}")
        with Image.open(rgb_path) as image:
            width, height = image.size
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        points, counts = read_points(
            csv_path, metadata, width, height, args.include_margin_beams
        )
        if len(points) < 12:
            raise RuntimeError(f"Only {len(points)} usable anchors in {stem}")
        shutil.copy2(rgb_path, rgb_out / rgb_path.name)
        np.save(full_points_out / f"{stem}.npy", rasterize(points, (height, width), 0))
        np.save(full_splat_out / f"{stem}.npy",
                rasterize(points, (height, width), args.splat_radius))

        sectors = {point: min(args.sectors - 1, point.u * args.sectors // width)
                   for point in points}
        for point in points:
            anchor_rows.append(point_row(stem, point, sectors[point], "all"))
        fold_counts = []
        for fold in range(args.folds):
            heldout = [point for point in points if sectors[point] % args.folds == fold]
            fit = [point for point in points if sectors[point] % args.folds != fold]
            fold_root = output / f"fold_{fold}"
            np.save(fold_root / "depth_fit_points" / f"{stem}.npy",
                    rasterize(fit, (height, width), 0))
            np.save(fold_root / "depth_fit_splat" / f"{stem}.npy",
                    rasterize(fit, (height, width), args.splat_radius))
            for point in heldout:
                heldout_rows[fold].append(point_row(stem, point, sectors[point], fold))
            fold_counts.append(len(heldout))

        manifest.append({
            "stem": stem,
            "width": width,
            "height": height,
            **counts,
            "usable_unique_points": len(points),
            "range_min_m": min(p.range_m for p in points),
            "range_max_m": max(p.range_m for p in points),
            "z_min_m": min(p.z_m for p in points),
            "z_max_m": max(p.z_m for p in points),
            **{f"fold_{fold}_heldout": fold_counts[fold] for fold in range(args.folds)},
        })

    write_csv(output / "manifest.csv", manifest)
    write_csv(output / "anchor_points.csv", anchor_rows)
    for fold, rows in heldout_rows.items():
        write_csv(output / f"fold_{fold}" / "heldout_points.csv", rows)
    shutil.copy2(root / "metadata/session.json", output / "session.json")

    total = sum(int(row["usable_unique_points"]) for row in manifest)
    print(f"Prepared {len(manifest)} frames with {total} usable unique points")
    print("DA3 alignment input: depth_full_points / fold_N/depth_fit_points")
    print("Any2Full input: depth_full_splat / fold_N/depth_fit_splat")
    print("Depth values are camera-axis Z metres, not radial LiDAR range")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
