#!/usr/bin/env python3
"""Locked KITTI four-beam benchmark for DA3 alignment + Poisson vs Any2Full.

The benchmark has five explicit stages:

``prepare``
    Extract four real Velodyne elevation bands from KITTI's 64-line input.
    The primary depth-coverage layout is line_spec 7 12 22 37, representing
    the elevation bands centred at -1, -3, -7, and -13 degrees.  Their nominal
    flat-ground intersections are logarithmically spread across about 99, 33,
    14, and 7.5 metres for a 1.73 m mount.  Dense ground truth is never sampled
    to form the input.  Exact raw .bin input is preferred; reconstruction from
    the official projected ``velodyne_raw`` depth maps is supported and recorded.

``audit``
    Verify matching RGB, sparse input, calibration, ground truth, split, and
    one-pixel-return invariants before any model is run.

``infer-da3``
    Cache one DA3-SMALL relative-depth prediction per frame.

``select``
    On development drives only, compare several per-frame alignment families,
    both before and after the project's validated ``existing_poisson``.  Select
    the alignment whose post-Poisson full-image macro RMSE is smallest.

``predict`` and ``evaluate``
    Freeze that choice, create DA3 predictions for locked test drives, then
    compare them with Any2Full using identical frames, sparse inputs, masks,
    KITTI quantization, per-scene macro averages, and drive-cluster bootstrap
    confidence intervals.  No per-image graphs are generated.

This is intentionally a benchmark rather than a training script.  A candidate
alignment may use the sparse four-beam measurements in its own test frame, but
it never sees that frame's dense ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


KITTI_SCALE = 256.0
KITTI_MIN_M = 1.0 / KITTI_SCALE
KITTI_MAX_M = 65535.0 / KITTI_SCALE
DEFAULT_TARGET_ANGLES_DEG = (-1.0, -3.0, -7.0, -13.0)
# With 0.4-degree rows beginning at +2 degrees, each ID k represents the
# target-centred interval 2 - (k + 1)*0.4 < alpha <= 2 - k*0.4.
DEFAULT_LINE_SPEC = (7, 12, 22, 37)
CAR_DETECTION_LINE_SPEC = (7, 8, 9, 10)
CONTROL_LINE_SPEC = (5, 7, 9, 11)
ALIGNMENTS = (
    "median_scale",
    "least_squares_scale",
    "least_squares_affine",
    "huber_affine",
    "inverse_huber_affine",
    "log_affine",
    "isotonic_monotonic",
)
DA3_METHODS = (
    "da3_median",
    "da3_median_existing_poisson",
    "da3_selected_alignment",
    "da3_selected_alignment_existing_poisson",
)
ALL_METHODS = (*DA3_METHODS, "any2full_vits")
METHOD_LABELS = {
    "da3_median": "DA3-SMALL + median",
    "da3_median_existing_poisson": "DA3-SMALL + median + existing Poisson",
    "da3_selected_alignment": "DA3-SMALL + selected alignment",
    "da3_selected_alignment_existing_poisson":
        "DA3-SMALL + selected alignment + existing Poisson",
    "any2full_vits": "Any2Full-vits",
}
REGIONS = (
    "all_valid",
    "outside_four_beam_pixels",
    "range_0_20m",
    "range_20_40m",
    "range_40_80m",
    "range_over_80m",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Create matched four-beam KITTI input")
    prepare.add_argument("--selection-root", type=Path, required=True)
    prepare.add_argument("--calib-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--source", choices=("auto", "raw", "projected"), default="auto")
    prepare.add_argument("--raw-root", type=Path, default=None)
    prepare.add_argument(
        "--line-spec", type=int, nargs=4, default=list(DEFAULT_LINE_SPEC),
        help=(
            "four 0.4-degree angular rows; default 7 12 22 37 equals "
            "the -1/-3/-7/-13 degree +/-0.2 degree bands"
        ),
    )
    prepare.add_argument("--angular-width", type=int, default=1024)
    prepare.add_argument("--dev-fraction", type=float, default=0.20)
    prepare.add_argument("--seed", type=int, default=2022)
    prepare.add_argument("--limit", type=int, default=None)
    prepare.add_argument("--scene", action="append", default=[])
    prepare.add_argument("--copy-files", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")

    audit = commands.add_parser("audit", help="Audit a prepared benchmark")
    audit.add_argument("--data-root", type=Path, required=True)

    infer = commands.add_parser("infer-da3", help="Cache DA3-SMALL relative depth")
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--relative-dir", type=Path, required=True)
    infer.add_argument("--checkpoint", default="depth-anything/DA3-SMALL")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--process-res", type=int, default=504)
    infer.add_argument("--split", choices=("all", "dev", "test"), default="all")
    infer.add_argument("--overwrite", action="store_true")

    select = commands.add_parser("select", help="Select alignment on development drives")
    add_model_inputs(select)
    select.add_argument("--selection-dir", type=Path, required=True)
    select.add_argument("--rtol", type=float, default=1e-5)
    select.add_argument("--maxiter", type=int, default=5000)
    select.add_argument("--resume", action="store_true")

    predict = commands.add_parser("predict", help="Create frozen DA3 test predictions")
    add_model_inputs(predict)
    predict.add_argument("--selection-json", type=Path, required=True)
    predict.add_argument("--prediction-root", type=Path, required=True)
    predict.add_argument("--rtol", type=float, default=1e-5)
    predict.add_argument("--maxiter", type=int, default=5000)
    predict.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Evaluate DA3 and Any2Full on locked test")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--any2full-dir", type=Path, required=True)
    evaluate.add_argument("--selection-json", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate.add_argument("--seed", type=int, default=12345)

    selftest = commands.add_parser("self-test", help="Run dependency-free numerical tests")
    selftest.add_argument("--verbose", action="store_true")
    return parser


def add_model_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--relative-dir", type=Path, required=True)
    parser.add_argument("--da3-root", type=Path, required=True)


def read_manifest(root: Path) -> list[dict[str, str]]:
    path = root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def manifest_rows(root: Path, split: str) -> list[dict[str, str]]:
    rows = read_manifest(root)
    return rows if split == "all" else [row for row in rows if row["split"] == split]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_to_gt_stem(stem: str) -> str:
    return stem.replace("_sync_image_", "_sync_groundtruth_depth_", 1)


def image_to_velodyne_stem(stem: str) -> str:
    return stem.replace("_sync_image_", "_sync_velodyne_raw_", 1)


def resolve_selection_file(directory: Path, stem: str, suffix: str, kind: str) -> Path:
    candidates = [directory / f"{stem}{suffix}"]
    if kind == "groundtruth":
        candidates.append(directory / f"{image_to_gt_stem(stem)}{suffix}")
    elif kind == "velodyne":
        candidates.append(directory / f"{image_to_velodyne_stem(stem)}{suffix}")
    for path in candidates:
        if path.is_file():
            return path
    frame = parse_frame_stem(stem)
    matches = sorted(directory.glob(f"{frame['drive']}*{frame['frame']}*image_{frame['camera']}{suffix}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve {kind} for {stem} in {directory}: {matches}")


STEM_PATTERN = re.compile(
    r"^(?P<drive>(?P<date>\d{4}_\d{2}_\d{2})_drive_\d{4}_sync)"
    r"_image_(?P<frame>\d{10})_image_(?P<camera>\d{2})$"
)


def parse_frame_stem(stem: str) -> dict[str, str]:
    match = STEM_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Unexpected KITTI selection filename: {stem}")
    return match.groupdict()


def load_depth_png(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if raw.ndim != 2:
        raise ValueError(f"Expected single-channel depth PNG: {path}")
    return raw.astype(np.float32) / KITTI_SCALE


def load_npy_2d(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.shape}: {path}")
    return array


def parse_plain_matrix(path: Path) -> np.ndarray:
    numbers = np.fromstring(path.read_text(encoding="utf-8").replace(",", " "), sep=" ")
    if numbers.size < 9:
        raise ValueError(f"Expected at least nine intrinsic values: {path}")
    return numbers[:9].reshape(3, 3)


def parse_calibration(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, text = line.split(":", 1)
        array = np.fromstring(text, sep=" ")
        if array.size:
            values[key.strip()] = array
    return values


def find_calibration_file(root: Path, date: str, name: str) -> Path:
    direct = [root / date / name, root / f"{date}_calib" / date / name, root / name]
    for path in direct:
        if path.is_file():
            return path
    matches = sorted(root.glob(f"**/{date}/{name}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely find {date}/{name} under {root}: {matches}")


def calibration_bundle(calib_root: Path, date: str, camera: str) -> dict[str, np.ndarray]:
    cam_values = parse_calibration(find_calibration_file(calib_root, date, "calib_cam_to_cam.txt"))
    velo_values = parse_calibration(find_calibration_file(calib_root, date, "calib_velo_to_cam.txt"))
    p_key = f"P_rect_{camera}"
    if p_key not in cam_values:
        raise KeyError(f"{p_key} missing from camera calibration for {date}")
    return {
        "P": cam_values[p_key].reshape(3, 4),
        "R_rect": cam_values["R_rect_00"].reshape(3, 3),
        "R_velo_cam": velo_values["R"].reshape(3, 3),
        "T_velo_cam": velo_values["T"].reshape(3),
    }


def rectified_to_velodyne(points_rect: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
    unrectified = points_rect @ calib["R_rect"]
    return (unrectified - calib["T_velo_cam"]) @ calib["R_velo_cam"]


def velodyne_to_rectified(points_velo: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
    camera = points_velo @ calib["R_velo_cam"].T + calib["T_velo_cam"]
    return camera @ calib["R_rect"].T


def map_crop_pixels_to_original(
    u: np.ndarray, v: np.ndarray, crop_k: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u_original = (u - crop_k[0, 2]) * (p[0, 0] / crop_k[0, 0]) + p[0, 2]
    v_original = (v - crop_k[1, 2]) * (p[1, 1] / crop_k[1, 1]) + p[1, 2]
    return u_original, v_original


def map_original_pixels_to_crop(
    u: np.ndarray, v: np.ndarray, crop_k: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u_crop = (u - p[0, 2]) * (crop_k[0, 0] / p[0, 0]) + crop_k[0, 2]
    v_crop = (v - p[1, 2]) * (crop_k[1, 1] / p[1, 1]) + crop_k[1, 2]
    return u_crop, v_crop


def backproject_known_z(
    u: np.ndarray, v: np.ndarray, z: np.ndarray, p: np.ndarray
) -> np.ndarray:
    a11 = p[0, 0] - u * p[2, 0]
    a12 = p[0, 1] - u * p[2, 1]
    a21 = p[1, 0] - v * p[2, 0]
    a22 = p[1, 1] - v * p[2, 1]
    b1 = -(p[0, 2] - u * p[2, 2]) * z - (p[0, 3] - u * p[2, 3])
    b2 = -(p[1, 2] - v * p[2, 2]) * z - (p[1, 3] - v * p[2, 3])
    determinant = a11 * a22 - a12 * a21
    if np.any(np.abs(determinant) < 1e-12):
        raise RuntimeError("Singular camera projection encountered during back-projection")
    x = (b1 * a22 - a12 * b2) / determinant
    y = (a11 * b2 - b1 * a21) / determinant
    return np.column_stack((x, y, z))


def project_rectified(points_rect: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((points_rect, np.ones(len(points_rect))))
    image = homogeneous @ p.T
    denom = image[:, 2]
    u = image[:, 0] / denom
    v = image[:, 1] / denom
    return u, v, points_rect[:, 2]


def official_angular_ids(points_velo: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points_velo[:, 0], points_velo[:, 1], points_velo[:, 2]
    distance = np.linalg.norm(points_velo[:, :3], axis=1)
    radius = np.hypot(x, y)
    distance = np.maximum(distance, 1e-6)
    radius = np.maximum(radius, 1e-6)
    dtheta = np.radians(0.4)
    dphi = np.radians(90.0 / width)
    phi = np.radians(45.0) - np.arcsin(np.clip(y / radius, -1.0, 1.0))
    theta = np.radians(2.0) - np.arcsin(np.clip(z / distance, -1.0, 1.0))
    line = np.clip((theta / dtheta).astype(np.int64), 0, 63)
    column = np.clip((phi / dphi).astype(np.int64), 0, width - 1)
    return line, column


def angular_map_keep_last(line: np.ndarray, column: np.ndarray, width: int) -> np.ndarray:
    linear = line * width + column
    reverse_unique = np.unique(linear[::-1], return_index=True)[1]
    return np.sort(len(linear) - 1 - reverse_unique)


def zbuffer_depth(
    u: np.ndarray, v: np.ndarray, z: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    height, width = shape
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    valid = np.isfinite(z) & (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    linear = y[valid] * width + x[valid]
    depth = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(depth, linear, z[valid])
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(shape).astype(np.float32)


def raw_bin_path(raw_root: Path, info: dict[str, str]) -> Path:
    direct = [
        raw_root / info["date"] / info["drive"] / "velodyne_points/data" / f"{info['frame']}.bin",
        raw_root / info["drive"] / "velodyne_points/data" / f"{info['frame']}.bin",
    ]
    for path in direct:
        if path.is_file():
            return path
    matches = sorted(raw_root.glob(f"**/{info['drive']}/velodyne_points/data/{info['frame']}.bin"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot find raw Velodyne frame for {info['drive']} {info['frame']}")


def four_beam_from_raw(
    path: Path,
    crop_k: np.ndarray,
    calib: dict[str, np.ndarray],
    shape: tuple[int, int],
    line_spec: tuple[int, ...],
    angular_width: int,
) -> tuple[np.ndarray, dict[int, int]]:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Malformed KITTI Velodyne file: {path}")
    points = raw.reshape(-1, 4)
    rect = velodyne_to_rectified(points[:, :3], calib)
    u_original, v_original, z = project_rectified(rect, calib["P"])
    u_crop, v_crop = map_original_pixels_to_crop(u_original, v_original, crop_k, calib["P"])
    height, width = shape
    fov = (
        np.isfinite(z) & (z > 0) & (u_crop >= 0) & (u_crop < width)
        & (v_crop >= 0) & (v_crop < height)
    )
    geometry = (
        (points[:, 0] >= 0) & (points[:, 0] < 120)
        & (points[:, 1] >= -50) & (points[:, 1] < 50)
        & (points[:, 2] >= -2.5) & (points[:, 2] < 1.5)
    )
    keep = fov & geometry
    points, rect = points[keep], rect[keep]
    u_crop, v_crop = u_crop[keep], v_crop[keep]
    line, column = official_angular_ids(points[:, :3], angular_width)
    mapped = angular_map_keep_last(line, column, angular_width)
    points, rect, u_crop, v_crop, line = (
        points[mapped], rect[mapped], u_crop[mapped], v_crop[mapped], line[mapped]
    )
    selected = np.isin(line, line_spec)
    counts = {beam: int(np.sum(line[selected] == beam)) for beam in line_spec}
    return zbuffer_depth(u_crop[selected], v_crop[selected], rect[selected, 2], shape), counts


def four_beam_from_projected(
    projected_path: Path,
    crop_k: np.ndarray,
    calib: dict[str, np.ndarray],
    line_spec: tuple[int, ...],
    angular_width: int,
) -> tuple[np.ndarray, dict[int, int]]:
    full = load_depth_png(projected_path)
    y, x = np.nonzero(np.isfinite(full) & (full > 0))
    z = full[y, x].astype(np.float64)
    u_original, v_original = map_crop_pixels_to_original(
        x.astype(np.float64), y.astype(np.float64), crop_k, calib["P"]
    )
    rect = backproject_known_z(u_original, v_original, z, calib["P"])
    velo = rectified_to_velodyne(rect, calib)
    line, column = official_angular_ids(velo, angular_width)
    mapped = angular_map_keep_last(line, column, angular_width)
    line, x, y, z = line[mapped], x[mapped], y[mapped], z[mapped]
    selected = np.isin(line, line_spec)
    counts = {beam: int(np.sum(line[selected] == beam)) for beam in line_spec}
    sparse = np.zeros_like(full, dtype=np.float32)
    sparse[y[selected], x[selected]] = z[selected].astype(np.float32)
    return sparse, counts


def deterministic_group_split(
    stems: list[str], dev_fraction: float, seed: int
) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        groups[parse_frame_stem(stem)["drive"]].append(stem)
    if len(groups) < 2:
        raise RuntimeError("A drive-disjoint split requires at least two KITTI drives")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    target = max(1, int(round(dev_fraction * len(stems))))
    dev_groups: set[str] = set()
    count = 0
    for group in ordered:
        if count >= target and dev_groups:
            break
        dev_groups.add(group)
        count += len(groups[group])
    if len(dev_groups) == len(groups):
        dev_groups.remove(ordered[-1])
    return {stem: ("dev" if parse_frame_stem(stem)["drive"] in dev_groups else "test") for stem in stems}


def link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def select_stems(selection_root: Path, scenes: list[str], limit: int | None) -> list[str]:
    stems = sorted(path.stem for path in (selection_root / "image").glob("*.png"))
    if scenes:
        wanted = set(scenes)
        stems = [stem for stem in stems if stem in wanted]
        missing = wanted - set(stems)
        if missing:
            raise FileNotFoundError(f"Requested scenes not found: {sorted(missing)}")
    if limit is not None:
        stems = stems[:limit]
    if not stems:
        raise RuntimeError(f"No KITTI RGB images in {selection_root / 'image'}")
    return stems


def command_prepare(args: argparse.Namespace) -> None:
    selection = args.selection_root.expanduser().resolve()
    calib_root = args.calib_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else None
    line_spec = tuple(args.line_spec)
    if len(set(line_spec)) != 4 or any(beam < 0 or beam > 63 for beam in line_spec):
        raise ValueError("--line-spec must contain four distinct IDs in [0, 63]")
    if not 0.0 < args.dev_fraction < 1.0:
        raise ValueError("--dev-fraction must lie in (0, 1)")
    if args.angular_width < 16:
        raise ValueError("--angular-width is implausibly small")
    stems = select_stems(selection, args.scene, args.limit)
    split = deterministic_group_split(stems, args.dev_fraction, args.seed)

    source_mode = args.source
    if source_mode == "auto":
        source_mode = "raw" if raw_root is not None else "projected"
    if source_mode == "raw" and raw_root is None:
        raise ValueError("--source raw requires --raw-root")
    if source_mode == "projected" and not (selection / "velodyne_raw").is_dir():
        raise FileNotFoundError(selection / "velodyne_raw")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Non-empty output exists; add --overwrite: {output}")
    if output.exists() and args.overwrite:
        # Remove only benchmark-owned children of the explicit output root.
        for name in (
            "rgb", "groundtruth_depth", "intrinsics", "sparse_input_m",
            "any2full_dev", "any2full_test",
        ):
            child = output / name
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        for name in ("manifest.csv", "protocol.json"):
            child = output / name
            if child.is_file() or child.is_symlink():
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for index, stem in enumerate(stems, 1):
        info = parse_frame_stem(stem)
        rgb_source = resolve_selection_file(selection / "image", stem, ".png", "image")
        gt_source = resolve_selection_file(selection / "groundtruth_depth", stem, ".png", "groundtruth")
        intr_source = resolve_selection_file(selection / "intrinsics", stem, ".txt", "intrinsics")
        with Image.open(rgb_source) as image:
            shape = (image.height, image.width)
        crop_k = parse_plain_matrix(intr_source)
        calib = calibration_bundle(calib_root, info["date"], info["camera"])
        if source_mode == "raw":
            source_path = raw_bin_path(raw_root, info)  # type: ignore[arg-type]
            sparse, beam_counts = four_beam_from_raw(
                source_path, crop_k, calib, shape, line_spec, args.angular_width
            )
        else:
            source_path = resolve_selection_file(
                selection / "velodyne_raw", stem, ".png", "velodyne"
            )
            sparse, beam_counts = four_beam_from_projected(
                source_path, crop_k, calib, line_spec, args.angular_width
            )
        anchors = np.isfinite(sparse) & (sparse > 0)
        if int(anchors.sum()) < 4:
            raise RuntimeError(f"Only {int(anchors.sum())} four-beam returns for {stem}")
        npy_path = output / "sparse_input_m" / f"{stem}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, sparse.astype(np.float32))
        link_or_copy(rgb_source, output / "rgb" / f"{stem}.png", args.copy_files)
        link_or_copy(gt_source, output / "groundtruth_depth" / f"{stem}.png", args.copy_files)
        link_or_copy(intr_source, output / "intrinsics" / f"{stem}.txt", args.copy_files)
        subset_root = output / f"any2full_{split[stem]}"
        link_or_copy(rgb_source, subset_root / "rgb" / f"{stem}.png", args.copy_files)
        link_or_copy(npy_path, subset_root / "sparse_input_m" / f"{stem}.npy", args.copy_files)
        depths = sparse[anchors]
        row: dict[str, Any] = {
            "stem": stem,
            "group": info["drive"],
            "split": split[stem],
            "height": shape[0],
            "width": shape[1],
            "source_mode": source_mode,
            "source_path": str(source_path),
            "anchor_count": int(anchors.sum()),
            "min_depth_m": float(depths.min()),
            "median_depth_m": float(np.median(depths)),
            "max_depth_m": float(depths.max()),
        }
        row.update({f"beam_{beam}_angular_cells": beam_counts[beam] for beam in line_spec})
        manifest.append(row)
        print(
            f"[{index:4d}/{len(stems)}] {stem} split={split[stem]} "
            f"anchors={int(anchors.sum())} beams={beam_counts}",
            flush=True,
        )
    write_csv(output / "manifest.csv", manifest)
    metadata = {
        "protocol": "KITTI real Velodyne four-beam reconstruction",
        "line_spec": list(line_spec),
        "target_elevation_angles_deg": (
            list(DEFAULT_TARGET_ANGLES_DEG) if line_spec == DEFAULT_LINE_SPEC else None
        ),
        "angle_band_half_width_deg": 0.2 if line_spec == DEFAULT_LINE_SPEC else None,
        "primary_log_depth_coverage_layout": list(DEFAULT_LINE_SPEC),
        "prior_car_detection_layout": list(CAR_DETECTION_LINE_SPEC),
        "equidistant_control_layout": list(CONTROL_LINE_SPEC),
        "angular_map_width": args.angular_width,
        "source_mode": source_mode,
        "depth_definition": "rectified camera-forward Z in metres",
        "input_splat_radius": 0,
        "groundtruth_used_to_form_input": False,
        "split_unit": "raw drive",
        "dev_fraction_requested": args.dev_fraction,
        "seed": args.seed,
        "scene_count": len(manifest),
        "dev_scene_count": sum(row["split"] == "dev" for row in manifest),
        "test_scene_count": sum(row["split"] == "test" for row in manifest),
        "projected_source_limitation": (
            "The official projected 64-line map cannot recover rare returns already lost in "
            "pixel collisions; raw .bin mode is the exact-source sensitivity check."
            if source_mode == "projected" else None
        ),
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nPrepared {len(manifest)} scenes at {output}")
    print(f"Development: {metadata['dev_scene_count']}  Locked test: {metadata['test_scene_count']}")
    print(f"Line spec: {line_spec}; source mode: {source_mode}; one pixel per visible return")
    if line_spec == DEFAULT_LINE_SPEC:
        print(f"Target elevation bands: {DEFAULT_TARGET_ANGLES_DEG} deg (half-width 0.2 deg)")


def command_audit(args: argparse.Namespace) -> None:
    root = args.data_root.expanduser().resolve()
    rows = read_manifest(root)
    errors: list[str] = []
    counts: list[int] = []
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        stem = row["stem"]
        split_groups[row["split"]].add(row["group"])
        expected = [
            root / "rgb" / f"{stem}.png",
            root / "groundtruth_depth" / f"{stem}.png",
            root / "intrinsics" / f"{stem}.txt",
            root / "sparse_input_m" / f"{stem}.npy",
            root / f"any2full_{row['split']}" / "rgb" / f"{stem}.png",
            root / f"any2full_{row['split']}" / "sparse_input_m" / f"{stem}.npy",
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            errors.append(f"{stem}: missing {missing}")
            continue
        with Image.open(expected[0]) as image:
            shape = (image.height, image.width)
        sparse = load_npy_2d(expected[3])
        gt = load_depth_png(expected[1])
        if sparse.shape != shape or gt.shape != shape:
            errors.append(f"{stem}: RGB {shape}, sparse {sparse.shape}, GT {gt.shape}")
        count = int(np.sum(np.isfinite(sparse) & (sparse > 0)))
        counts.append(count)
        if count != int(row["anchor_count"]):
            errors.append(f"{stem}: manifest anchors {row['anchor_count']} != {count}")
    overlap = split_groups["dev"] & split_groups["test"]
    if overlap:
        errors.append(f"Drive leakage between dev and test: {sorted(overlap)}")
    if errors:
        raise RuntimeError("AUDIT FAILED\n" + "\n".join(errors[:30]))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    print("===== FOUR-BEAM KITTI AUDIT PASSED =====")
    print(f"Scenes:             {len(rows)}")
    print(f"Development/test:   {sum(r['split']=='dev' for r in rows)}/{sum(r['split']=='test' for r in rows)}")
    print(f"Drive groups:       {len(split_groups['dev'])}/{len(split_groups['test'])}")
    print(f"Angular row IDs:    {protocol['line_spec']}")
    if protocol.get("target_elevation_angles_deg"):
        print(f"Target angles:      {protocol['target_elevation_angles_deg']} deg")
    print(f"Source mode:        {protocol['source_mode']}")
    print(f"Anchors min/median/max: {min(counts)}/{np.median(counts):.0f}/{max(counts)}")
    min_depths = np.array([float(row["min_depth_m"]) for row in rows])
    max_depths = np.array([float(row["max_depth_m"]) for row in rows])
    print(
        "Per-scene union depth span (median min/max): "
        f"{np.median(min_depths):.2f}/{np.median(max_depths):.2f} m"
    )
    for line_index, beam in enumerate(protocol["line_spec"]):
        field = f"beam_{beam}_angular_cells"
        if field not in rows[0]:
            continue
        line_counts = np.array([int(row[field]) for row in rows])
        missed = int(np.sum(line_counts < 5))
        target_angles = protocol.get("target_elevation_angles_deg")
        angle_text = (
            f" ({target_angles[line_index]:+.1f} deg)" if target_angles else ""
        )
        print(
            f"Row {beam:2d}{angle_text} anchors min/median/max: "
            f"{line_counts.min()}/{np.median(line_counts):.0f}/{line_counts.max()}  "
            f"scenes<5={missed}/{len(rows)}"
        )
        if missed:
            print(f"WARNING: angular row {beam} missed or nearly missed {missed} scenes")
    print("Splat radius:       0")
    print("GT used as input:   no")


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32, copy=False)
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(
        image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32
    )


def extract_da3_depth(prediction: Any) -> np.ndarray:
    value = prediction.depth if hasattr(prediction, "depth") else prediction["depth"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"DA3 returned unexpected shape: {np.asarray(value).shape}")
    return array


def command_infer_da3(args: argparse.Namespace) -> None:
    import torch
    from depth_anything_3.api import DepthAnything3

    root = args.data_root.expanduser().resolve()
    relative_dir = args.relative_dir.expanduser().resolve()
    relative_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(root, args.split)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    print(f"Loading {args.checkpoint} on {args.device}", flush=True)
    model = DepthAnything3.from_pretrained(args.checkpoint).to(args.device)
    model.eval()
    for index, row in enumerate(rows, 1):
        stem = row["stem"]
        output = relative_dir / f"{stem}.npy"
        if output.is_file() and not args.overwrite:
            cached = load_npy_2d(output)
            if cached.shape == (int(row["height"]), int(row["width"])) and np.all(np.isfinite(cached)):
                print(f"[{index:4d}/{len(rows)}] {stem} cached", flush=True)
                continue
        rgb = root / "rgb" / f"{stem}.png"
        with torch.inference_mode():
            prediction = model.inference(image=[str(rgb)], process_res=args.process_res)
        depth = resize_depth(extract_da3_depth(prediction), (int(row["height"]), int(row["width"])))
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise RuntimeError(f"DA3 returned no positive finite pixels for {stem}")
        depth = np.array(depth, dtype=np.float32, copy=True)
        depth[~valid] = float(np.median(depth[valid]))
        np.save(output, depth.astype(np.float32))
        print(f"[{index:4d}/{len(rows)}] {stem} saved", flush=True)


def anchor_arrays(relative: np.ndarray, sparse: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(relative) & (relative > 0) & np.isfinite(sparse) & (sparse > 0)
    if int(mask.sum()) < 8:
        raise RuntimeError(f"Only {int(mask.sum())} valid alignment anchors")
    return mask, relative[mask].astype(np.float64), sparse[mask].astype(np.float64)


def huber_regression(x: np.ndarray, y: np.ndarray, affine: bool = True) -> np.ndarray:
    design = np.column_stack((x, np.ones_like(x))) if affine else x[:, None]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(40):
        residual = y - design @ beta
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center)) + 1e-9
        threshold = 1.345 * scale
        weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1e-12))
        weighted = design * np.sqrt(weights)[:, None]
        target = y * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted, target, rcond=None)[0]
        if np.linalg.norm(updated - beta) <= 1e-9 * (1.0 + np.linalg.norm(beta)):
            beta = updated
            break
        beta = updated
    return beta


def pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    means: list[float] = []
    totals: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        means.append(float(value)); totals.append(float(weight)); starts.append(index); ends.append(index + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            total = totals[-2] + totals[-1]
            mean = (means[-2] * totals[-2] + means[-1] * totals[-1]) / total
            means[-2:] = [mean]; totals[-2:] = [total]
            starts[-2:] = [starts[-2]]; ends[-2:] = [ends[-1]]
    output = np.empty_like(values, dtype=np.float64)
    for mean, start, end in zip(means, starts, ends):
        output[start:end] = mean
    return output


def alignment_prediction(
    name: str, relative: np.ndarray, sparse: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    mask, x, y = anchor_arrays(relative, sparse)
    median_scale = float(np.median(y / np.maximum(x, 1e-12)))
    median_prediction = relative.astype(np.float64) * median_scale
    diagnostics: dict[str, Any] = {"anchors": int(mask.sum()), "median_scale": median_scale}
    if name == "median_scale":
        prediction = median_prediction
        diagnostics.update({"a": median_scale, "b": 0.0})
    elif name == "least_squares_scale":
        a = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        if not np.isfinite(a) or a <= 0:
            a = median_scale
        prediction = a * relative
        diagnostics.update({"a": a, "b": 0.0})
    elif name in ("least_squares_affine", "huber_affine"):
        design = np.column_stack((x, np.ones_like(x)))
        beta = (
            np.linalg.lstsq(design, y, rcond=None)[0]
            if name == "least_squares_affine" else huber_regression(x, y, affine=True)
        )
        a, b = map(float, beta)
        if not np.isfinite(a) or a <= 0 or not np.isfinite(b):
            a, b = median_scale, 0.0
            diagnostics["fit_fallback"] = "median_scale"
        prediction = a * relative + b
        diagnostics.update({"a": a, "b": b})
    elif name == "inverse_huber_affine":
        beta = huber_regression(1.0 / np.maximum(x, 1e-8), 1.0 / np.maximum(y, 1e-8), True)
        a, b = map(float, beta)
        denominator = a / np.maximum(relative.astype(np.float64), 1e-8) + b
        prediction = 1.0 / denominator
        diagnostics.update({"a": a, "b": b})
    elif name == "log_affine":
        beta = huber_regression(np.log(x), np.log(y), True)
        a, b = map(float, beta)
        if not np.isfinite(a) or a <= 0 or not np.isfinite(b):
            a, b = 1.0, math.log(median_scale)
            diagnostics["fit_fallback"] = "median_scale"
        prediction = np.exp(a * np.log(np.maximum(relative.astype(np.float64), 1e-8)) + b)
        diagnostics.update({"a": a, "b": b})
    elif name == "isotonic_monotonic":
        order = np.argsort(x)
        sorted_x, sorted_y = x[order], y[order]
        unique_x, inverse = np.unique(sorted_x, return_inverse=True)
        grouped_y = np.zeros(len(unique_x)); grouped_w = np.zeros(len(unique_x))
        np.add.at(grouped_y, inverse, sorted_y); np.add.at(grouped_w, inverse, 1.0)
        grouped_y /= grouped_w
        fitted = pava(grouped_y, grouped_w)
        prediction = np.interp(relative.astype(np.float64), unique_x, fitted)
        diagnostics.update({"knots": len(unique_x), "a": math.nan, "b": math.nan})
    else:
        raise KeyError(name)
    invalid = ~np.isfinite(prediction) | (prediction <= KITTI_MIN_M)
    diagnostics["invalid_pixels_before_repair"] = int(invalid.sum())
    if float(np.mean(invalid)) > 0.05:
        prediction = median_prediction
        diagnostics["prediction_fallback"] = "full_median_scale"
    else:
        prediction = np.where(invalid, median_prediction, prediction)
    prediction = np.clip(prediction, KITTI_MIN_M, KITTI_MAX_M)
    return prediction.astype(np.float32), diagnostics


def load_existing_poisson(da3_root: Path) -> Callable[..., Any]:
    path = da3_root / "experiments/lidar_alignment/ibims/compare_median_poisson_oasis_100.py"
    if not path.is_file():
        raise FileNotFoundError(f"Validated existing_poisson source missing: {path}")
    spec = importlib.util.spec_from_file_location("validated_existing_poisson", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    function = getattr(module, "existing_poisson", None)
    if not callable(function):
        raise AttributeError(f"{path} does not define callable existing_poisson")
    print(f"Using existing_poisson{inspect.signature(function)} from {path}", flush=True)
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
        "base": base, "base_depth": base, "depth": base, "initial": base,
        "initial_depth": base, "aligned": base, "aligned_depth": base,
        "prediction": base, "pred": base, "sparse": sparse,
        "sparse_depth": sparse, "lidar": sparse, "lidar_depth": sparse,
        "metric_depth": sparse, "anchors": anchors, "anchor_mask": anchors,
        "sparse_mask": anchors, "valid_mask": anchors, "rtol": rtol,
        "tol": rtol, "maxiter": maxiter, "max_iter": maxiter,
    }
    signature = inspect.signature(function)
    kwargs: dict[str, Any] = {}
    unknown: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            unknown.append(name)
    result = function(base, sparse, anchors, rtol, maxiter) if unknown else function(**kwargs)
    if isinstance(result, tuple):
        refined, diagnostics = result[0], result[1]
    else:
        refined, diagnostics = result, {}
    refined = np.asarray(refined, dtype=np.float32).squeeze()
    if refined.shape != base.shape:
        raise ValueError(f"Poisson returned {refined.shape}; expected {base.shape}")
    invalid = ~np.isfinite(refined) | (refined <= KITTI_MIN_M)
    repaired = np.where(invalid, base, refined)
    repaired = np.clip(repaired, KITTI_MIN_M, KITTI_MAX_M).astype(np.float32)
    diag = diagnostics if isinstance(diagnostics, dict) else {"value": diagnostics}
    diag = {str(key): json_safe(value) for key, value in diag.items()}
    diag["repaired_pixels"] = int(invalid.sum())
    diag["full_fallback"] = False
    if not np.all(np.isfinite(repaired)) or np.any(repaired <= 0):
        repaired = base.astype(np.float32)
        diag["full_fallback"] = True
    return repaired, diag


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def quantize_prediction(depth: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    source = np.asarray(depth, dtype=np.float64)
    stats = {
        "nonfinite": int(np.sum(~np.isfinite(source))),
        "nonpositive": int(np.sum(np.isfinite(source) & (source <= 0))),
        "above_kitti_max": int(np.sum(np.isfinite(source) & (source > KITTI_MAX_M))),
    }
    safe = np.nan_to_num(source, nan=KITTI_MIN_M, posinf=KITTI_MAX_M, neginf=KITTI_MIN_M)
    safe = np.clip(safe, KITTI_MIN_M, KITTI_MAX_M)
    encoded = np.maximum((safe * KITTI_SCALE).astype(np.uint16), np.uint16(1))
    return encoded.astype(np.float64) / KITTI_SCALE, stats


def metric_values(gt: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(gt) & (gt > 0) & np.isfinite(prediction) & (prediction > 0)
    if not valid.any():
        return {key: math.nan for key in (
            "pixel_count", "rmse_m", "mae_m", "absrel_pct", "bias_m", "delta1_pct",
            "median_depth_ratio", "scale_log_error", "irmse_per_km", "imae_per_km"
        )}
    truth = gt[valid].astype(np.float64); pred = prediction[valid].astype(np.float64)
    error = pred - truth; absolute = np.abs(error)
    ratio = np.maximum(pred / truth, truth / pred)
    median_ratio = float(np.median(pred / truth))
    inverse_error = 1.0 / pred - 1.0 / truth
    return {
        "pixel_count": int(valid.sum()),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "mae_m": float(np.mean(absolute)),
        "absrel_pct": float(100.0 * np.mean(absolute / truth)),
        "bias_m": float(np.mean(error)),
        "delta1_pct": float(100.0 * np.mean(ratio < 1.25)),
        "median_depth_ratio": median_ratio,
        "scale_log_error": float(abs(math.log(max(median_ratio, 1e-12)))),
        "irmse_per_km": float(1000.0 * np.sqrt(np.mean(inverse_error ** 2))),
        "imae_per_km": float(1000.0 * np.mean(np.abs(inverse_error))),
    }


def scene_regions(gt: np.ndarray, sparse: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(gt) & (gt > 0)
    anchors = np.isfinite(sparse) & (sparse > 0)
    return {
        "all_valid": valid,
        "outside_four_beam_pixels": valid & ~anchors,
        "range_0_20m": valid & (gt <= 20),
        "range_20_40m": valid & (gt > 20) & (gt <= 40),
        "range_40_80m": valid & (gt > 40) & (gt <= 80),
        "range_over_80m": valid & (gt > 80),
    }


def load_scene(root: Path, relative_dir: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = row["stem"]
    gt = load_depth_png(root / "groundtruth_depth" / f"{stem}.png")
    sparse = load_npy_2d(root / "sparse_input_m" / f"{stem}.npy")
    relative = load_npy_2d(relative_dir / f"{stem}.npy")
    expected = (int(row["height"]), int(row["width"]))
    if gt.shape != expected or sparse.shape != expected or relative.shape != expected:
        raise ValueError(f"{stem}: GT/sparse/relative shapes {gt.shape}/{sparse.shape}/{relative.shape}, expected {expected}")
    return gt, sparse, relative


def aggregate_metrics(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in key_fields)].append(row)
    output: list[dict[str, Any]] = []
    metric_names = (
        "rmse_m", "mae_m", "absrel_pct", "bias_m", "delta1_pct",
        "median_depth_ratio", "scale_log_error", "irmse_per_km", "imae_per_km",
    )
    for key, selected in grouped.items():
        item = dict(zip(key_fields, key)); item["scene_count"] = len(selected)
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            values = values[np.isfinite(values)]
            item[f"mean_{metric}"] = float(values.mean()) if values.size else math.nan
        output.append(item)
    return output


def command_select(args: argparse.Namespace) -> None:
    root = args.data_root.expanduser().resolve()
    relative_dir = args.relative_dir.expanduser().resolve()
    output = args.selection_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    da3_root = args.da3_root.expanduser().resolve()
    rows = manifest_rows(root, "dev")
    function = load_existing_poisson(da3_root)
    metrics_path = output / "development_per_scene.csv"
    result_rows: list[dict[str, Any]] = read_csv_if_exists(metrics_path) if args.resume else []
    completed = defaultdict(set)
    for result in result_rows:
        completed[result["scene"]].add((result["alignment"], result["variant"]))
    expected = {(alignment, variant) for alignment in ALIGNMENTS for variant in ("aligned", "existing_poisson")}
    diagnostic_path = output / "development_diagnostics.csv"
    diagnostics: list[dict[str, Any]] = read_csv_if_exists(diagnostic_path) if args.resume else []
    for index, row in enumerate(rows, 1):
        stem = row["stem"]
        if expected.issubset(completed[stem]):
            print(f"[{index:4d}/{len(rows)}] {stem} completed", flush=True); continue
        result_rows = [item for item in result_rows if item.get("scene") != stem]
        diagnostics = [item for item in diagnostics if item.get("scene") != stem]
        gt, sparse, relative = load_scene(root, relative_dir, row)
        mask = scene_regions(gt, sparse)["all_valid"]
        anchors = np.isfinite(sparse) & (sparse > 0)
        for alignment in ALIGNMENTS:
            aligned, align_diag = alignment_prediction(alignment, relative, sparse)
            refined, poisson_diag = call_poisson(function, aligned, sparse, anchors, args.rtol, args.maxiter)
            for variant, prediction in (("aligned", aligned), ("existing_poisson", refined)):
                scored, quant_diag = quantize_prediction(prediction)
                result_rows.append({
                    "scene": stem, "group": row["group"], "alignment": alignment,
                    "variant": variant, **metric_values(gt, scored, mask), **quant_diag,
                })
            diagnostics.append({
                "scene": stem, "alignment": alignment,
                "alignment_diagnostics": json.dumps(json_safe(align_diag), sort_keys=True),
                "poisson_diagnostics": json.dumps(json_safe(poisson_diag), sort_keys=True),
            })
        write_csv(metrics_path, result_rows)
        write_csv(diagnostic_path, diagnostics)
        print(f"[{index:4d}/{len(rows)}] {stem} evaluated {len(ALIGNMENTS)} alignments", flush=True)
    summary = aggregate_metrics(result_rows, ("alignment", "variant"))
    summary.sort(key=lambda item: (
        item["variant"] != "existing_poisson", item["mean_rmse_m"],
        item["mean_absrel_pct"], item["mean_scale_log_error"],
    ))
    write_csv(output / "development_summary.csv", summary)
    poisson_rows = [item for item in summary if item["variant"] == "existing_poisson"]
    winner = min(poisson_rows, key=lambda item: (
        item["mean_rmse_m"], item["mean_absrel_pct"], item["mean_scale_log_error"]
    ))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    selection = {
        "selected_alignment": winner["alignment"],
        "selected_pipeline": f"{winner['alignment']} + existing_poisson",
        "selection_split": "development drives only",
        "primary_selection_metric": "full-image per-scene macro RMSE after KITTI quantization",
        "tie_breakers": ["AbsRel", "absolute log median-depth-ratio error"],
        "development_scene_count": winner["scene_count"],
        "winner_development_metrics": winner,
        "line_spec": protocol["line_spec"],
        "source_mode": protocol["source_mode"],
        "alignments_compared": list(ALIGNMENTS),
    }
    (output / "selected_alignment.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("\n===== DEVELOPMENT ALIGNMENT SELECTION =====")
    for rank, item in enumerate(sorted(poisson_rows, key=lambda x: x["mean_rmse_m"]), 1):
        print(
            f"{rank:2d}. {item['alignment']:26s} RMSE={item['mean_rmse_m']:.4f} m  "
            f"AbsRel={item['mean_absrel_pct']:.3f}%  ratio={item['mean_median_depth_ratio']:.4f}"
        )
    print(f"SELECTED AND LOCKED: {selection['selected_pipeline']}")
    print(f"Selection file: {output / 'selected_alignment.json'}")


def valid_cached_prediction(path: Path, shape: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_npy_2d(path)
    except Exception:
        return False
    return value.shape == shape and np.all(np.isfinite(value)) and np.all(value > 0)


def command_predict(args: argparse.Namespace) -> None:
    root = args.data_root.expanduser().resolve()
    relative_dir = args.relative_dir.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve(); prediction_root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection_json.expanduser().resolve().read_text(encoding="utf-8"))
    selected = selection["selected_alignment"]
    if selected not in ALIGNMENTS:
        raise ValueError(f"Unknown selected alignment: {selected}")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if selection["line_spec"] != protocol["line_spec"] or selection["source_mode"] != protocol["source_mode"]:
        raise RuntimeError("Selection file protocol does not match prepared data")
    function = load_existing_poisson(args.da3_root.expanduser().resolve())
    rows = manifest_rows(root, "test")
    diagnostic_path = prediction_root / "prediction_diagnostics.csv"
    diagnostics: list[dict[str, Any]] = read_csv_if_exists(diagnostic_path)
    for index, row in enumerate(rows, 1):
        stem = row["stem"]; shape = (int(row["height"]), int(row["width"]))
        outputs = {method: prediction_root / method / f"{stem}.npy" for method in DA3_METHODS}
        if not args.overwrite and all(valid_cached_prediction(path, shape) for path in outputs.values()):
            print(f"[{index:4d}/{len(rows)}] {stem} cached", flush=True); continue
        _gt, sparse, relative = load_scene(root, relative_dir, row)
        anchors = np.isfinite(sparse) & (sparse > 0)
        median, median_diag = alignment_prediction("median_scale", relative, sparse)
        median_p, median_p_diag = call_poisson(function, median, sparse, anchors, args.rtol, args.maxiter)
        if selected == "median_scale":
            chosen, chosen_diag = median.copy(), dict(median_diag)
            chosen_p, chosen_p_diag = median_p.copy(), dict(median_p_diag)
            chosen_diag["reused_median_control"] = True
            chosen_p_diag["reused_median_control"] = True
        else:
            chosen, chosen_diag = alignment_prediction(selected, relative, sparse)
            chosen_p, chosen_p_diag = call_poisson(function, chosen, sparse, anchors, args.rtol, args.maxiter)
        predictions = {
            "da3_median": median,
            "da3_median_existing_poisson": median_p,
            "da3_selected_alignment": chosen,
            "da3_selected_alignment_existing_poisson": chosen_p,
        }
        for method, prediction in predictions.items():
            outputs[method].parent.mkdir(parents=True, exist_ok=True)
            np.save(outputs[method], prediction.astype(np.float32))
        diagnostics = [item for item in diagnostics if item.get("scene") != stem]
        diagnostics.append({
            "scene": stem, "selected_alignment": selected,
            "median_alignment": json.dumps(json_safe(median_diag), sort_keys=True),
            "median_poisson": json.dumps(json_safe(median_p_diag), sort_keys=True),
            "selected_alignment_diag": json.dumps(json_safe(chosen_diag), sort_keys=True),
            "selected_poisson": json.dumps(json_safe(chosen_p_diag), sort_keys=True),
        })
        write_csv(diagnostic_path, diagnostics)
        print(f"[{index:4d}/{len(rows)}] {stem} saved selected={selected}", flush=True)
    (prediction_root / "frozen_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"\nFrozen DA3 test predictions: {prediction_root}")


def resolve_any2full_prediction(directory: Path, stem: str) -> Path:
    exact = [directory / f"{stem}.npy", directory / f"{stem}.png"]
    for path in exact:
        if path.is_file():
            return path
    matches = [
        path for path in sorted(directory.glob(f"{stem}*"))
        if path.suffix.lower() in (".npy", ".png") and not path.stem.endswith("_rel")
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve Any2Full metric prediction for {stem}: {matches}")


def load_metric_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return load_npy_2d(path)
    return load_depth_png(path)


def cluster_bootstrap(
    differences: dict[str, list[float]], samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    groups = sorted(differences)
    if not groups:
        return math.nan, math.nan
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        values = [value for group_index in chosen for value in differences[groups[group_index]]]
        boot[index] = np.mean(values)
    return tuple(map(float, np.quantile(boot, (0.025, 0.975))))


def paired_comparisons(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> list[dict[str, Any]]:
    primary = [row for row in rows if row["region"] == "all_valid"]
    by_key = {(row["scene"], row["method"]): row for row in primary}
    scene_group = {row["scene"]: row["group"] for row in primary}
    metrics = ("rmse_m", "absrel_pct", "mae_m", "bias_m", "delta1_pct", "scale_log_error")
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    scenes = sorted({row["scene"] for row in primary})
    contrasts = [
        (method, "any2full_vits", "DA3_vs_Any2Full") for method in DA3_METHODS
    ] + [
        (
            "da3_selected_alignment",
            "da3_median",
            "selected_alignment_vs_median_before_poisson",
        ),
        (
            "da3_selected_alignment_existing_poisson",
            "da3_median_existing_poisson",
            "selected_alignment_vs_median_after_poisson",
        ),
    ]
    for first_method, second_method, contrast in contrasts:
        for metric in metrics:
            grouped: dict[str, list[float]] = defaultdict(list)
            values: list[float] = []
            for scene in scenes:
                first = by_key.get((scene, first_method))
                second = by_key.get((scene, second_method))
                if first is None or second is None:
                    continue
                difference = float(first[metric]) - float(second[metric])
                if metric == "bias_m":
                    difference = abs(float(first[metric])) - abs(float(second[metric]))
                if np.isfinite(difference):
                    values.append(difference); grouped[scene_group[scene]].append(difference)
            low, high = cluster_bootstrap(grouped, samples, rng)
            lower_is_better = metric != "delta1_pct"
            first_better = (high < 0) if lower_is_better else (low > 0)
            second_better = (low > 0) if lower_is_better else (high < 0)
            if first_better:
                decision = "first better"
            elif second_better:
                decision = "second better"
            else:
                decision = "inconclusive"
            output.append({
                "contrast": contrast,
                "first_method": first_method,
                "first_label": METHOD_LABELS[first_method],
                "second_method": second_method,
                "second_label": METHOD_LABELS[second_method],
                "metric": "absolute_bias_m" if metric == "bias_m" else metric,
                "test_scene_count": len(values), "drive_group_count": len(grouped),
                "mean_first_minus_second": float(np.mean(values)),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high, "decision": decision,
            })
    return output


def markdown_report(
    summary: list[dict[str, Any]], paired: list[dict[str, Any]], selection: dict[str, Any], protocol: dict[str, Any]
) -> str:
    overall = [row for row in summary if row["region"] == "all_valid"]
    overall.sort(key=lambda row: row["mean_rmse_m"])
    lines = [
        "# KITTI log-depth-coverage four-beam dense metric-depth benchmark", "",
        f"- Angular row IDs: `{protocol['line_spec']}`",
        f"- Target elevation bands: `{protocol.get('target_elevation_angles_deg', 'legacy custom rows')}` degrees",
        f"- Four-beam source: `{protocol['source_mode']}`", "- Input splat radius: `0`",
        "- Alignment selected on drive-disjoint development scenes; final numbers use locked test drives only.",
        "- Dense ground truth was not used to generate sparse inputs or fit any test-frame alignment.",
        "- Every test scene has equal weight; predictions use KITTI uint16-equivalent quantization.",
        f"- Frozen selected alignment: `{selection['selected_alignment']}`", "",
        "## Primary full-image test averages", "",
        "| Rank | Method | Scenes | RMSE m ↓ | AbsRel % ↓ | MAE m ↓ | Bias m →0 | δ1 % ↑ | Median ratio →1 | iRMSE 1/km ↓ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if selection["selected_alignment"] == "median_scale":
        lines.insert(
            9,
            "- The selected alignment is median scale, so the two selected-alignment rows duplicate the median carryover controls by construction.",
        )
    for rank, row in enumerate(overall, 1):
        lines.append(
            f"| {rank} | {row['method_label']} | {row['scene_count']} | {row['mean_rmse_m']:.4f} | "
            f"{row['mean_absrel_pct']:.3f} | {row['mean_mae_m']:.4f} | {row['mean_bias_m']:+.4f} | "
            f"{row['mean_delta1_pct']:.2f} | {row['mean_median_depth_ratio']:.4f} | {row['mean_irmse_per_km']:.3f} |"
        )
    lines.extend(["", "## Distance-band diagnostics", ""])
    for region in ("range_0_20m", "range_20_40m", "range_40_80m", "range_over_80m"):
        selected = sorted([row for row in summary if row["region"] == region], key=lambda row: row["mean_rmse_m"])
        if not selected or not np.isfinite(selected[0]["mean_rmse_m"]):
            continue
        lines.append(f"### {region.replace('_', ' ')}")
        lines.append("")
        lines.append("| Method | RMSE m ↓ | AbsRel % ↓ | δ1 % ↑ |")
        lines.append("|---|---:|---:|---:|")
        for row in selected:
            lines.append(f"| {row['method_label']} | {row['mean_rmse_m']:.4f} | {row['mean_absrel_pct']:.3f} | {row['mean_delta1_pct']:.2f} |")
        lines.append("")
    lines.extend([
        "## Paired drive-cluster bootstrap decisions", "",
        "Difference is first method minus second method. For error metrics, negative favors the first; for δ1, positive favors the first.", "",
        "| Contrast | Metric | Mean Δ | 95% CI | Decision |", "|---|---|---:|---:|---|",
    ])
    for row in paired:
        lines.append(
            f"| {row['first_label']} vs {row['second_label']} | {row['metric']} | "
            f"{row['mean_first_minus_second']:+.5f} | "
            f"[{row['bootstrap_ci_low']:+.5f}, {row['bootstrap_ci_high']:+.5f}] | {row['decision']} |"
        )
    lines.extend([
        "", "## Interpretation guardrails", "",
        "- The headline winner is the lowest locked-test macro RMSE; metric-wise winners may differ.",
        "- The `{7,12,22,37}` placement is a physics-informed log-depth-coverage hypothesis, not a mathematically proven optimum. It must be judged by held-out depth-bin occupancy and metric-depth errors.",
        "- The earlier `{7,8,9,10}` placement was optimized in prior work for KITTI car detection, not for broad depth-range anchor coverage, and is retained only as a separate control.",
        "- A projected-source run is based on real 64-line Velodyne measurements but cannot recover rare selected-beam returns already lost in projection collisions. Repeat with raw `.bin` input as a source-sensitivity check before claiming exact reproduction.",
        "- Any2Full paper numbers are context only unless checkpoint, split, beam extraction, resizing, mask, and aggregation all match this run.",
    ])
    return "\n".join(lines) + "\n"


def command_evaluate(args: argparse.Namespace) -> None:
    root = args.data_root.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve()
    any2full_dir = args.any2full_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection_json.expanduser().resolve().read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    test = manifest_rows(root, "test")
    for index, item in enumerate(test, 1):
        stem = item["stem"]; shape = (int(item["height"]), int(item["width"]))
        gt = load_depth_png(root / "groundtruth_depth" / f"{stem}.png")
        sparse = load_npy_2d(root / "sparse_input_m" / f"{stem}.npy")
        predictions = {
            method: load_npy_2d(prediction_root / method / f"{stem}.npy") for method in DA3_METHODS
        }
        predictions["any2full_vits"] = load_metric_prediction(resolve_any2full_prediction(any2full_dir, stem))
        for method, prediction in predictions.items():
            if prediction.shape != shape:
                raise ValueError(f"{stem}: {method} shape {prediction.shape}; expected {shape}. Resizing is disabled for fair scoring.")
            scored, quant_diag = quantize_prediction(prediction)
            for region, mask in scene_regions(gt, sparse).items():
                rows.append({
                    "scene": stem, "group": item["group"], "method": method,
                    "method_label": METHOD_LABELS[method], "region": region,
                    **metric_values(gt, scored, mask), **quant_diag,
                })
        print(f"[{index:4d}/{len(test)}] {stem} evaluated", flush=True)
    write_csv(output / "locked_test_per_scene.csv", rows)
    summary = aggregate_metrics(rows, ("method", "method_label", "region"))
    summary.sort(key=lambda row: (row["region"], row["mean_rmse_m"]))
    write_csv(output / "locked_test_summary.csv", summary)
    paired = paired_comparisons(rows, args.bootstrap_samples, args.seed)
    write_csv(output / "paired_drive_bootstrap.csv", paired)
    report = markdown_report(summary, paired, selection, protocol)
    (output / "comparison_report.md").write_text(report, encoding="utf-8")
    overall = sorted([row for row in summary if row["region"] == "all_valid"], key=lambda row: row["mean_rmse_m"])
    print("\n===== LOCKED KITTI FOUR-BEAM FULL-IMAGE AVERAGES =====")
    print(f"Angular row IDs: {protocol['line_spec']}  Selected alignment: {selection['selected_alignment']}")
    print(f"{'Method':52s} {'RMSE':>9s} {'AbsRel':>10s} {'MAE':>9s} {'bias':>9s} {'delta1':>9s} {'ratio':>9s}")
    for row in overall:
        print(
            f"{row['method_label'][:52]:52s} {row['mean_rmse_m']:8.4f}m "
            f"{row['mean_absrel_pct']:9.3f}% {row['mean_mae_m']:8.4f}m "
            f"{row['mean_bias_m']:+8.4f}m {row['mean_delta1_pct']:8.2f}% "
            f"{row['mean_median_depth_ratio']:9.4f}"
        )
    print(f"\nPrimary RMSE winner: {overall[0]['method_label']}")
    print(f"Report: {output / 'comparison_report.md'}")


def command_self_test(args: argparse.Namespace) -> None:
    row_centres = 2.0 - (np.asarray(DEFAULT_LINE_SPEC, dtype=np.float64) + 0.5) * 0.4
    assert np.allclose(row_centres, np.asarray(DEFAULT_TARGET_ANGLES_DEG))
    rng = np.random.default_rng(3)
    relative = np.linspace(1.0, 8.0, 200, dtype=np.float32).reshape(10, 20)
    sparse = np.zeros_like(relative)
    mask = np.zeros_like(relative, dtype=bool); mask[2, ::2] = True; mask[6, 1::2] = True
    sparse[mask] = 2.5 * relative[mask] + rng.normal(0, 0.01, mask.sum())
    for name in ALIGNMENTS:
        prediction, diagnostics = alignment_prediction(name, relative, sparse)
        assert prediction.shape == relative.shape and np.all(np.isfinite(prediction)) and np.all(prediction > 0)
        if args.verbose:
            print(name, diagnostics)
    points = np.array([[10.0, 0.0, -0.2], [10.0, 0.0, -0.3], [10.0, 0.0, -0.4]])
    line, column = official_angular_ids(points, 1024)
    assert line.shape == (3,) and column.shape == (3,)
    depth = zbuffer_depth(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 3.0]), (3, 3))
    assert depth[1, 1] == 3.0
    print("SELF-TEST PASSED")


def main() -> None:
    args = build_parser().parse_args()
    commands = {
        "prepare": command_prepare,
        "audit": command_audit,
        "infer-da3": command_infer_da3,
        "select": command_select,
        "predict": command_predict,
        "evaluate": command_evaluate,
        "self-test": command_self_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
