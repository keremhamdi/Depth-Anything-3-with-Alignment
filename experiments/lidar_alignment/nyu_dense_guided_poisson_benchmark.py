#!/usr/bin/env python3
"""Controlled NYU one-line LiDAR density × Poisson benchmark.

This companion script intentionally leaves the validated
``nyu_1line_metric_benchmark.py`` unchanged.  It provides three commands:

``prepare-dense``
    Densify only the *angular sampling geometry* of an existing empirical
    one-line dataset.  New ranges are sampled from NYU dense ground truth;
    low-density ranges are never interpolated or copied.

``guided-poisson``
    Refine a median-aligned DA3 metric prior with a positive-by-construction,
    screened log-depth Poisson solve.  Pairwise conductance is reduced at both
    RGB and DA3 depth edges.

``evaluate-factorial``
    Report equal-scene-weight full-image and outside-support metrics for the
    complete 2 × 2 design: empirical/dense points × existing/guided Poisson.
    It writes tables and paired bootstrap confidence intervals, never one plot
    per image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import LinearOperator, cg


GT_DIR_CANDIDATES = (
    "gt_m",
    "ground_truth_m",
    "dense_gt_m",
    "depth_gt_m",
    "gt_depth_m",
)
SPARSE_DIR_CANDIDATES = ("sparse_exact_m", "sparse_input_m")
METRIC_KEYS = (
    "rmse_m",
    "absrel_pct",
    "mae_m",
    "bias_m",
    "abs_bias_m",
    "delta1_pct",
    "median_depth_ratio",
    "scale_log_error",
)


def positive_map(path: Path) -> np.ndarray:
    value = np.squeeze(np.load(path)).astype(np.float32)
    if value.ndim != 2:
        raise ValueError(f"Expected a 2-D map in {path}, got {value.shape}")
    return value


def locate_subdir(root: Path, candidates: Iterable[str], explicit: Path | None = None) -> Path:
    if explicit is not None:
        result = explicit.expanduser().resolve()
        if not result.is_dir():
            raise FileNotFoundError(result)
        return result
    for name in candidates:
        result = root / name
        if result.is_dir():
            return result
    raise FileNotFoundError(f"None of {tuple(candidates)} exists below {root}")


def prediction_path(directory: Path, stem: str) -> Path:
    exact = directory / f"{stem}.npy"
    if exact.is_file():
        return exact
    matches = sorted(directory.glob(f"{stem}*.npy"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve {stem} in {directory}: {matches}")


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32, copy=False)
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def hardlink_or_copy(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def mirror_static_dataset(source: Path, destination: Path, overwrite: bool) -> None:
    """Mirror non-sparse data with hard links when possible."""
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.name == "protocol.json" or child.name.startswith("sparse"):
            continue
        if child.is_file():
            hardlink_or_copy(child, destination / child.name, overwrite)
        elif child.is_dir():
            for item in child.rglob("*"):
                if item.is_file():
                    relative = item.relative_to(source)
                    hardlink_or_copy(item, destination / relative, overwrite)


def grouped_scan_curve(sparse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.where(np.isfinite(sparse) & (sparse > 0))
    if len(x) < 2:
        raise RuntimeError(f"Only {len(x)} empirical scan points")
    unique_x = np.unique(x)
    median_y = np.array([np.median(y[x == value]) for value in unique_x])
    return unique_x.astype(np.float64), median_y.astype(np.float64)


def angular_centers(
    count: int,
    fov_degrees: float,
    u_min: int,
    u_max: int,
) -> np.ndarray:
    """Project equal-angle ray centers into the empirical image support."""
    if count < 2:
        raise ValueError("--points must be at least 2")
    if not 0 < fov_degrees < 180:
        raise ValueError("--horizontal-fov-deg must be in (0, 180)")
    if u_max - u_min + 1 < count:
        raise ValueError(
            f"Cannot place {count} unique rays in [{u_min}, {u_max}]"
        )
    half = math.radians(fov_degrees / 2)
    step = 2 * half / count
    theta = -half + (np.arange(count, dtype=np.float64) + 0.5) * step
    normalized = np.tan(theta) / math.tan(half)
    center = 0.5 * (u_min + u_max)
    span = 0.5 * (u_max - u_min)
    u = np.rint(center + span * normalized).astype(np.int32)
    if len(np.unique(u)) != count:
        # This is only a pixel-quantization fallback.  It never changes depth.
        u = np.rint(np.linspace(u_min, u_max, count)).astype(np.int32)
    if len(np.unique(u)) != count:
        raise RuntimeError("Could not create unique projected ray centers")
    return u


def sample_dense_scan(
    gt: np.ndarray,
    empirical_sparse: np.ndarray,
    points: int,
    fov_degrees: float,
    max_depth_m: float,
) -> np.ndarray:
    source_u, source_v = grouped_scan_curve(empirical_sparse)
    u = angular_centers(points, fov_degrees, int(source_u.min()), int(source_u.max()))
    v = np.rint(np.interp(u, source_u, source_v)).astype(np.int32)
    v = np.clip(v, 0, gt.shape[0] - 1)
    u = np.clip(u, 0, gt.shape[1] - 1)
    dense = np.zeros_like(gt, dtype=np.float32)
    values = gt[v, u]
    valid = np.isfinite(values) & (values > 0) & (values <= max_depth_m)
    dense[v[valid], u[valid]] = values[valid]
    return dense


def prepare_dense(args: argparse.Namespace) -> None:
    source = args.source_data_root.expanduser().resolve()
    output = args.output_data_root.expanduser().resolve()
    if source == output:
        raise ValueError("Source and output data roots must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; pass --overwrite to resume/replace")

    source_sparse = locate_subdir(source, SPARSE_DIR_CANDIDATES, args.source_sparse_dir)
    gt_dir = locate_subdir(source, GT_DIR_CANDIDATES, args.gt_dir)
    mirror_static_dataset(source, output, args.overwrite)
    exact_out = output / "sparse_exact_m"
    input_out = output / "sparse_input_m"
    exact_out.mkdir(parents=True, exist_ok=True)
    input_out.mkdir(parents=True, exist_ok=True)

    sparse_files = sorted(source_sparse.glob("*.npy"))
    if args.limit is not None:
        sparse_files = sparse_files[: args.limit]
    if not sparse_files:
        raise RuntimeError(f"No NPY files in {source_sparse}")

    rows: list[dict] = []
    counts: list[int] = []
    for index, sparse_path in enumerate(sparse_files, 1):
        stem = sparse_path.stem
        gt_path = prediction_path(gt_dir, stem)
        gt = positive_map(gt_path)
        empirical = resize_depth(positive_map(sparse_path), gt.shape)
        dense = sample_dense_scan(
            gt,
            empirical,
            args.points,
            args.horizontal_fov_deg,
            args.max_depth_m,
        )
        count = int(np.count_nonzero(dense > 0))
        if count < args.minimum_valid_points:
            raise RuntimeError(
                f"{stem}: only {count}/{args.points} valid dense scan returns"
            )
        np.save(exact_out / f"{stem}.npy", dense)
        np.save(input_out / f"{stem}.npy", dense)
        counts.append(count)
        rows.append(
            {
                "stem": stem,
                "source_empirical_points": int(np.count_nonzero(empirical > 0)),
                "requested_rays": args.points,
                "valid_returns": count,
                "horizontal_fov_deg": args.horizontal_fov_deg,
                "nominal_angular_step_deg": args.horizontal_fov_deg / args.points,
            }
        )
        print(
            f"[{index:3d}/{len(sparse_files)}] {stem}: "
            f"empirical={rows[-1]['source_empirical_points']} dense={count}",
            flush=True,
        )

    write_csv(output / "dense_scan_manifest.csv", rows)
    source_protocol = {}
    protocol_path = source / "protocol.json"
    if protocol_path.is_file():
        source_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol = {
        **source_protocol,
        "benchmark_extension": "commercial_2d_lidar_factorial_v1",
        "line_protocol": "commercial_05deg_from_empirical_scan_geometry",
        "independent_line_measurements_requested": args.points,
        "horizontal_fov_degrees": args.horizontal_fov_deg,
        "nominal_angular_step_degrees": args.horizontal_fov_deg / args.points,
        "splat_radius_pixels": 0,
        "one_pixel_per_independent_return": True,
        "source_empirical_data_root": str(source),
        "geometry_rule": (
            "equal-angle ray centers projected within each empirical scan's image support; "
            "scan-row geometry is interpolated, never range values"
        ),
        "range_rule": (
            "each new ray reads its own NYU dense-GT metric depth; no empirical LiDAR "
            "depth is copied or interpolated"
        ),
        "dense_gt_use": "construct sparse input and evaluate only; never post-inference alignment",
        "max_valid_depth_m": args.max_depth_m,
        "scientific_label": (
            "idealized denser commercial 2D single-line scan; not a claim about every LiDAR"
        ),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    count_array = np.asarray(counts)
    print(
        f"Prepared {len(rows)} dense one-line scenes at {output}\n"
        f"Valid independent returns: min={count_array.min()} "
        f"median={np.median(count_array):.0f} max={count_array.max()}"
    )


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8/float sRGB to CIE Lab (D65) without extra dependencies."""
    value = np.asarray(rgb, dtype=np.float64)
    if value.max(initial=0) > 1.0:
        value /= 255.0
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    epsilon = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lab = np.empty_like(xyz)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab.astype(np.float32)


def guided_weights(
    rgb: np.ndarray,
    log_prior: np.ndarray,
    color_sigma: float,
    depth_sigma: float,
    weight_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    lab = srgb_to_lab(rgb)
    color_h = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=2)
    color_v = np.linalg.norm(lab[1:] - lab[:-1], axis=2)
    depth_h = np.abs(log_prior[:, 1:] - log_prior[:, :-1])
    depth_v = np.abs(log_prior[1:] - log_prior[:-1])
    weight_h = np.exp(
        -0.5 * (color_h / color_sigma) ** 2
        -0.5 * (depth_h / depth_sigma) ** 2
    )
    weight_v = np.exp(
        -0.5 * (color_v / color_sigma) ** 2
        -0.5 * (depth_v / depth_sigma) ** 2
    )
    weight_h = weight_floor + (1 - weight_floor) * weight_h
    weight_v = weight_floor + (1 - weight_floor) * weight_v
    return weight_h.astype(np.float64), weight_v.astype(np.float64)


def weighted_laplacian(
    value: np.ndarray,
    weight_h: np.ndarray,
    weight_v: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(value, dtype=np.float64)
    difference_h = value[:, :-1] - value[:, 1:]
    result[:, :-1] += weight_h * difference_h
    result[:, 1:] -= weight_h * difference_h
    difference_v = value[:-1] - value[1:]
    result[:-1] += weight_v * difference_v
    result[1:] -= weight_v * difference_v
    return result


def rgb_da3_guided_log_poisson(
    prior_m: np.ndarray,
    sparse_m: np.ndarray,
    rgb: np.ndarray,
    screen_weight: float,
    anchor_weight: float,
    color_sigma: float,
    depth_sigma: float,
    weight_floor: float,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict]:
    valid_prior = np.isfinite(prior_m) & (prior_m > 0)
    if not np.all(valid_prior):
        raise ValueError("Median-aligned DA3 prior contains invalid depth")
    anchors = np.isfinite(sparse_m) & (sparse_m > 0)
    if int(anchors.sum()) < 8:
        raise RuntimeError(f"Only {int(anchors.sum())} valid anchors")
    if screen_weight <= 0 or anchor_weight <= 0:
        raise ValueError("Screen and anchor weights must be positive")
    if color_sigma <= 0 or depth_sigma <= 0:
        raise ValueError("Guidance sigmas must be positive")
    if not 0 < weight_floor <= 1:
        raise ValueError("--weight-floor must be in (0, 1]")

    log_prior = np.log(prior_m.astype(np.float64))
    log_sparse = np.zeros_like(log_prior)
    log_sparse[anchors] = np.log(sparse_m[anchors].astype(np.float64))
    weight_h, weight_v = guided_weights(
        rgb, log_prior, color_sigma, depth_sigma, weight_floor
    )
    diagonal = np.full(log_prior.shape, screen_weight, dtype=np.float64)
    diagonal[:, :-1] += weight_h
    diagonal[:, 1:] += weight_h
    diagonal[:-1] += weight_v
    diagonal[1:] += weight_v
    diagonal[anchors] += anchor_weight

    def matvec(flat: np.ndarray) -> np.ndarray:
        value = flat.reshape(log_prior.shape)
        result = weighted_laplacian(value, weight_h, weight_v)
        result += screen_weight * value
        result[anchors] += anchor_weight * value[anchors]
        return result.ravel()

    rhs = weighted_laplacian(log_prior, weight_h, weight_v)
    rhs += screen_weight * log_prior
    rhs[anchors] += anchor_weight * log_sparse[anchors]
    size = log_prior.size
    operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator(
        (size, size), matvec=lambda flat: flat / diagonal.ravel(), dtype=np.float64
    )
    try:
        solution, info = cg(
            operator,
            rhs.ravel(),
            x0=log_prior.ravel(),
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            M=preconditioner,
        )
    except TypeError:  # scipy < 1.12
        solution, info = cg(
            operator,
            rhs.ravel(),
            x0=log_prior.ravel(),
            tol=rtol,
            maxiter=maxiter,
            M=preconditioner,
        )
    if info < 0:
        raise RuntimeError(f"Guided Poisson CG failed with info={info}")
    log_result = solution.reshape(log_prior.shape)
    result = np.exp(np.clip(log_result, -20, 20)).astype(np.float32)
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise RuntimeError("Guided Poisson produced invalid metric depth")
    residual = matvec(solution) - rhs.ravel()
    diagnostics = {
        "anchors": int(anchors.sum()),
        "cg_info": int(info),
        "relative_linear_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs.ravel()), 1e-12)
        ),
        "anchor_mae_before_m": float(np.mean(np.abs(prior_m[anchors] - sparse_m[anchors]))),
        "anchor_mae_after_m": float(np.mean(np.abs(result[anchors] - sparse_m[anchors]))),
        "horizontal_weight_mean": float(weight_h.mean()),
        "vertical_weight_mean": float(weight_v.mean()),
        "min_depth_m": float(result.min()),
        "max_depth_m": float(result.max()),
    }
    return result, diagnostics


def guided_poisson(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve()
    median_dir = args.median_dir.expanduser().resolve()
    if (median_dir / "metric_m").is_dir():
        median_dir = median_dir / "metric_m"
    output = args.output_dir.expanduser().resolve()
    rgb_dir = locate_subdir(data_root, ("rgb",), args.rgb_dir)
    sparse_dir = locate_subdir(data_root, SPARSE_DIR_CANDIDATES, args.sparse_dir)
    output.mkdir(parents=True, exist_ok=True)

    sparse_files = sorted(sparse_dir.glob("*.npy"))
    if args.limit is not None:
        sparse_files = sparse_files[: args.limit]
    if not sparse_files:
        raise RuntimeError(f"No sparse inputs in {sparse_dir}")
    diagnostics: list[dict] = []
    for index, sparse_path in enumerate(sparse_files, 1):
        stem = sparse_path.stem
        destination = output / f"{stem}.npy"
        if destination.is_file() and not args.overwrite:
            print(f"[{index:3d}/{len(sparse_files)}] SKIP {stem}")
            continue
        sparse = positive_map(sparse_path)
        prior = resize_depth(positive_map(prediction_path(median_dir, stem)), sparse.shape)
        rgb_path = None
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = rgb_dir / f"{stem}{suffix}"
            if candidate.is_file():
                rgb_path = candidate
                break
        if rgb_path is None:
            raise FileNotFoundError(f"RGB for {stem} in {rgb_dir}")
        rgb_image = Image.open(rgb_path).convert("RGB")
        if rgb_image.size != (sparse.shape[1], sparse.shape[0]):
            rgb_image = rgb_image.resize(
                (sparse.shape[1], sparse.shape[0]), Image.Resampling.BILINEAR
            )
        rgb = np.asarray(rgb_image)
        result, row = rgb_da3_guided_log_poisson(
            prior,
            sparse,
            rgb,
            args.screen_weight,
            args.anchor_weight,
            args.color_sigma_lab,
            args.depth_sigma_log,
            args.weight_floor,
            args.rtol,
            args.maxiter,
        )
        np.save(destination, result)
        diagnostics.append({"stem": stem, **row})
        print(
            f"[{index:3d}/{len(sparse_files)}] {stem}: anchors={row['anchors']} "
            f"anchor MAE {row['anchor_mae_before_m']:.3f}->{row['anchor_mae_after_m']:.3f}m "
            f"residual={row['relative_linear_residual']:.2e}",
            flush=True,
        )
    if diagnostics:
        write_csv(output / "guided_poisson_diagnostics.csv", diagnostics)
    config = {
        "method": "DA3-SMALL + median + RGB/DA3-guided screened log-Poisson",
        "screen_weight": args.screen_weight,
        "anchor_weight": args.anchor_weight,
        "color_sigma_lab": args.color_sigma_lab,
        "depth_sigma_log": args.depth_sigma_log,
        "weight_floor": args.weight_floor,
        "rtol": args.rtol,
        "maxiter": args.maxiter,
        "parameter_policy": "lock on development data before evaluating the 654-image test set",
    }
    (output / "guided_poisson_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Guided Poisson predictions: {output}")


def compute_metrics(prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & np.isfinite(prediction) & (prediction > 0)
    pred = prediction[valid].astype(np.float64)
    target = gt[valid].astype(np.float64)
    if not len(pred):
        raise RuntimeError("No valid evaluation pixels")
    error = pred - target
    absolute = np.abs(error)
    ratio = np.maximum(pred / target, target / pred)
    median_ratio = float(np.median(pred) / np.median(target))
    bias = float(np.mean(error))
    return {
        "valid_pixels": int(len(pred)),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "absrel_pct": float(100 * np.mean(absolute / target)),
        "mae_m": float(np.mean(absolute)),
        "bias_m": bias,
        "abs_bias_m": abs(bias),
        "delta1_pct": float(100 * np.mean(ratio < 1.25)),
        "median_depth_ratio": median_ratio,
        "scale_log_error": abs(math.log(max(median_ratio, 1e-12))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def method_directory(root: Path, child: str) -> Path:
    root = root.expanduser().resolve()
    candidate = root / child
    if candidate.is_dir():
        return candidate
    if root.is_dir() and root.name == child:
        return root
    raise FileNotFoundError(f"Expected {child} below {root}")


def bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    means = np.empty(samples, dtype=np.float64)
    chunk = 250
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_values(rows: list[dict], first: str, second: str, scope: str, metric: str) -> np.ndarray:
    lookup = {
        (row["method"], row["scope"], row["stem"]): float(row[metric])
        for row in rows
    }
    stems_first = {
        stem for method, row_scope, stem in lookup if method == first and row_scope == scope
    }
    stems_second = {
        stem for method, row_scope, stem in lookup if method == second and row_scope == scope
    }
    stems = sorted(stems_first & stems_second)
    return np.array(
        [lookup[(first, scope, stem)] - lookup[(second, scope, stem)] for stem in stems],
        dtype=np.float64,
    )


def evaluate_factorial(args: argparse.Namespace) -> None:
    low_data = args.low_data_root.expanduser().resolve()
    dense_data = args.dense_data_root.expanduser().resolve()
    gt_dir = locate_subdir(low_data, GT_DIR_CANDIDATES, args.gt_dir)
    low_sparse = locate_subdir(low_data, SPARSE_DIR_CANDIDATES)
    dense_sparse = locate_subdir(dense_data, SPARSE_DIR_CANDIDATES)

    method_specs = [
        ("low_median", "Empirical points — DA3 + median", method_directory(args.low_da3_root, "metric_m"), low_data, low_sparse),
        ("low_existing", "Empirical points — DA3 + median + existing Poisson", method_directory(args.low_da3_root, "metric_m_poisson"), low_data, low_sparse),
        ("low_guided", "Empirical points — DA3 + median + guided log-Poisson", args.low_guided_dir.expanduser().resolve(), low_data, low_sparse),
        ("dense_median", "Dense points — DA3 + median", method_directory(args.dense_da3_root, "metric_m"), dense_data, dense_sparse),
        ("dense_existing", "Dense points — DA3 + median + existing Poisson", method_directory(args.dense_da3_root, "metric_m_poisson"), dense_data, dense_sparse),
        ("dense_guided", "Dense points — DA3 + median + guided log-Poisson", args.dense_guided_dir.expanduser().resolve(), dense_data, dense_sparse),
    ]
    if args.low_any2full_dir is not None:
        method_specs.append(
            ("low_any2full", "Empirical points — Any2Full", args.low_any2full_dir.expanduser().resolve(), low_data, low_sparse)
        )
    if args.dense_any2full_dir is not None:
        method_specs.append(
            ("dense_any2full", "Dense points — Any2Full", args.dense_any2full_dir.expanduser().resolve(), dense_data, dense_sparse)
        )
    labels = {key: label for key, label, *_ in method_specs}
    gt_files = sorted(gt_dir.glob("*.npy"))
    if args.limit is not None:
        gt_files = gt_files[: args.limit]
    if args.expected_images > 0 and len(gt_files) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} GT images, found {len(gt_files)} in {gt_dir}"
        )

    per_scene: list[dict] = []
    for index, gt_path in enumerate(gt_files, 1):
        stem = gt_path.stem
        gt = positive_map(gt_path)
        full_mask = np.isfinite(gt) & (gt > 0) & (gt <= args.max_depth_m)
        outside_by_sparse: dict[Path, np.ndarray] = {}
        for sparse_dir in (low_sparse, dense_sparse):
            sparse = resize_depth(positive_map(prediction_path(sparse_dir, stem)), gt.shape)
            support = np.isfinite(sparse) & (sparse > 0)
            outside_by_sparse[sparse_dir] = (
                full_mask & (distance_transform_edt(~support) > args.outside_margin_px)
            )
        for method, _label, directory, _data, sparse_dir in method_specs:
            prediction = resize_depth(positive_map(prediction_path(directory, stem)), gt.shape)
            for scope, mask in (
                ("full_image", full_mask),
                ("outside_support", outside_by_sparse[sparse_dir]),
            ):
                per_scene.append(
                    {
                        "method": method,
                        "scope": scope,
                        "stem": stem,
                        **compute_metrics(prediction, gt, mask),
                    }
                )
        print(f"[{index:3d}/{len(gt_files)}] {stem}", flush=True)

    summary: list[dict] = []
    for method, _label, *_ in method_specs:
        for scope in ("full_image", "outside_support"):
            selected = [
                row for row in per_scene if row["method"] == method and row["scope"] == scope
            ]
            result = {
                "method": method,
                "label": labels[method],
                "scope": scope,
                "scenes": len(selected),
                "valid_pixels_total": sum(int(row["valid_pixels"]) for row in selected),
            }
            for metric in METRIC_KEYS:
                result[f"mean_{metric}"] = float(np.mean([float(row[metric]) for row in selected]))
            summary.append(result)

    pairs = [
        ("low_guided", "low_existing", "guided_vs_existing_at_empirical_density"),
        ("dense_guided", "dense_existing", "guided_vs_existing_at_dense_density"),
        ("dense_median", "low_median", "dense_vs_empirical_for_median"),
        ("dense_existing", "low_existing", "dense_vs_empirical_for_existing_poisson"),
        ("dense_guided", "low_guided", "dense_vs_empirical_for_guided_poisson"),
    ]
    if args.low_any2full_dir is not None and args.dense_any2full_dir is not None:
        pairs.append(("dense_any2full", "low_any2full", "dense_vs_empirical_for_any2full"))
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap: list[dict] = []
    contrast_metrics = (
        "rmse_m",
        "absrel_pct",
        "mae_m",
        "abs_bias_m",
        "delta1_pct",
        "scale_log_error",
    )
    for scope in ("full_image", "outside_support"):
        for first, second, contrast in pairs:
            for metric in contrast_metrics:
                values = paired_values(per_scene, first, second, scope, metric)
                low, high = bootstrap_ci(values, args.bootstrap_samples, rng)
                higher_better = metric == "delta1_pct"
                if higher_better:
                    decision = "first better" if low > 0 else "second better" if high < 0 else "inconclusive"
                else:
                    decision = "first better" if high < 0 else "second better" if low > 0 else "inconclusive"
                bootstrap.append(
                    {
                        "scope": scope,
                        "contrast": contrast,
                        "first": first,
                        "second": second,
                        "metric": metric,
                        "paired_scenes": len(values),
                        "mean_first_minus_second": float(values.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "decision": decision,
                    }
                )

    # Difference-in-differences isolates whether the guided solver's effect changes with density.
    for scope in ("full_image", "outside_support"):
        for metric in contrast_metrics:
            dense_effect = paired_values(per_scene, "dense_guided", "dense_existing", scope, metric)
            low_effect = paired_values(per_scene, "low_guided", "low_existing", scope, metric)
            interaction = dense_effect - low_effect
            low, high = bootstrap_ci(interaction, args.bootstrap_samples, rng)
            bootstrap.append(
                {
                    "scope": scope,
                    "contrast": "interaction_(guided-existing)_dense_minus_empirical",
                    "first": "dense_guided-dense_existing",
                    "second": "low_guided-low_existing",
                    "metric": metric,
                    "paired_scenes": len(interaction),
                    "mean_first_minus_second": float(interaction.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "decision": "interaction present" if high < 0 or low > 0 else "inconclusive",
                }
            )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_scene_metrics.csv", per_scene)
    write_csv(output / "summary_equal_scene_weight.csv", summary)
    write_csv(output / "paired_bootstrap.csv", bootstrap)
    report = render_report(summary, bootstrap, labels, args)
    (output / "comparison_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Report: {output / 'comparison_report.md'}")


def render_report(
    summary: list[dict],
    bootstrap: list[dict],
    labels: dict[str, str],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# NYU one-line LiDAR density × Poisson factorial benchmark",
        "",
        "Every scene has equal weight. Dense NYU ground truth is used only to create simulated "
        "LiDAR returns and to evaluate final predictions; it is not used for post-inference alignment.",
        "",
    ]
    for scope, title in (
        ("full_image", "Primary: full-image averages"),
        ("outside_support", f"Secondary: outside {args.outside_margin_px:g}-pixel support margin"),
    ):
        lines += [f"## {title}", ""]
        lines.append("| Method | Scenes | RMSE m ↓ | AbsRel % ↓ | MAE m ↓ | Bias m →0 | δ1 % ↑ | Median ratio →1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary:
            if row["scope"] != scope:
                continue
            lines.append(
                f"| {row['label']} | {row['scenes']} | {row['mean_rmse_m']:.4f} | "
                f"{row['mean_absrel_pct']:.3f} | {row['mean_mae_m']:.4f} | "
                f"{row['mean_bias_m']:+.4f} | {row['mean_delta1_pct']:.2f} | "
                f"{row['mean_median_depth_ratio']:.4f} |"
            )
        lines.append("")
    lines += [
        "## Paired full-image bootstrap decisions",
        "",
        "The reported difference is first minus second. Negative favors the first method for error "
        "metrics; positive favors the first method for δ1.",
        "",
        "| Contrast | Metric | Mean Δ | 95% CI | Decision |",
        "|---|---|---:|---:|---|",
    ]
    for row in bootstrap:
        if row["scope"] != "full_image":
            continue
        lines.append(
            f"| {row['contrast']} | {row['metric']} | "
            f"{row['mean_first_minus_second']:+.5f} | "
            f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] | {row['decision']} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- The empirical 32–50-return condition remains the deployment-matched RPLidar baseline.",
        "- The 120-ray condition is an idealized denser commercial 2D single-line scan (60° / 120 = 0.5° nominal sampling), not a claim that every low-cost LiDAR supplies 120 camera-visible returns.",
        "- The two density levels use the same scenes and one-pixel returns. No 3×3 splat is used.",
        "- Hyperparameters for guided Poisson must be locked on development data before this 654-image test.",
        "- The interaction row is the clean test of whether guided-vs-existing Poisson changes with point density.",
    ]
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-dense", help="Create the 120-ray one-line dataset")
    prepare.add_argument("--source-data-root", type=Path, required=True)
    prepare.add_argument("--output-data-root", type=Path, required=True)
    prepare.add_argument("--source-sparse-dir", type=Path)
    prepare.add_argument("--gt-dir", type=Path)
    prepare.add_argument("--points", type=int, default=120)
    prepare.add_argument("--horizontal-fov-deg", type=float, default=60.0)
    prepare.add_argument("--minimum-valid-points", type=int, default=110)
    prepare.add_argument("--max-depth-m", type=float, default=10.0)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(function=prepare_dense)

    guided = commands.add_parser("guided-poisson", help="Run RGB/DA3-guided log-Poisson")
    guided.add_argument("--data-root", type=Path, required=True)
    guided.add_argument("--median-dir", type=Path, required=True)
    guided.add_argument("--output-dir", type=Path, required=True)
    guided.add_argument("--rgb-dir", type=Path)
    guided.add_argument("--sparse-dir", type=Path)
    guided.add_argument("--screen-weight", type=float, default=0.05)
    guided.add_argument("--anchor-weight", type=float, default=100.0)
    guided.add_argument("--color-sigma-lab", type=float, default=12.0)
    guided.add_argument("--depth-sigma-log", type=float, default=0.08)
    guided.add_argument("--weight-floor", type=float, default=1e-3)
    guided.add_argument("--rtol", type=float, default=1e-6)
    guided.add_argument("--maxiter", type=int, default=2000)
    guided.add_argument("--limit", type=int)
    guided.add_argument("--overwrite", action="store_true")
    guided.set_defaults(function=guided_poisson)

    evaluate = commands.add_parser("evaluate-factorial", help="Evaluate the complete 2 × 2 design")
    evaluate.add_argument("--low-data-root", type=Path, required=True)
    evaluate.add_argument("--dense-data-root", type=Path, required=True)
    evaluate.add_argument("--low-da3-root", type=Path, required=True)
    evaluate.add_argument("--dense-da3-root", type=Path, required=True)
    evaluate.add_argument("--low-guided-dir", type=Path, required=True)
    evaluate.add_argument("--dense-guided-dir", type=Path, required=True)
    evaluate.add_argument("--low-any2full-dir", type=Path)
    evaluate.add_argument("--dense-any2full-dir", type=Path)
    evaluate.add_argument("--gt-dir", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--outside-margin-px", type=float, default=10.0)
    evaluate.add_argument("--max-depth-m", type=float, default=10.0)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=20260901)
    evaluate.add_argument("--expected-images", type=int, default=654)
    evaluate.add_argument("--limit", type=int)
    evaluate.set_defaults(function=evaluate_factorial)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
