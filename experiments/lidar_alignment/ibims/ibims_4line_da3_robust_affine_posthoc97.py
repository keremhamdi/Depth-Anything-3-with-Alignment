#!/usr/bin/env python3
"""Post-hoc DA3 robust-affine alignment study on the frozen iBims 97 scenes.

This script deliberately leaves the original locked DA3-vs-Any2Full result
untouched.  It reuses the exact saved four-line sparse inputs, cached DA3
relative maps, evaluation masks, and Any2Full metric predictions.

For every scene it compares three two-parameter affine formulations:

    depth_affine:       z = a * r + b
    reciprocal_affine:  1/z = a * (1/r) + b
    disparity_affine:   1/z = a * r + b

where r is the cached DA3 relative map and z is metric depth.  The formulation
is selected without dense GT by leave-one-LiDAR-line-out validation.  Each fit
uses deterministic RANSAC followed by beam-balanced Huber IRLS.  The selected
model is refit on all four lines, then optionally passed through the exact same
validated Poisson implementation used by the locked median baseline.

Because the 97-scene result has already been inspected, this is a post-hoc
development experiment, not a replacement locked test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import binomtest, wilcoxon


VERSION = "1.0-posthoc-robust-affine"
PRIMARY_REGION = "outside_shared_four_line"
ROW_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
DOMAINS = ("depth_affine", "reciprocal_affine", "disparity_affine")
METHODS = {
    "median_poisson": "DA3-SMALL + median + Poisson (locked baseline)",
    "robust_affine": "DA3-SMALL + robust affine (alignment only)",
    "robust_affine_poisson": "DA3-SMALL + robust affine + Poisson",
    "any2full": "Any2Full-vits (native pipeline)",
}
REGIONS = {
    "all_valid": "All valid GT pixels",
    "outside_original_line": "Outside original 1-line support",
    "outside_shared_four_line": "Outside shared 4-line support (primary)",
}
METRICS = (
    "rmse_m",
    "absrel_pct",
    "mae_m",
    "delta1_pct",
    "bad_050_pct",
    "bad_100_pct",
)
LOWER_IS_BETTER = {
    "rmse_m": True,
    "absrel_pct": True,
    "mae_m": True,
    "delta1_pct": False,
    "bad_050_pct": True,
    "bad_100_pct": True,
}


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="Fit robust affine alignment and evaluate all scenes."
    )
    evaluate.add_argument("--da3-root", type=Path, required=True)
    evaluate.add_argument("--prepared-data-root", type=Path, required=True)
    evaluate.add_argument("--any2full-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--expected-scenes", type=positive_int, default=97)
    evaluate.add_argument("--bootstrap-samples", type=positive_int, default=20000)
    evaluate.add_argument("--seed", type=int, default=20260902)
    evaluate.add_argument("--ransac-trials", type=positive_int, default=512)
    evaluate.add_argument("--ransac-mad-multiplier", type=float, default=2.5)
    evaluate.add_argument("--min-inlier-fraction", type=float, default=0.35)
    evaluate.add_argument("--huber-iterations", type=positive_int, default=30)
    evaluate.add_argument("--huber-c", type=float, default=1.345)
    evaluate.add_argument("--rtol", type=float, default=1e-6)
    evaluate.add_argument("--maxiter", type=positive_int, default=5000)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--skip-panels", action="store_true")
    evaluate.add_argument("--plot-max-depth-m", type=float, default=10.0)
    evaluate.add_argument("--plot-error-max-m", type=float, default=1.0)

    subparsers.add_parser("self-test", help="Run deterministic numerical tests.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ransac_mad_multiplier <= 0:
        raise ValueError("--ransac-mad-multiplier must be positive")
    if not 0 < args.min_inlier_fraction <= 1:
        raise ValueError("--min-inlier-fraction must be in (0, 1]")
    if args.huber_c <= 0:
        raise ValueError("--huber-c must be positive")
    if args.rtol <= 0:
        raise ValueError("--rtol must be positive")
    if args.plot_max_depth_m <= 0 or args.plot_error_max_m <= 0:
        raise ValueError("plot limits must be positive")


def resolve_directory(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def configuration_hash(payload: dict[str, Any]) -> str:
    source = {key: value for key, value in payload.items() if key != "configuration_sha256"}
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
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


def load_npy_2d(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.squeeze(np.load(path))
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D NPY at {path}, got {value.shape}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{path}: {value.shape} != {shape}")
    return value


def load_prediction(path: Path, shape: tuple[int, int], valid: np.ndarray) -> np.ndarray:
    value = load_npy_2d(path, shape).astype(np.float32)
    bad = valid & (~np.isfinite(value) | (value <= 0))
    if np.any(bad):
        raise RuntimeError(f"{path}: {int(np.count_nonzero(bad))} invalid evaluation pixels")
    return value


def import_sibling(script_dir: Path) -> Any:
    path = script_dir / "ibims_1line_vs_4line_da3_comparison.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required sibling evaluator is missing: {path}\n"
            "Place this script in the same experiments/lidar_alignment/ibims directory."
        )
    spec = importlib.util.spec_from_file_location("ibims_paired_robust_affine", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "metrics", None)):
        raise AttributeError(f"{path} has no callable metrics")
    return module


def load_poisson(da3_root: Path) -> Callable[..., Any]:
    path = da3_root / "experiments/lidar_alignment/ibims/compare_median_poisson_oasis_100.py"
    if not path.is_file():
        raise FileNotFoundError(f"Validated Poisson source missing: {path}")
    spec = importlib.util.spec_from_file_location("validated_ibims_poisson_posthoc", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "existing_poisson", None)
    if not callable(function):
        raise AttributeError(f"{path} has no callable existing_poisson")
    print(f"Reusing existing_poisson{inspect.signature(function)} from {path}")
    return function


def call_poisson(
    function: Callable[..., Any],
    base: np.ndarray,
    sparse: np.ndarray,
    anchors: np.ndarray,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, int]:
    aliases = {
        "base": base,
        "base_depth": base,
        "depth": base,
        "initial": base,
        "initial_depth": base,
        "aligned": base,
        "aligned_depth": base,
        "prediction": base,
        "pred": base,
        "sparse": sparse,
        "sparse_depth": sparse,
        "lidar": sparse,
        "lidar_depth": sparse,
        "metric_depth": sparse,
        "anchors": anchors,
        "anchor_mask": anchors,
        "sparse_mask": anchors,
        "valid_mask": anchors,
        "rtol": rtol,
        "tol": rtol,
        "maxiter": maxiter,
        "max_iter": maxiter,
    }
    signature = inspect.signature(function)
    kwargs: dict[str, Any] = {}
    unknown: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            unknown.append(name)
    result = (
        function(base, sparse, anchors, rtol, maxiter)
        if unknown
        else function(**kwargs)
    )
    prediction = result[0] if isinstance(result, tuple) else result
    prediction = np.squeeze(np.asarray(prediction, dtype=np.float32))
    if prediction.shape != base.shape:
        raise ValueError(f"Poisson returned {prediction.shape}; expected {base.shape}")
    invalid = ~np.isfinite(prediction) | (prediction <= 0)
    repaired = int(np.count_nonzero(invalid))
    if repaired:
        prediction = prediction.copy()
        prediction[invalid] = base[invalid]
    if not np.all(np.isfinite(prediction) & (prediction > 0)):
        raise RuntimeError("Poisson result remains invalid after same-pixel base repair")
    return prediction.astype(np.float32), repaired


def stable_seed(base: int, *parts: str) -> int:
    text = ":".join((str(base), *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def beam_balanced_weights(rows: np.ndarray) -> np.ndarray:
    result = np.zeros(rows.shape, dtype=np.float64)
    unique = np.unique(rows)
    if unique.size < 2:
        raise ValueError("At least two LiDAR lines are required")
    for row in unique:
        members = rows == row
        result[members] = 1.0 / (unique.size * int(np.count_nonzero(members)))
    result *= result.size / np.sum(result)
    return result


def weighted_line_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x = x[usable].astype(np.float64)
    y = y[usable].astype(np.float64)
    weights = weights[usable].astype(np.float64)
    if x.size < 2:
        raise ValueError("Not enough samples for an affine fit")
    center = float(np.average(x, weights=weights))
    scale = float(np.sqrt(np.average((x - center) ** 2, weights=weights)))
    if not math.isfinite(scale) or scale < 1e-12:
        raise ValueError("Degenerate affine feature")
    normalized = (x - center) / scale
    design = np.column_stack((normalized, np.ones_like(normalized)))
    root_weight = np.sqrt(weights)
    beta, _, rank, _ = np.linalg.lstsq(
        design * root_weight[:, None], y * root_weight, rcond=None
    )
    if rank < 2:
        raise ValueError("Rank-deficient affine fit")
    a = float(beta[0] / scale)
    b = float(beta[1] - beta[0] * center / scale)
    return a, b


def domain_arrays(
    relative: np.ndarray, metric_depth: np.ndarray, domain: str
) -> tuple[np.ndarray, np.ndarray]:
    if domain == "depth_affine":
        return relative.astype(np.float64), metric_depth.astype(np.float64)
    if domain == "reciprocal_affine":
        return 1.0 / relative.astype(np.float64), 1.0 / metric_depth.astype(np.float64)
    if domain == "disparity_affine":
        return relative.astype(np.float64), 1.0 / metric_depth.astype(np.float64)
    raise KeyError(domain)


def decode_depth(relative: np.ndarray, domain: str, a: float, b: float) -> np.ndarray:
    relative64 = relative.astype(np.float64)
    if domain == "depth_affine":
        return a * relative64 + b
    feature = 1.0 / relative64 if domain == "reciprocal_affine" else relative64
    denominator = a * feature + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / denominator


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def residual_floor(target: np.ndarray, domain: str) -> float:
    median = float(np.median(np.abs(target)))
    absolute_floor = 0.005 if domain == "depth_affine" else 1e-4
    return max(absolute_floor, 0.005 * median)


def valid_full_prediction(
    prediction: np.ndarray, maximum_allowed_depth_m: float
) -> bool:
    return bool(
        prediction.ndim == 2
        and np.all(np.isfinite(prediction))
        and np.all(prediction > 0)
        and float(np.max(prediction)) <= maximum_allowed_depth_m
    )


def robust_affine_fit(
    relative: np.ndarray,
    metric_depth: np.ndarray,
    rows: np.ndarray,
    relative_full: np.ndarray,
    domain: str,
    rng_seed: int,
    ransac_trials: int,
    mad_multiplier: float,
    min_inlier_fraction: float,
    huber_iterations: int,
    huber_c: float,
    maximum_allowed_depth_m: float,
) -> dict[str, Any]:
    usable = (
        np.isfinite(relative)
        & (relative > 0)
        & np.isfinite(metric_depth)
        & (metric_depth > 0)
    )
    relative = relative[usable].astype(np.float64)
    metric_depth = metric_depth[usable].astype(np.float64)
    rows = rows[usable]
    if relative.size < 12 or np.unique(rows).size < 2:
        raise RuntimeError("Insufficient robust-affine anchors")
    x, target = domain_arrays(relative, metric_depth, domain)
    base_weights = beam_balanced_weights(rows)
    initial_a, initial_b = weighted_line_fit(x, target, base_weights)
    initial_residual = target - (initial_a * x + initial_b)
    centered = initial_residual - np.median(initial_residual)
    sigma = 1.4826 * float(np.median(np.abs(centered)))
    threshold = max(residual_floor(target, domain), mad_multiplier * sigma)

    candidates: list[tuple[float, float]] = [(initial_a, initial_b)]
    rng = np.random.default_rng(rng_seed)
    for _ in range(ransac_trials):
        indices = rng.choice(x.size, size=2, replace=False)
        x0, x1 = float(x[indices[0]]), float(x[indices[1]])
        if abs(x1 - x0) <= 1e-10 * max(1.0, abs(x0), abs(x1)):
            continue
        a = float((target[indices[1]] - target[indices[0]]) / (x1 - x0))
        b = float(target[indices[0]] - a * x0)
        if math.isfinite(a) and math.isfinite(b) and a > 0:
            candidates.append((a, b))

    best: tuple[float, float, np.ndarray, tuple[float, float, float]] | None = None
    for a, b in candidates:
        if a <= 0:
            continue
        prediction_full = decode_depth(relative_full, domain, a, b)
        if not valid_full_prediction(prediction_full, maximum_allowed_depth_m):
            continue
        residual = np.abs(target - (a * x + b))
        inliers = residual <= threshold
        mass = float(np.sum(base_weights[inliers]) / np.sum(base_weights))
        if mass < min_inlier_fraction:
            continue
        median_error = weighted_median(residual, base_weights)
        mean_error = float(np.average(residual * residual, weights=base_weights))
        score = (mass, -median_error, -mean_error)
        if best is None or score > best[3]:
            best = (a, b, inliers, score)

    if best is None:
        a, b = initial_a, initial_b
        inliers = np.ones(x.size, dtype=bool)
    else:
        a, b, inliers, _ = best

    ransac_weights = base_weights * np.where(inliers, 1.0, 0.05)
    a, b = weighted_line_fit(x, target, ransac_weights)
    for _ in range(huber_iterations):
        residual = target - (a * x + b)
        centered = residual - weighted_median(residual, base_weights)
        robust_sigma = max(
            residual_floor(target, domain),
            1.4826 * weighted_median(np.abs(centered), base_weights),
        )
        cutoff = huber_c * robust_sigma
        absolute = np.abs(residual)
        huber_weight = np.ones_like(absolute)
        large = absolute > cutoff
        huber_weight[large] = cutoff / absolute[large]
        new_a, new_b = weighted_line_fit(x, target, base_weights * huber_weight)
        if max(abs(new_a - a), abs(new_b - b)) < 1e-10:
            a, b = new_a, new_b
            break
        a, b = new_a, new_b

    if a <= 0:
        raise RuntimeError(f"{domain}: fitted a non-positive slope")
    prediction_full = decode_depth(relative_full, domain, a, b)
    if not valid_full_prediction(prediction_full, maximum_allowed_depth_m):
        raise RuntimeError(f"{domain}: fitted model is invalid over the full map")
    anchor_prediction = decode_depth(relative, domain, a, b)
    error = anchor_prediction - metric_depth
    return {
        "domain": domain,
        "a": float(a),
        "b": float(b),
        "prediction": prediction_full.astype(np.float32),
        "anchor_count": int(relative.size),
        "ransac_inlier_count": int(np.count_nonzero(inliers)),
        "ransac_inlier_fraction": float(np.mean(inliers)),
        "ransac_threshold": float(threshold),
        "anchor_rmse_m": float(np.sqrt(np.mean(error * error))),
        "anchor_absrel_pct": float(100.0 * np.mean(np.abs(error) / metric_depth)),
    }


def select_robust_affine(
    scene: str,
    relative_full: np.ndarray,
    sparse: np.ndarray,
    anchors: np.ndarray,
    args: argparse.Namespace,
    maximum_allowed_depth_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_pixels, x_pixels = np.where(
        anchors & np.isfinite(relative_full) & (relative_full > 0)
    )
    if y_pixels.size < 12:
        raise RuntimeError(f"{scene}: only {y_pixels.size} usable anchors")
    relative = relative_full[y_pixels, x_pixels].astype(np.float64)
    metric = sparse[y_pixels, x_pixels].astype(np.float64)
    unique_rows = np.unique(y_pixels)
    if unique_rows.size != 4:
        raise RuntimeError(
            f"{scene}: expected exactly four physical line rows, got {unique_rows.tolist()}"
        )

    domain_results: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        fold_squared_errors: list[float] = []
        fold_absrel: list[float] = []
        failure = ""
        for heldout_row in unique_rows:
            train = y_pixels != heldout_row
            test = ~train
            try:
                fit = robust_affine_fit(
                    relative[train],
                    metric[train],
                    y_pixels[train],
                    relative_full,
                    domain,
                    stable_seed(args.seed, scene, domain, str(int(heldout_row))),
                    args.ransac_trials,
                    args.ransac_mad_multiplier,
                    args.min_inlier_fraction,
                    args.huber_iterations,
                    args.huber_c,
                    maximum_allowed_depth_m,
                )
                predicted = decode_depth(relative[test], domain, fit["a"], fit["b"])
                if not np.all(np.isfinite(predicted) & (predicted > 0)):
                    raise RuntimeError("invalid held-out prediction")
                error = predicted - metric[test]
                fold_squared_errors.append(float(np.mean(error * error)))
                fold_absrel.append(float(np.mean(np.abs(error) / metric[test])))
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                failure = str(error)
                break
        if failure:
            domain_results[domain] = {
                "cv_rmse_m": math.inf,
                "cv_absrel_pct": math.inf,
                "failure": failure,
            }
            continue
        domain_results[domain] = {
            "cv_rmse_m": float(np.sqrt(np.mean(fold_squared_errors))),
            "cv_absrel_pct": float(100.0 * np.mean(fold_absrel)),
            "failure": "",
        }

    ranked = sorted(
        DOMAINS,
        key=lambda domain: (
            float(domain_results[domain]["cv_rmse_m"]),
            float(domain_results[domain]["cv_absrel_pct"]),
            DOMAINS.index(domain),
        ),
    )
    selected = ranked[0]
    if not math.isfinite(float(domain_results[selected]["cv_rmse_m"])):
        raise RuntimeError(f"{scene}: all affine domains failed: {domain_results}")
    final_fit = robust_affine_fit(
        relative,
        metric,
        y_pixels,
        relative_full,
        selected,
        stable_seed(args.seed, scene, selected, "all"),
        args.ransac_trials,
        args.ransac_mad_multiplier,
        args.min_inlier_fraction,
        args.huber_iterations,
        args.huber_c,
        maximum_allowed_depth_m,
    )
    diagnostics: dict[str, Any] = {
        "scene": scene,
        "selected_domain": selected,
        "selected_cv_rmse_m": domain_results[selected]["cv_rmse_m"],
        "selected_cv_absrel_pct": domain_results[selected]["cv_absrel_pct"],
        "a": final_fit["a"],
        "b": final_fit["b"],
        "anchor_count": final_fit["anchor_count"],
        "ransac_inlier_count": final_fit["ransac_inlier_count"],
        "ransac_inlier_fraction": final_fit["ransac_inlier_fraction"],
        "ransac_threshold": final_fit["ransac_threshold"],
        "fit_anchor_rmse_m": final_fit["anchor_rmse_m"],
        "fit_anchor_absrel_pct": final_fit["anchor_absrel_pct"],
    }
    for domain in DOMAINS:
        diagnostics[f"{domain}_cv_rmse_m"] = domain_results[domain]["cv_rmse_m"]
        diagnostics[f"{domain}_cv_absrel_pct"] = domain_results[domain]["cv_absrel_pct"]
        diagnostics[f"{domain}_failure"] = domain_results[domain]["failure"]
    return final_fit["prediction"], diagnostics


def common_masks(
    valid: np.ndarray,
    one_anchors: np.ndarray,
    four_anchors: np.ndarray,
    margin_px: int,
) -> dict[str, np.ndarray]:
    if margin_px < 0:
        raise ValueError("outside margin cannot be negative")
    one_distance = distance_transform_edt(~one_anchors)
    four_distance = distance_transform_edt(~four_anchors)
    result = {
        "all_valid": valid.copy(),
        "outside_original_line": valid & (one_distance > margin_px),
        "outside_shared_four_line": valid & (four_distance > margin_px),
    }
    for name, mask in result.items():
        if not np.any(mask):
            raise RuntimeError(f"Empty evaluation mask: {name}")
    return result


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["region"]))].append(row)
    result: list[dict[str, Any]] = []
    for (method, region), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "method": method,
            "method_label": METHODS[method],
            "region": region,
            "region_label": REGIONS[region],
            "scene_count": len(group),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"mean_{metric}"] = float(np.mean(values))
            item[f"median_{metric}"] = float(np.median(values))
            item[f"p90_{metric}"] = float(np.quantile(values, 0.90))
            item[f"max_{metric}"] = float(np.max(values))
        result.append(item)
    return result


def bootstrap_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    remaining = samples
    while remaining:
        take = min(1000, remaining)
        indices = rng.integers(0, values.size, size=(take, values.size))
        chunks.append(np.mean(values[indices], axis=1))
        remaining -= take
    low, high = np.quantile(np.concatenate(chunks), (0.025, 0.975))
    return float(low), float(high)


def wilcoxon_pvalue(values: np.ndarray) -> float:
    if np.allclose(values, 0):
        return 1.0
    try:
        return float(
            wilcoxon(
                values,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except ValueError:
        return 1.0


def paired_statistics(
    rows: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["scene"]), str(row["method"]), str(row["region"])): row
        for row in rows
    }
    scenes = sorted({str(row["scene"]) for row in rows})
    comparisons = (
        ("robust_vs_median", "robust_affine_poisson", "median_poisson"),
        ("robust_vs_any2full", "robust_affine_poisson", "any2full"),
        ("alignment_only_vs_median", "robust_affine", "median_poisson"),
    )
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for comparison, candidate, reference in comparisons:
        for region in REGIONS:
            for metric in METRICS:
                candidate_values = np.asarray(
                    [float(lookup[(scene, candidate, region)][metric]) for scene in scenes]
                )
                reference_values = np.asarray(
                    [float(lookup[(scene, reference, region)][metric]) for scene in scenes]
                )
                improvement = (
                    reference_values - candidate_values
                    if LOWER_IS_BETTER[metric]
                    else candidate_values - reference_values
                )
                low, high = bootstrap_ci(improvement, bootstrap_samples, rng)
                wins = int(np.count_nonzero(improvement > 0))
                ties = int(np.count_nonzero(np.isclose(improvement, 0)))
                win_ci = binomtest(wins, len(scenes), 0.5).proportion_ci(
                    confidence_level=0.95, method="exact"
                )
                denominator = abs(float(np.mean(reference_values)))
                result.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "reference": reference,
                        "region": region,
                        "metric": metric,
                        "scene_count": len(scenes),
                        "candidate_mean": float(np.mean(candidate_values)),
                        "reference_mean": float(np.mean(reference_values)),
                        "candidate_improvement_mean": float(np.mean(improvement)),
                        "candidate_relative_improvement_pct": (
                            100.0 * float(np.mean(improvement)) / denominator
                            if denominator > 0
                            else math.nan
                        ),
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "wilcoxon_two_sided_p": wilcoxon_pvalue(improvement),
                        "candidate_scene_wins": wins,
                        "ties": ties,
                        "reference_scene_wins": int(np.count_nonzero(improvement < 0)),
                        "candidate_win_rate_pct": 100.0 * wins / len(scenes),
                        "candidate_win_rate_ci95_low_pct": 100.0 * float(win_ci.low),
                        "candidate_win_rate_ci95_high_pct": 100.0 * float(win_ci.high),
                        "positive_improvement_means": "candidate is better",
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


def paired_lookup(
    rows: list[dict[str, Any]], comparison: str, region: str, metric: str
) -> dict[str, Any]:
    for row in rows:
        if (
            row["comparison"] == comparison
            and row["region"] == region
            and row["metric"] == metric
        ):
            return row
    raise KeyError((comparison, region, metric))


def metric_lookup(
    rows: list[dict[str, Any]], scene: str, method: str, region: str
) -> dict[str, Any]:
    for row in rows:
        if row["scene"] == scene and row["method"] == method and row["region"] == region:
            return row
    raise KeyError((scene, method, region))


def posthoc_decision(paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = paired_lookup(
        paired_rows, "robust_vs_any2full", PRIMARY_REGION, "rmse_m"
    )
    improvement_pct = float(row["candidate_relative_improvement_pct"])
    low = float(row["bootstrap_ci95_low"])
    high = float(row["bootstrap_ci95_high"])
    pvalue = float(row["wilcoxon_two_sided_p"])
    win_rate = float(row["candidate_win_rate_pct"])
    if improvement_pct >= 5.0 and low > 0 and pvalue < 0.05 and win_rate > 50:
        code = "POSTHOC_DA3_ROBUST_AFFINE_LOWER_RMSE"
        winner = "DA3 robust affine + Poisson"
    elif improvement_pct <= -5.0 and high < 0 and pvalue < 0.05 and win_rate < 50:
        code = "POSTHOC_ANY2FULL_LOWER_RMSE"
        winner = "Any2Full"
    else:
        code = "POSTHOC_NO_CONCLUSIVE_RMSE_WINNER"
        winner = "inconclusive"
    return {
        "decision_code": code,
        "winner": winner,
        "status": "post-hoc development evidence; not a fresh locked test",
        "primary_region": PRIMARY_REGION,
        "primary_metric": "equal-scene-weight RMSE",
        "robust_affine_poisson_rmse_m": row["candidate_mean"],
        "any2full_rmse_m": row["reference_mean"],
        "robust_relative_improvement_pct": improvement_pct,
        "paired_improvement_ci95_low_m": low,
        "paired_improvement_ci95_high_m": high,
        "wilcoxon_two_sided_p": pvalue,
        "robust_scene_win_rate_pct": win_rate,
    }


def protocol_payload(
    args: argparse.Namespace,
    prepared_root: Path,
    any2full_dir: Path,
    source_protocol: Path,
) -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "status": "post-hoc development experiment",
        "prepared_data_root": str(prepared_root),
        "prepared_protocol_sha256": sha256_file(source_protocol),
        "any2full_dir": str(any2full_dir),
        "domains": list(DOMAINS),
        "selection": "four-fold leave-one-physical-LiDAR-line-out anchor RMSE",
        "fit": "deterministic RANSAC followed by beam-balanced Huber IRLS",
        "ransac_trials": args.ransac_trials,
        "ransac_mad_multiplier": args.ransac_mad_multiplier,
        "min_inlier_fraction": args.min_inlier_fraction,
        "huber_iterations": args.huber_iterations,
        "huber_c": args.huber_c,
        "poisson_rtol": args.rtol,
        "poisson_maxiter": args.maxiter,
        "primary_region": PRIMARY_REGION,
        "dense_gt_use": "evaluation only; never fitting or model selection",
        "claim_boundary": "97 scenes were previously inspected; results are post-hoc",
    }
    payload["configuration_sha256"] = configuration_hash(payload)
    return payload


def completed_output(
    row: dict[str, str], output_dir: Path, shape: tuple[int, int]
) -> bool:
    try:
        alignment = output_dir / row["alignment_prediction"]
        poisson = output_dir / row["poisson_prediction"]
        if not alignment.is_file() or not poisson.is_file():
            return False
        if sha256_file(alignment) != row.get("alignment_prediction_sha256"):
            return False
        if sha256_file(poisson) != row.get("poisson_prediction_sha256"):
            return False
        for path in (alignment, poisson):
            value = load_npy_2d(path, shape)
            if not np.all(np.isfinite(value) & (value > 0)):
                return False
        return True
    except (KeyError, OSError, ValueError, EOFError):
        return False


def summary_figure(metric_rows: list[dict[str, Any]], output: Path) -> None:
    scenes = sorted({str(row["scene"]) for row in metric_rows})
    robust = np.asarray(
        [
            float(
                metric_lookup(
                    metric_rows, scene, "robust_affine_poisson", PRIMARY_REGION
                )["rmse_m"]
            )
            for scene in scenes
        ]
    )
    median = np.asarray(
        [
            float(metric_lookup(metric_rows, scene, "median_poisson", PRIMARY_REGION)["rmse_m"])
            for scene in scenes
        ]
    )
    any2full = np.asarray(
        [
            float(metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION)["rmse_m"])
            for scene in scenes
        ]
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for axis, reference, label in (
        (axes[0], median, "Median + Poisson"),
        (axes[1], any2full, "Any2Full"),
    ):
        limit = 1.05 * max(float(np.max(reference)), float(np.max(robust)))
        axis.scatter(reference, robust, s=24, alpha=0.75)
        axis.plot([0, limit], [0, limit], "k--", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_xlabel(f"{label} RMSE (m)")
        axis.set_ylabel("Robust affine + Poisson RMSE (m)")
        axis.set_title("Below line = robust affine wins")
        axis.grid(alpha=0.2)
    improvement = any2full - robust
    axes[2].hist(improvement, bins=20, color="#3b82f6", alpha=0.85)
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Any2Full RMSE - robust DA3 RMSE (m)")
    axes[2].set_ylabel("Scenes")
    axes[2].set_title("Positive = robust DA3 wins")
    axes[2].grid(alpha=0.2)
    figure.suptitle("Post-hoc robust-affine DA3 evaluation — primary region")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def comparison_panel(
    scene: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    anchors: np.ndarray,
    primary_mask: np.ndarray,
    robust: np.ndarray,
    any2full: np.ndarray,
    robust_metric: dict[str, Any],
    any2full_metric: dict[str, Any],
    output: Path,
    depth_max: float,
    error_max: float,
) -> None:
    robust_error = np.where(valid, np.abs(robust - gt), np.nan)
    a2f_error = np.where(valid, np.abs(any2full - gt), np.nan)
    figure, axes = plt.subplots(2, 4, figsize=(22, 10.5), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    y, x = np.where(anchors)
    axes[0, 0].scatter(x, y, s=3, c="cyan", linewidths=0)
    axes[0, 0].set_title(f"Shared four-line input ({len(x)} anchors)")
    depth_image = axes[0, 1].imshow(
        np.where(valid, gt, np.nan), cmap="turbo", vmin=0, vmax=depth_max
    )
    axes[0, 1].set_title("Metric GT")
    axes[0, 2].imshow(
        np.where(valid, robust, np.nan), cmap="turbo", vmin=0, vmax=depth_max
    )
    axes[0, 2].set_title(
        "DA3 robust affine + Poisson\n"
        f"RMSE {float(robust_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(robust_metric['absrel_pct']):.2f}%"
    )
    axes[0, 3].imshow(
        np.where(valid, any2full, np.nan), cmap="turbo", vmin=0, vmax=depth_max
    )
    axes[0, 3].set_title(
        "Any2Full\n"
        f"RMSE {float(any2full_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(any2full_metric['absrel_pct']):.2f}%"
    )
    axes[1, 0].imshow(primary_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Primary outside-four-line mask")
    error_image = axes[1, 1].imshow(
        robust_error, cmap="magma", vmin=0, vmax=error_max
    )
    axes[1, 1].set_title("Robust DA3 absolute error")
    axes[1, 2].imshow(a2f_error, cmap="magma", vmin=0, vmax=error_max)
    axes[1, 2].set_title("Any2Full absolute error")
    gain = a2f_error - robust_error
    gain_image = axes[1, 3].imshow(
        gain, cmap="RdBu_r", vmin=-error_max, vmax=error_max
    )
    axes[1, 3].set_title("Error difference\nred = DA3 better; blue = Any2Full better")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.colorbar(depth_image, ax=axes[0, 1:].tolist(), shrink=0.72, label="Depth (m)")
    figure.colorbar(error_image, ax=axes[1, 1:3].tolist(), shrink=0.72, label="Error (m)")
    figure.colorbar(gain_image, ax=axes[1, 3], shrink=0.72, label="A2F error - DA3 error (m)")
    figure.suptitle(scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Post-hoc iBims four-line robust-affine DA3 result",
        "",
        f"## Result: {decision['winner']}",
        "",
        "**Status:** POST-HOC DEVELOPMENT EVIDENCE — the same 97 scenes were previously inspected.",
        "",
        "The robust-affine formulation is selected per scene using only leave-one-LiDAR-line-out sparse-anchor error. Dense GT is used only after prediction for evaluation.",
        "",
        "## Aggregate results",
        "",
        "| Common region | Method | RMSE | AbsRel | MAE | delta1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        for method in METHODS:
            row = summary_lookup(summary_rows, method, region)
            lines.append(
                f"| {REGIONS[region]} | {METHODS[method]} | "
                f"{float(row['mean_rmse_m']):.4f} m | "
                f"{float(row['mean_absrel_pct']):.3f}% | "
                f"{float(row['mean_mae_m']):.4f} m | "
                f"{float(row['mean_delta1_pct']):.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Primary paired comparisons",
            "",
            "A positive improvement means robust-affine DA3 reduced error.",
            "",
            "| Reference | Metric | Robust DA3 | Reference | Improvement | 95% CI | p-value | Win rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison, label in (
        ("robust_vs_median", "Median + Poisson"),
        ("robust_vs_any2full", "Any2Full"),
    ):
        for metric in ("rmse_m", "absrel_pct"):
            row = paired_lookup(paired_rows, comparison, PRIMARY_REGION, metric)
            unit = " m" if metric == "rmse_m" else "%"
            lines.append(
                f"| {label} | {metric} | {float(row['candidate_mean']):.4f}{unit} | "
                f"{float(row['reference_mean']):.4f}{unit} | "
                f"{float(row['candidate_relative_improvement_pct']):+.2f}% | "
                f"[{float(row['bootstrap_ci95_low']):+.4f}, {float(row['bootstrap_ci95_high']):+.4f}]{unit} | "
                f"{float(row['wilcoxon_two_sided_p']):.6g} | "
                f"{float(row['candidate_win_rate_pct']):.1f}% |"
            )
    counts = Counter(str(row["selected_domain"]) for row in diagnostics)
    lines.extend(
        [
            "",
            "## Sparse-only model selection",
            "",
            "| Affine formulation | Scenes selected |",
            "|---|---:|",
        ]
    )
    for domain in DOMAINS:
        lines.append(f"| {domain} | {counts.get(domain, 0)} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This experiment can determine whether robust affine alignment is a useful DA3-v2 development direction on these data. It cannot retroactively replace the earlier locked decision. A superiority claim requires a new untouched indoor test set or a predeclared external validation benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    validate_args(args)
    da3_root = resolve_directory(args.da3_root, "DA3 root")
    prepared_root = resolve_directory(args.prepared_data_root, "prepared locked data")
    any2full_dir = resolve_directory(args.any2full_dir, "Any2Full predictions")
    output_dir = args.output_dir.expanduser().resolve()
    source_protocol_path = prepared_root / "protocol.json"
    source_protocol = read_json(source_protocol_path)
    if source_protocol.get("configuration_sha256") != configuration_hash(source_protocol):
        raise RuntimeError("Prepared protocol hash is invalid or was edited")
    scenes = [str(value) for value in source_protocol.get("locked_scenes", [])]
    if len(scenes) != args.expected_scenes or len(set(scenes)) != len(scenes):
        raise RuntimeError(
            f"Prepared protocol contains {len(scenes)} unique-scene entries; expected {args.expected_scenes}"
        )
    fractions = tuple(float(value) for value in source_protocol.get("row_fractions", []))
    if len(fractions) != 4 or not np.allclose(fractions, ROW_FRACTIONS):
        raise RuntimeError(f"Prepared placement is not the expected maximum coverage: {fractions}")
    manifest_rows = read_csv(prepared_root / "manifest.csv")
    manifest_by_scene = {row.get("scene", ""): row for row in manifest_rows}
    if set(manifest_by_scene) != set(scenes):
        raise RuntimeError("Prepared manifest is incomplete or contains extra scenes")

    expected_predictions = {f"{scene}.npy" for scene in scenes}
    found_predictions = {
        path.name
        for path in any2full_dir.glob("*.npy")
        if path.is_file() and not path.stem.endswith("_rel")
    }
    missing = sorted(expected_predictions - found_predictions)
    extras = sorted(found_predictions - expected_predictions)
    if missing or extras:
        raise RuntimeError(
            "Any2Full directory must contain exactly the 97 metric predictions. "
            f"missing={missing}, extras={extras}"
        )

    protocol = protocol_payload(
        args, prepared_root, any2full_dir, source_protocol_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        old = read_json(protocol_path)
        if old.get("configuration_sha256") != protocol["configuration_sha256"]:
            raise RuntimeError(
                f"{output_dir} contains a different experiment; use another output directory"
            )
        if not args.resume:
            raise FileExistsError(
                f"{output_dir} already exists; pass --resume to reuse verified scene outputs"
            )
    else:
        atomic_json(protocol_path, protocol)

    paired = import_sibling(Path(__file__).resolve().parent)
    poisson = load_poisson(da3_root)
    diagnostics_path = output_dir / "fit_diagnostics.csv"
    diagnostics_rows = read_csv(diagnostics_path) if args.resume else []
    diagnostics_by_scene = {row.get("scene", ""): row for row in diagnostics_rows}
    prediction_dir = output_dir / "robust_affine_predictions_m"
    poisson_dir = output_dir / "robust_affine_poisson_predictions_m"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    poisson_dir.mkdir(parents=True, exist_ok=True)

    margin = int(source_protocol["outside_margin_px"])
    sensor_max = float(source_protocol.get("sensor_max_depth_m", 32.0))
    maximum_allowed_depth_m = max(100.0, 4.0 * sensor_max)
    metric_rows: list[dict[str, Any]] = []

    for index, scene in enumerate(scenes, start=1):
        manifest = manifest_by_scene[scene]
        paths = {
            key: prepared_root / manifest[key]
            for key in ("rgb", "sparse", "gt", "valid", "one_mask", "four_mask", "da3")
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256_file(paths["sparse"]) != manifest.get("sparse_sha256"):
            raise RuntimeError(f"{scene}: sparse input hash changed")
        source_relative = Path(manifest["source_da3_relative_path"]).expanduser().resolve()
        if not source_relative.is_file():
            raise FileNotFoundError(source_relative)
        if sha256_file(source_relative) != manifest.get("source_da3_relative_sha256"):
            raise RuntimeError(f"{scene}: cached DA3 relative-map hash changed")

        gt = load_npy_2d(paths["gt"]).astype(np.float32)
        valid = load_npy_2d(paths["valid"], gt.shape).astype(bool)
        one_anchors = load_npy_2d(paths["one_mask"], gt.shape).astype(bool)
        four_anchors = load_npy_2d(paths["four_mask"], gt.shape).astype(bool)
        sparse = load_npy_2d(paths["sparse"], gt.shape).astype(np.float32)
        if not np.array_equal(sparse > 0, four_anchors):
            raise RuntimeError(f"{scene}: sparse support changed")
        relative = load_npy_2d(source_relative, gt.shape).astype(np.float32)
        if not np.all(np.isfinite(relative) & (relative > 0)):
            raise RuntimeError(f"{scene}: cached DA3 relative map is not strictly positive")

        alignment_path = prediction_dir / f"{scene}.npy"
        poisson_path = poisson_dir / f"{scene}.npy"
        old = diagnostics_by_scene.get(scene)
        if args.resume and old is not None and completed_output(old, output_dir, gt.shape):
            robust_alignment = load_prediction(alignment_path, gt.shape, valid)
            robust_poisson = load_prediction(poisson_path, gt.shape, valid)
            print(f"[{index:3d}/{len(scenes)}] {scene} resume-skip", flush=True)
        else:
            robust_alignment, diagnostic = select_robust_affine(
                scene,
                relative,
                sparse,
                four_anchors,
                args,
                maximum_allowed_depth_m,
            )
            robust_poisson, repaired = call_poisson(
                poisson,
                robust_alignment,
                sparse,
                four_anchors,
                args.rtol,
                args.maxiter,
            )
            np.save(alignment_path, robust_alignment.astype(np.float32))
            np.save(poisson_path, robust_poisson.astype(np.float32))
            diagnostic.update(
                {
                    "poisson_repaired_pixels": repaired,
                    "source_relative_path": str(source_relative),
                    "source_relative_sha256": sha256_file(source_relative),
                    "sparse_sha256": sha256_file(paths["sparse"]),
                    "alignment_prediction": str(alignment_path.relative_to(output_dir)),
                    "alignment_prediction_sha256": sha256_file(alignment_path),
                    "poisson_prediction": str(poisson_path.relative_to(output_dir)),
                    "poisson_prediction_sha256": sha256_file(poisson_path),
                }
            )
            diagnostics_rows = [row for row in diagnostics_rows if row.get("scene") != scene]
            diagnostics_rows.append(diagnostic)
            diagnostics_rows.sort(key=lambda row: str(row["scene"]))
            write_csv(diagnostics_path, diagnostics_rows)
            diagnostics_by_scene[scene] = {
                key: str(value) for key, value in diagnostic.items()
            }
            print(
                f"[{index:3d}/{len(scenes)}] {scene}: "
                f"{diagnostic['selected_domain']}; "
                f"CV RMSE {float(diagnostic['selected_cv_rmse_m']):.4f} m",
                flush=True,
            )

        median_poisson = load_prediction(paths["da3"], gt.shape, valid)
        any2full = load_prediction(any2full_dir / f"{scene}.npy", gt.shape, valid)
        masks = common_masks(valid, one_anchors, four_anchors, margin)
        for method, prediction in (
            ("median_poisson", median_poisson),
            ("robust_affine", robust_alignment),
            ("robust_affine_poisson", robust_poisson),
            ("any2full", any2full),
        ):
            for region, mask in masks.items():
                metric_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "method_label": METHODS[method],
                        "region": region,
                        "region_label": REGIONS[region],
                        "physical_anchor_count": int(np.count_nonzero(four_anchors)),
                        **paired.metrics(prediction, gt, mask),
                    }
                )

    diagnostics_rows = read_csv(diagnostics_path)
    if set(row.get("scene", "") for row in diagnostics_rows) != set(scenes):
        raise RuntimeError("Fit diagnostics are incomplete after processing")
    summary_rows = aggregate_metrics(metric_rows)
    paired_rows = paired_statistics(metric_rows, args.bootstrap_samples, args.seed)
    decision = posthoc_decision(paired_rows)
    write_csv(output_dir / "per_scene_metrics.csv", metric_rows)
    write_csv(output_dir / "summary_equal_scene_weight.csv", summary_rows)
    write_csv(output_dir / "paired_statistics.csv", paired_rows)
    atomic_json(output_dir / "posthoc_decision.json", decision)
    summary_figure(metric_rows, output_dir / "posthoc_summary.png")

    if not args.skip_panels:
        differences: list[tuple[float, str]] = []
        for scene in scenes:
            robust_metric = metric_lookup(
                metric_rows, scene, "robust_affine_poisson", PRIMARY_REGION
            )
            a2f_metric = metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION)
            differences.append(
                (float(a2f_metric["rmse_m"]) - float(robust_metric["rmse_m"]), scene)
            )
        differences.sort()
        choices = (
            ("any2full_best", differences[0][1]),
            ("typical", differences[len(differences) // 2][1]),
            ("robust_da3_best", differences[-1][1]),
        )
        for role, scene in choices:
            manifest = manifest_by_scene[scene]
            paths = {
                key: prepared_root / manifest[key]
                for key in ("rgb", "sparse", "gt", "valid", "four_mask", "one_mask")
            }
            gt = load_npy_2d(paths["gt"]).astype(np.float32)
            valid = load_npy_2d(paths["valid"], gt.shape).astype(bool)
            anchors = load_npy_2d(paths["four_mask"], gt.shape).astype(bool)
            one_anchors = load_npy_2d(paths["one_mask"], gt.shape).astype(bool)
            rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)
            robust = load_prediction(poisson_dir / f"{scene}.npy", gt.shape, valid)
            any2full = load_prediction(any2full_dir / f"{scene}.npy", gt.shape, valid)
            masks = common_masks(valid, one_anchors, anchors, margin)
            comparison_panel(
                scene,
                rgb,
                gt,
                valid,
                anchors,
                masks[PRIMARY_REGION],
                robust,
                any2full,
                metric_lookup(metric_rows, scene, "robust_affine_poisson", PRIMARY_REGION),
                metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION),
                output_dir / "automatic_examples" / f"{role}__{scene}.png",
                args.plot_max_depth_m,
                args.plot_error_max_m,
            )

    report = output_dir / "posthoc_robust_affine_report.md"
    write_report(report, summary_rows, paired_rows, diagnostics_rows, decision)
    print("\n===== POST-HOC ROBUST-AFFINE RESULT =====\n")
    print(report.read_text(encoding="utf-8"))
    print(f"Decision JSON: {output_dir / 'posthoc_decision.json'}")
    print(f"Summary chart: {output_dir / 'posthoc_summary.png'}")
    print(f"Full report: {report}")


def synthetic_args() -> argparse.Namespace:
    return argparse.Namespace(
        seed=19,
        ransac_trials=256,
        ransac_mad_multiplier=2.5,
        min_inlier_fraction=0.35,
        huber_iterations=30,
        huber_c=1.345,
    )


def self_test() -> None:
    args = synthetic_args()
    height, width = 80, 100
    yy, xx = np.mgrid[0:height, 0:width]
    rows = np.asarray([10, 30, 50, 70])
    anchors = np.zeros((height, width), dtype=bool)
    anchors[rows, :] = True

    relative_depth = 0.4 + 0.006 * xx + 0.008 * yy
    gt_depth = 2.3 * relative_depth + 0.45
    sparse_depth = np.zeros_like(gt_depth)
    sparse_depth[anchors] = gt_depth[anchors]
    selected_depth, diagnostic_depth = select_robust_affine(
        "synthetic_depth",
        relative_depth.astype(np.float32),
        sparse_depth.astype(np.float32),
        anchors,
        args,
        100.0,
    )
    if diagnostic_depth["selected_domain"] != "depth_affine":
        raise AssertionError(diagnostic_depth)
    if float(np.sqrt(np.mean((selected_depth - gt_depth) ** 2))) > 1e-5:
        raise AssertionError("depth-affine recovery is inaccurate")

    relative_disparity = 0.08 + 0.003 * xx + 0.002 * yy
    gt_inverse = 1.7 * relative_disparity + 0.035
    gt_from_disparity = 1.0 / gt_inverse
    sparse_disparity = np.zeros_like(gt_from_disparity)
    sparse_disparity[anchors] = gt_from_disparity[anchors]
    selected_disparity, diagnostic_disparity = select_robust_affine(
        "synthetic_disparity",
        relative_disparity.astype(np.float32),
        sparse_disparity.astype(np.float32),
        anchors,
        args,
        100.0,
    )
    if diagnostic_disparity["selected_domain"] != "disparity_affine":
        raise AssertionError(diagnostic_disparity)
    if float(np.sqrt(np.mean((selected_disparity - gt_from_disparity) ** 2))) > 1e-4:
        raise AssertionError("disparity-affine recovery is inaccurate")
    print("SELF-TEST PASSED")


def main() -> None:
    args = arguments()
    if args.command == "evaluate":
        evaluate(args)
    elif args.command == "self-test":
        self_test()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
