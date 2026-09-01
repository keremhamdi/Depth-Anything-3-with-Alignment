#!/usr/bin/env python3
"""Training-free DA3 alignment sweep for the real one-line RPLidar dataset.

The script performs the same four-fold blocked held-out evaluation for every
alignment.  An alignment is fitted only on ``depth_fit_points`` from a fold and
is evaluated only at that fold's ``heldout_points.csv`` coordinates.

The primary ranking is pooled camera-Z RMSE in metres.  MAE, P90/P95 absolute
error, bad-10-cm/bad-25-cm rates, bias, delta1 and AbsRel are also retained.
AbsRel is deliberately secondary for this metric-measurement application.

Full-input dense maps are generated separately for qualitative inspection.
They use all available LiDAR anchors and therefore are not the maps used for
the held-out scores.  Without dense ground truth, those maps must not be called
quantitatively accurate outside the scan-line support.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


EPS = 1e-9
PRIMARY_METHODS = (
    "da3_median",
    "da3_l2_scale",
    "da3_log_scale",
    "da3_affine_ls",
    "da3_affine_huber",
    "da3_relative_wls_affine",
    "da3_inverse_affine_huber",
    "da3_log_affine_ls",
    "da3_isotonic",
)

DISPLAY_NAMES = {
    "da3_median": "Median scale",
    "da3_l2_scale": "L2 scale (RMSE fit)",
    "da3_log_scale": "Log-LS scale",
    "da3_affine_ls": "Depth affine LS",
    "da3_affine_huber": "Depth affine Huber",
    "da3_relative_wls_affine": "Relative-WLS affine",
    "da3_inverse_affine_huber": "Inverse-depth affine Huber",
    "da3_log_affine_ls": "Log-depth affine LS",
    "da3_isotonic": "Monotonic isotonic",
    "da3_median_poisson": "Median + Poisson (reference)",
    "any2full": "Any2Full (reference)",
}


@dataclass
class Alignment:
    name: str
    parameters: dict[str, float | str]
    function: Callable[[np.ndarray], np.ndarray]

    def predict(
        self,
        relative: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> tuple[np.ndarray, float]:
        raw = np.asarray(self.function(relative.astype(np.float64)), dtype=np.float64)
        if raw.shape != relative.shape:
            raise ValueError(f"{self.name} returned {raw.shape}; expected {relative.shape}")
        invalid = ~np.isfinite(raw)
        outside = invalid | (raw < min_depth) | (raw > max_depth)
        safe = np.nan_to_num(raw, nan=min_depth, posinf=max_depth, neginf=min_depth)
        safe = np.clip(safe, min_depth, max_depth).astype(np.float32)
        return safe, float(100.0 * np.mean(outside))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--huber-delta", type=float, default=1.345)
    parser.add_argument("--huber-iterations", type=int, default=50)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument(
        "--reference-points",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional existing per_point_metrics.csv; repeat for DA3+Poisson "
            "and Any2Full. Duplicate da3_median rows are ignored."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=PRIMARY_METHODS,
        default=list(PRIMARY_METHODS),
    )
    parser.add_argument(
        "--save-full-npy",
        action="store_true",
        help="Also save every full-input dense metric map (large output).",
    )
    parser.add_argument("--no-panels", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_npy(directory: Path, stem: str) -> Path:
    candidates = (
        directory / f"{stem}.npy",
        directory / f"{stem}_da3small.npy",
        directory / f"{stem}_da3.npy",
    )
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(directory.glob(f"{stem}*.npy"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve {stem} in {directory}: {matches}")


def load_map(path: Path, shape: tuple[int, int]) -> np.ndarray:
    array = np.squeeze(np.load(path)).astype(np.float64)
    if array.shape != shape:
        raise ValueError(f"{path}: shape {array.shape}; expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: non-finite values")
    return array


def anchors_from_sparse(
    relative: np.ndarray,
    sparse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(relative)
        & (relative > 0)
        & np.isfinite(sparse)
        & (sparse > 0)
    )
    x = relative[mask].astype(np.float64)
    y = sparse[mask].astype(np.float64)
    if x.size < 8:
        raise ValueError(f"Only {x.size} usable alignment anchors")
    if float(np.ptp(x)) <= EPS:
        raise ValueError("DA3 anchors have no usable variation")
    return x, y


def weighted_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if weights is None:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    weights = np.asarray(weights, dtype=np.float64)
    root = np.sqrt(np.maximum(weights, EPS))
    return np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]


def huber_irls(
    design: np.ndarray,
    target: np.ndarray,
    base_weights: np.ndarray | None,
    delta: float,
    iterations: int,
) -> np.ndarray:
    base = (
        np.ones(target.size, dtype=np.float64)
        if base_weights is None
        else np.asarray(base_weights, dtype=np.float64)
    )
    beta = weighted_lstsq(design, target, base)
    for _ in range(iterations):
        residual = target - design @ beta
        centre = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - centre)))
        if not math.isfinite(scale) or scale < 1e-8:
            break
        cutoff = delta * scale
        robust = np.ones_like(residual)
        large = np.abs(residual) > cutoff
        robust[large] = cutoff / np.maximum(np.abs(residual[large]), EPS)
        updated = weighted_lstsq(design, target, base * robust)
        if np.linalg.norm(updated - beta) <= 1e-9 * (1.0 + np.linalg.norm(beta)):
            beta = updated
            break
        beta = updated
    return beta


def positive_slope_affine(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, bool]:
    a, b = float(beta[0]), float(beta[1])
    constrained = False
    if not math.isfinite(a) or a <= EPS:
        constrained = True
        a = EPS
        if weights is None:
            b = float(np.mean(y - a * x))
        else:
            b = float(np.average(y - a * x, weights=np.maximum(weights, EPS)))
    return a, b, constrained


def isotonic_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    unique_x, inverse = np.unique(xs, return_inverse=True)
    unique_y = np.zeros(unique_x.size, dtype=np.float64)
    unique_w = np.zeros(unique_x.size, dtype=np.float64)
    for index, group in enumerate(inverse):
        unique_y[group] += ys[index]
        unique_w[group] += 1.0
    unique_y /= np.maximum(unique_w, EPS)

    values: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(unique_y, unique_w)):
        values.append(float(value)); weights.append(float(weight))
        starts.append(index); ends.append(index)
        while len(values) >= 2 and values[-2] > values[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_value = (
                values[-2] * weights[-2] + values[-1] * weights[-1]
            ) / merged_weight
            values[-2:] = [merged_value]
            weights[-2:] = [merged_weight]
            ends[-2:] = [ends[-1]]
            starts.pop()
    fitted = np.empty(unique_x.size, dtype=np.float64)
    for value, start, end in zip(values, starts, ends):
        fitted[start : end + 1] = value
    return unique_x, fitted


def build_alignments(
    relative: np.ndarray,
    sparse: np.ndarray,
    huber_delta: float,
    huber_iterations: int,
) -> dict[str, Alignment]:
    x, y = anchors_from_sparse(relative, sparse)
    design_scale = x[:, None]
    design_affine = np.column_stack([x, np.ones_like(x)])

    median_scale = float(np.median(y / x))
    l2_scale = float(weighted_lstsq(design_scale, y)[0])
    log_scale = float(np.exp(np.mean(np.log(y) - np.log(x))))

    affine_ls_raw = weighted_lstsq(design_affine, y)
    affine_ls = positive_slope_affine(x, y, affine_ls_raw)

    affine_huber_raw = huber_irls(
        design_affine, y, None, huber_delta, huber_iterations
    )
    affine_huber = positive_slope_affine(x, y, affine_huber_raw)

    relative_weights = 1.0 / np.maximum(y * y, EPS)
    relative_raw = weighted_lstsq(design_affine, y, relative_weights)
    relative_affine = positive_slope_affine(
        x, y, relative_raw, relative_weights
    )

    inv_x, inv_y = 1.0 / x, 1.0 / y
    inv_design = np.column_stack([inv_x, np.ones_like(inv_x)])
    inverse_raw = huber_irls(
        inv_design, inv_y, None, huber_delta, huber_iterations
    )
    inv_a, inv_b, inv_constrained = positive_slope_affine(
        inv_x, inv_y, inverse_raw
    )

    log_x, log_y = np.log(x), np.log(y)
    log_design = np.column_stack([log_x, np.ones_like(log_x)])
    log_raw = weighted_lstsq(log_design, log_y)
    log_a = max(float(log_raw[0]), EPS)
    log_b = float(log_raw[1])
    log_constrained = bool(float(log_raw[0]) <= EPS)

    iso_x, iso_y = isotonic_increasing(x, y)

    def affine_alignment(name: str, fit: tuple[float, float, bool]) -> Alignment:
        a, b, constrained = fit
        return Alignment(
            name,
            {"a": a, "b": b, "positive_slope_constrained": str(constrained)},
            lambda depth, a=a, b=b: a * depth + b,
        )

    return {
        "da3_median": Alignment(
            "da3_median", {"scale": median_scale},
            lambda depth, scale=median_scale: scale * depth,
        ),
        "da3_l2_scale": Alignment(
            "da3_l2_scale", {"scale": l2_scale},
            lambda depth, scale=l2_scale: scale * depth,
        ),
        "da3_log_scale": Alignment(
            "da3_log_scale", {"scale": log_scale},
            lambda depth, scale=log_scale: scale * depth,
        ),
        "da3_affine_ls": affine_alignment("da3_affine_ls", affine_ls),
        "da3_affine_huber": affine_alignment("da3_affine_huber", affine_huber),
        "da3_relative_wls_affine": affine_alignment(
            "da3_relative_wls_affine", relative_affine
        ),
        "da3_inverse_affine_huber": Alignment(
            "da3_inverse_affine_huber",
            {
                "a": inv_a,
                "b": inv_b,
                "positive_slope_constrained": str(inv_constrained),
            },
            lambda depth, a=inv_a, b=inv_b: 1.0 / (
                a / np.maximum(depth, EPS) + b
            ),
        ),
        "da3_log_affine_ls": Alignment(
            "da3_log_affine_ls",
            {
                "a": log_a,
                "b": log_b,
                "positive_slope_constrained": str(log_constrained),
            },
            lambda depth, a=log_a, b=log_b: np.exp(b)
            * np.power(np.maximum(depth, EPS), a),
        ),
        "da3_isotonic": Alignment(
            "da3_isotonic",
            {
                "knots": float(iso_x.size),
                "min_anchor_relative": float(iso_x[0]),
                "max_anchor_relative": float(iso_x[-1]),
            },
            lambda depth, knots=iso_x, values=iso_y: np.interp(
                depth, knots, values, left=values[0], right=values[-1]
            ),
        ),
    }


def evaluate_prediction(
    prediction: np.ndarray,
    heldout: Iterable[dict[str, str]],
    method: str,
    fold: int,
) -> list[dict]:
    rows: list[dict] = []
    for point in heldout:
        u, v = int(point["u"]), int(point["v"])
        predicted = float(prediction[v, u])
        target = float(point["z_m"])
        if not math.isfinite(predicted) or predicted <= 0:
            raise ValueError(f"{method} invalid at {point['stem']} ({u}, {v})")
        error = predicted - target
        rows.append(
            {
                "method": method,
                "stem": point["stem"],
                "fold": fold,
                "sector": point.get("sector", ""),
                "u": u,
                "v": v,
                "gt_z_m": target,
                "prediction_m": predicted,
                "signed_error_m": error,
                "abs_error_m": abs(error),
                "absrel_pct": 100.0 * abs(error) / target,
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot aggregate no rows")
    prediction = np.asarray([float(row["prediction_m"]) for row in rows])
    target = np.asarray([float(row["gt_z_m"]) for row in rows])
    error = prediction - target
    absolute = np.abs(error)
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "points": len(rows),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mae_m": float(np.mean(absolute)),
        "p90_abs_m": float(np.percentile(absolute, 90)),
        "p95_abs_m": float(np.percentile(absolute, 95)),
        "bias_m": float(np.mean(error)),
        "bad_010_pct": float(100.0 * np.mean(absolute > 0.10)),
        "bad_025_pct": float(100.0 * np.mean(absolute > 0.25)),
        "delta1_pct": float(100.0 * np.mean(ratio < 1.25)),
        "absrel_pct": float(100.0 * np.mean(absolute / target)),
    }


def metric_value(rows: list[dict], metric: str) -> float:
    return float(aggregate(rows)[metric])


def merge_reference_rows(
    computed: list[dict],
    reference_paths: list[Path],
) -> list[dict]:
    merged = list(computed)
    seen = {
        (str(row["method"]), str(row["stem"]), str(row["fold"]), str(row["u"]), str(row["v"]))
        for row in merged
    }
    for path in reference_paths:
        for row in read_rows(path.expanduser().resolve()):
            key = (
                row["method"], row["stem"], row["fold"], row["u"], row["v"]
            )
            if key in seen:
                continue
            required = ("prediction_m", "gt_z_m")
            if any(field not in row for field in required):
                raise ValueError(f"{path} lacks one of {required}")
            merged.append(row)
            seen.add(key)
    return merged


def summary_tables(
    rows: list[dict],
    stems: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    methods = sorted({str(row["method"]) for row in rows})
    summary = []
    per_scene = []
    by_depth = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                **aggregate(method_rows),
            }
        )
        for stem in stems:
            selected = [row for row in method_rows if row["stem"] == stem]
            if selected:
                per_scene.append({"method": method, "stem": stem, **aggregate(selected)})
        bins = (
            ("near_0_1m", 0.0, 1.0),
            ("middle_1_2m", 1.0, 2.0),
            ("far_ge_2m", 2.0, math.inf),
        )
        for label, low, high in bins:
            selected = [
                row for row in method_rows
                if low <= float(row["gt_z_m"]) < high
            ]
            if selected:
                by_depth.append(
                    {"method": method, "depth_bin": label, **aggregate(selected)}
                )
    summary.sort(key=lambda row: (float(row["rmse_m"]), float(row["mae_m"])))
    return summary, per_scene, by_depth


def paired_bootstrap(
    rows: list[dict],
    baseline: str,
    repetitions: int,
    seed: int,
) -> list[dict]:
    methods = sorted({str(row["method"]) for row in rows})
    if baseline not in methods or repetitions <= 0:
        return []
    key = lambda row: (
        str(row["stem"]), str(row["fold"]), str(row["u"]), str(row["v"])
    )
    baseline_map = {key(row): row for row in rows if row["method"] == baseline}
    rng = np.random.default_rng(seed)
    output = []
    for method in methods:
        if method == baseline:
            continue
        method_map = {key(row): row for row in rows if row["method"] == method}
        common = sorted(set(method_map) & set(baseline_map))
        if not common:
            continue
        scene_keys: dict[str, list[tuple[str, str, str, str]]] = {}
        for item in common:
            scene_keys.setdefault(item[0], []).append(item)
        scenes = sorted(scene_keys)
        if len(scenes) < 2:
            continue
        actual_method = [method_map[item] for item in common]
        actual_base = [baseline_map[item] for item in common]
        for metric in ("rmse_m", "mae_m", "p90_abs_m", "absrel_pct"):
            actual_delta = metric_value(actual_method, metric) - metric_value(actual_base, metric)
            samples = np.empty(repetitions, dtype=np.float64)
            for index in range(repetitions):
                sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
                sampled_keys = [
                    item
                    for scene in sampled_scenes
                    for item in scene_keys[str(scene)]
                ]
                sample_method = [method_map[item] for item in sampled_keys]
                sample_base = [baseline_map[item] for item in sampled_keys]
                samples[index] = (
                    metric_value(sample_method, metric)
                    - metric_value(sample_base, metric)
                )
            output.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "common_points": len(common),
                    "scenes": len(scenes),
                    "delta_method_minus_baseline": actual_delta,
                    "ci95_low": float(np.percentile(samples, 2.5)),
                    "ci95_high": float(np.percentile(samples, 97.5)),
                    "negative_favors_method": True,
                }
            )
    return output


def scene_rmse(rows: list[dict], method: str, stem: str) -> float:
    selected = [
        row for row in rows if row["method"] == method and row["stem"] == stem
    ]
    return float(aggregate(selected)["rmse_m"]) if selected else math.nan


def render_alignment_panel(
    stem: str,
    rgb_path: Path,
    sparse: np.ndarray,
    predictions: dict[str, np.ndarray],
    point_rows: list[dict],
    output: Path,
) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    all_values = np.concatenate(
        [prediction[np.isfinite(prediction) & (prediction > 0)] for prediction in predictions.values()]
    )
    vmax = float(np.clip(np.percentile(all_values, 99.0), 3.0, 10.0))
    methods = list(predictions)
    columns = 4
    cells = 1 + len(methods)
    rows_count = int(math.ceil(cells / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(4.3 * columns, 3.2 * rows_count),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    axes_array[0].imshow(rgb)
    y, x = np.where(sparse > 0)
    axes_array[0].scatter(x, y, s=7, c="cyan")
    axes_array[0].set_title(f"RGB + all {int((sparse > 0).sum())} LiDAR anchors")
    axes_array[0].set_axis_off()

    image = None
    for axis, method in zip(axes_array[1:], methods):
        image = axis.imshow(predictions[method], cmap="turbo", vmin=0, vmax=vmax)
        rmse = scene_rmse(point_rows, method, stem)
        suffix = "" if not math.isfinite(rmse) else f"\nheld-out RMSE {rmse:.3f} m"
        axis.set_title(DISPLAY_NAMES.get(method, method) + suffix, fontsize=10)
        axis.set_axis_off()
    for axis in axes_array[cells:]:
        axis.set_axis_off()
    if image is not None:
        fig.colorbar(image, ax=list(axes_array[:cells]), shrink=0.78, label="camera-Z metric depth (m)")
    fig.suptitle(
        f"{stem} — full-input qualitative DA3 alignment maps on one shared scale\n"
        "All anchors are used here; numerical titles come from separate four-fold held-out maps.",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def print_summary(summary: list[dict]) -> None:
    print("\n===== DA3 ALIGNMENT SWEEP: HELD-OUT SCAN-LINE RESULT =====")
    print("Ranked by pooled RMSE in camera-Z metres; AbsRel is secondary.")
    header = (
        f"{'method':31s} {'pts':>5s} {'RMSE m':>9s} {'MAE m':>9s} "
        f"{'P90 m':>9s} {'bad10%':>8s} {'AbsRel':>8s}"
    )
    print(header)
    for row in summary:
        print(
            f"{str(row['method']):31s} {int(row['points']):5d} "
            f"{float(row['rmse_m']):9.4f} {float(row['mae_m']):9.4f} "
            f"{float(row['p90_abs_m']):9.4f} {float(row['bad_010_pct']):7.2f}% "
            f"{float(row['absrel_pct']):7.2f}%"
        )


def main() -> None:
    args = arguments()
    prepared = args.prepared_root.expanduser().resolve()
    da3_dir = args.da3_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = read_rows(prepared / "manifest.csv")
    stems = [row["stem"] for row in manifest]
    computed_rows: list[dict] = []
    fit_rows: list[dict] = []

    for fold in range(args.folds):
        heldout_by_stem: dict[str, list[dict[str, str]]] = {}
        for row in read_rows(prepared / f"fold_{fold}" / "heldout_points.csv"):
            heldout_by_stem.setdefault(row["stem"], []).append(row)
        for index, manifest_row in enumerate(manifest, 1):
            stem = manifest_row["stem"]
            shape = (int(manifest_row["height"]), int(manifest_row["width"]))
            relative = load_map(resolve_npy(da3_dir, stem), shape)
            sparse = load_map(
                prepared / f"fold_{fold}" / "depth_fit_points" / f"{stem}.npy",
                shape,
            )
            alignments = build_alignments(
                relative, sparse, args.huber_delta, args.huber_iterations
            )
            for method in args.methods:
                alignment = alignments[method]
                prediction, clipped_pct = alignment.predict(
                    relative, args.min_depth, args.max_depth
                )
                computed_rows.extend(
                    evaluate_prediction(
                        prediction, heldout_by_stem.get(stem, []), method, fold
                    )
                )
                fit_rows.append(
                    {
                        "method": method,
                        "stem": stem,
                        "fold": fold,
                        "fit_anchors": int(np.sum(sparse > 0)),
                        "clipped_full_image_pct": clipped_pct,
                        **alignment.parameters,
                    }
                )
            print(
                f"fold {fold} [{index:02d}/{len(stems)}] {stem} "
                f"fit={int(np.sum(sparse > 0))} "
                f"heldout={len(heldout_by_stem.get(stem, []))}",
                flush=True,
            )

    expected = sum(
        len(read_rows(prepared / f"fold_{fold}" / "heldout_points.csv"))
        for fold in range(args.folds)
    )
    for method in args.methods:
        count = sum(row["method"] == method for row in computed_rows)
        if count != expected:
            raise RuntimeError(f"{method}: evaluated {count}; expected {expected}")

    all_rows = merge_reference_rows(computed_rows, args.reference_points)
    summary, per_scene, by_depth = summary_tables(all_rows, stems)
    bootstrap = paired_bootstrap(
        all_rows,
        "any2full",
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )

    write_rows(output / "per_point_metrics.csv", all_rows)
    write_rows(output / "per_scene_metrics.csv", per_scene)
    write_rows(output / "summary_ranked_by_rmse.csv", summary)
    write_rows(output / "metrics_by_depth_range.csv", by_depth)
    write_rows(output / "fit_parameters.csv", fit_rows)
    write_rows(output / "paired_scene_bootstrap_vs_any2full.csv", bootstrap)
    print_summary(summary)

    if not args.no_panels or args.save_full_npy:
        print("\nGenerating full-input qualitative maps...", flush=True)
        full_sparse_dir = prepared / "depth_full_points"
        for index, manifest_row in enumerate(manifest, 1):
            stem = manifest_row["stem"]
            shape = (int(manifest_row["height"]), int(manifest_row["width"]))
            relative = load_map(resolve_npy(da3_dir, stem), shape)
            sparse = load_map(full_sparse_dir / f"{stem}.npy", shape)
            alignments = build_alignments(
                relative, sparse, args.huber_delta, args.huber_iterations
            )
            predictions: dict[str, np.ndarray] = {}
            for method in args.methods:
                prediction, _ = alignments[method].predict(
                    relative, args.min_depth, args.max_depth
                )
                predictions[method] = prediction
                if args.save_full_npy:
                    destination = output / "full_predictions_m" / method / f"{stem}.npy"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.save(destination, prediction)
            if not args.no_panels:
                render_alignment_panel(
                    stem,
                    prepared / "rgb" / f"{stem}.png",
                    sparse,
                    predictions,
                    all_rows,
                    output / "panels" / f"{stem}__da3_alignment_sweep.png",
                )
            print(f"full [{index:02d}/{len(stems)}] {stem}", flush=True)

    print(f"\nOutput: {output}")
    print("These remain sparse held-out scan-line metrics, not dense-GT metrics.")


if __name__ == "__main__":
    main()
