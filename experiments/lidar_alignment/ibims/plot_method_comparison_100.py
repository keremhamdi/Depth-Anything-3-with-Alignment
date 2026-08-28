#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive 100-scene comparison of Any2Full and DA3 alignment methods.

The input is ``per_scene_metrics.csv`` written by
``compare_any2full_da3_100.py``.  The script intentionally treats scenes as the
independent experimental units.  It reports macro statistics, pixel-pooled
statistics, paired scene-by-scene comparisons against Any2Full, method ranks,
tail-failure rates, and optional LiDAR-coverage diagnostics.

Example
-------
python plot_method_comparison_100.py \
  --csv experiments/lidar_alignment/outputs/comparison_any2full_da3_v21_100/per_scene_metrics.csv \
  --fits-csv auto \
  --out-dir experiments/lidar_alignment/outputs/comparison_any2full_da3_v21_100/analysis

Outputs
-------
* 01_overview_dashboard.png
* 02_absrel_distributions.png
* 03_paired_vs_baseline.png
* 04_metric_dashboard.png
* 05_coverage_analysis.png (when per_scene_fits.csv is available)
* summary_by_method_region.csv
* paired_vs_baseline.csv
* scene_rankings.csv
* threshold_failure_rates.csv
* worst_scenes.csv
* coverage_correlations.csv (when fits are available)
* comparison_report.md

Positive paired improvement means the candidate has LOWER error than the
baseline.  Dense ground truth is used only through the evaluator's metric CSV;
this plotting script does not fit or recalibrate any prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

try:
    from scipy.stats import spearmanr, wilcoxon
except ImportError:  # The core report still works without SciPy.
    spearmanr = None
    wilcoxon = None


METRICS = ("rmse_m", "mae_m", "absrel_pct", "rmsrel_pct", "bias_m")
REGION_ORDER = (
    "all",
    "non_anchor",
    "inside_support",
    "below_support",
    "above_support",
    "outside_support",
    "near_0_2m",
    "anchors",
)
REGION_LABELS = {
    "all": "All valid",
    "non_anchor": "Non-anchor",
    "inside_support": "Inside support",
    "below_support": "Below support",
    "above_support": "Above support",
    "outside_support": "Outside support",
    "near_0_2m": "Near 0–2 m",
    "anchors": "Anchors",
}
CORE_REGION_ORDER = ("all", "non_anchor", "outside_support", "near_0_2m")
METHOD_ORDER_HINTS = (
    "any2full",
    "any2full_monotonic",
    "any2full_mono",
    "da3_median",
    "da3_ls",
    "da3_log_ls",
    "da3_huber",
    "da3_monotonic",
    "da3_ls_poisson",
    "da3_poisson",
    "da3_oasis",
)
FAILURE_THRESHOLDS = (5.0, 10.0, 20.0, 40.0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a scene-level, paired, comprehensive comparison of Any2Full "
            "and DA3 alignment variants."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="per_scene_metrics.csv")
    parser.add_argument(
        "--fits-csv",
        default="auto",
        help=(
            "per_scene_fits.csv, 'auto' to use a sibling file when present, "
            "or 'none' to disable coverage analysis (default: auto)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory (default: <CSV directory>/comprehensive_analysis)",
    )
    parser.add_argument(
        "--baseline",
        default="any2full",
        help="Baseline method id or exact method label (default: any2full)",
    )
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=100,
        help="Expected number of unique scenes; a mismatch is reported as a warning",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Paired bootstrap samples for 95%% confidence intervals",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--top-worst",
        type=int,
        default=10,
        help="Worst scenes retained per method and region",
    )
    return parser.parse_args()


def _finite_float(value: object, field: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number}: {field}={value!r} is not numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"Row {row_number}: {field}={value!r} is not finite")
    return number


def _slug(text: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    return result or "method"


def load_metric_rows(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"scene", "region", "absrel_pct"}
        missing = required - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if "method" not in fields and "method_label" not in fields:
            raise ValueError(f"{path} needs a method or method_label column")

        rows: List[dict] = []
        seen = set()
        labels_by_method: Dict[str, str] = {}
        for row_number, raw in enumerate(reader, start=2):
            scene = (raw.get("scene") or "").strip()
            region = (raw.get("region") or "").strip()
            label = (raw.get("method_label") or raw.get("method") or "").strip()
            method = (raw.get("method") or _slug(label)).strip()
            if not scene or not region or not method or not label:
                raise ValueError(
                    f"Row {row_number}: scene, region, method, and method label must be non-empty"
                )
            key = (scene, method, region)
            if key in seen:
                raise ValueError(f"Duplicate scene/method/region row: {key}")
            seen.add(key)
            previous = labels_by_method.setdefault(method, label)
            if previous != label:
                raise ValueError(
                    f"Method id {method!r} has conflicting labels: {previous!r}, {label!r}"
                )

            parsed = {
                "scene": scene,
                "method": method,
                "method_label": label,
                "region": region,
                "region_label": (raw.get("region_label") or REGION_LABELS.get(region, region)),
            }
            for metric in METRICS:
                value = raw.get(metric)
                parsed[metric] = (
                    _finite_float(value, metric, row_number)
                    if value not in (None, "")
                    else float("nan")
                )
            n_value = raw.get("n")
            parsed["n"] = (
                int(round(_finite_float(n_value, "n", row_number)))
                if n_value not in (None, "")
                else 0
            )
            if parsed["n"] < 0:
                raise ValueError(f"Row {row_number}: n cannot be negative")
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} contains no metric rows")
    return rows


def load_fit_rows(path: Path) -> Dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "scene" not in fields:
            raise ValueError(f"{path} needs a scene column")
        output: Dict[str, dict] = {}
        for row_number, raw in enumerate(reader, start=2):
            scene = (raw.get("scene") or "").strip()
            if not scene:
                raise ValueError(f"Row {row_number}: empty scene in {path}")
            if scene in output:
                raise ValueError(f"Duplicate fit row for scene {scene!r}")
            row = {"scene": scene}
            for key, value in raw.items():
                if key == "scene":
                    continue
                try:
                    row[key] = float(value) if value not in (None, "") else float("nan")
                except ValueError:
                    row[key] = value
            output[scene] = row
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ordered_methods(rows: Sequence[dict]) -> List[str]:
    present = {row["method"] for row in rows}
    labels = {row["method"]: row["method_label"] for row in rows}

    def order_key(method: str) -> Tuple[int, str]:
        try:
            hinted = METHOD_ORDER_HINTS.index(method)
        except ValueError:
            hinted = len(METHOD_ORDER_HINTS)
        return hinted, labels[method].casefold()

    return sorted(present, key=order_key)


def ordered_regions(rows: Sequence[dict]) -> List[str]:
    present = {row["region"] for row in rows}
    known = [region for region in REGION_ORDER if region in present]
    return known + sorted(present - set(known))


def resolve_baseline(rows: Sequence[dict], selector: str) -> str:
    methods = ordered_methods(rows)
    labels = {row["method"]: row["method_label"] for row in rows}
    exact_ids = [method for method in methods if method.casefold() == selector.casefold()]
    exact_labels = [
        method for method in methods if labels[method].casefold() == selector.casefold()
    ]
    matches = list(dict.fromkeys(exact_ids + exact_labels))
    if len(matches) == 1:
        return matches[0]
    if not matches and selector.casefold() == "any2full":
        matches = [
            method
            for method in methods
            if labels[method].casefold() == "any2full"
            or method.casefold() == "any2full"
        ]
    if len(matches) != 1:
        choices = ", ".join(f"{method}={labels[method]!r}" for method in methods)
        raise ValueError(f"Could not resolve baseline {selector!r}. Available: {choices}")
    return matches[0]


def group_rows(rows: Sequence[dict]) -> Dict[Tuple[str, str], List[dict]]:
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["region"])].append(row)
    return grouped


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def summarize(rows: Sequence[dict], methods: Sequence[str], regions: Sequence[str]) -> List[dict]:
    grouped = group_rows(rows)
    output: List[dict] = []
    for region in regions:
        for method in methods:
            subset = grouped.get((method, region), [])
            if not subset:
                continue
            label = subset[0]["method_label"]
            row: Dict[str, object] = {
                "method": method,
                "method_label": label,
                "region": region,
                "region_label": subset[0]["region_label"],
                "scene_count": len(subset),
                "pixel_count": int(sum(item["n"] for item in subset)),
            }
            for metric in METRICS:
                values = np.asarray([item[metric] for item in subset], dtype=float)
                values = values[np.isfinite(values)]
                row[f"macro_mean_{metric}"] = float(np.mean(values))
                row[f"macro_median_{metric}"] = float(np.median(values))
                row[f"macro_std_{metric}"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
                row[f"p10_{metric}"] = _quantile(values, 0.10)
                row[f"p25_{metric}"] = _quantile(values, 0.25)
                row[f"p75_{metric}"] = _quantile(values, 0.75)
                row[f"p90_{metric}"] = _quantile(values, 0.90)

            weights = np.asarray([item["n"] for item in subset], dtype=float)
            usable_weights = np.isfinite(weights) & (weights > 0)
            if usable_weights.any():
                total = float(weights[usable_weights].sum())
                for metric in ("mae_m", "absrel_pct", "bias_m"):
                    values = np.asarray([item[metric] for item in subset], dtype=float)
                    mask = usable_weights & np.isfinite(values)
                    row[f"pooled_{metric}"] = (
                        float(np.average(values[mask], weights=weights[mask]))
                        if mask.any()
                        else float("nan")
                    )
                for metric in ("rmse_m", "rmsrel_pct"):
                    values = np.asarray([item[metric] for item in subset], dtype=float)
                    mask = usable_weights & np.isfinite(values)
                    row[f"pooled_{metric}"] = (
                        float(np.sqrt(np.average(values[mask] ** 2, weights=weights[mask])))
                        if mask.any()
                        else float("nan")
                    )
                row["pooled_scene_pixel_denominator"] = int(total)
            else:
                for metric in METRICS:
                    row[f"pooled_{metric}"] = float("nan")
                row["pooled_scene_pixel_denominator"] = 0

            bias_values = np.asarray([abs(item["bias_m"]) for item in subset], dtype=float)
            row["macro_mean_abs_bias_m"] = float(np.mean(bias_values))
            output.append(row)
    return output


def _bootstrap_mean_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or samples <= 0:
        only = float(values.mean())
        return only, only
    # Small chunks avoid a large temporary matrix when users request many samples.
    means: List[np.ndarray] = []
    remaining = samples
    while remaining:
        chunk = min(2000, remaining)
        indices = rng.integers(0, values.size, size=(chunk, values.size))
        means.append(values[indices].mean(axis=1))
        remaining -= chunk
    distribution = np.concatenate(means)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def paired_comparisons(
    rows: Sequence[dict],
    methods: Sequence[str],
    regions: Sequence[str],
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> List[dict]:
    lookup = {
        (row["scene"], row["method"], row["region"]): row for row in rows
    }
    scenes_by_group: Dict[Tuple[str, str], set] = defaultdict(set)
    labels = {row["method"]: row["method_label"] for row in rows}
    for row in rows:
        scenes_by_group[(row["method"], row["region"])].add(row["scene"])

    rng = np.random.default_rng(seed)
    output: List[dict] = []
    for region in regions:
        baseline_scenes = scenes_by_group.get((baseline, region), set())
        for method in methods:
            if method == baseline:
                continue
            scenes = sorted(baseline_scenes & scenes_by_group.get((method, region), set()))
            if not scenes:
                continue
            base_absrel = np.asarray(
                [lookup[(scene, baseline, region)]["absrel_pct"] for scene in scenes]
            )
            candidate_absrel = np.asarray(
                [lookup[(scene, method, region)]["absrel_pct"] for scene in scenes]
            )
            base_rmse = np.asarray(
                [lookup[(scene, baseline, region)]["rmse_m"] for scene in scenes]
            )
            candidate_rmse = np.asarray(
                [lookup[(scene, method, region)]["rmse_m"] for scene in scenes]
            )
            base_abs_bias = np.asarray(
                [abs(lookup[(scene, baseline, region)]["bias_m"]) for scene in scenes]
            )
            candidate_abs_bias = np.asarray(
                [abs(lookup[(scene, method, region)]["bias_m"]) for scene in scenes]
            )
            absrel_delta = base_absrel - candidate_absrel
            rmse_delta = base_rmse - candidate_rmse
            abs_bias_delta = base_abs_bias - candidate_abs_bias
            tolerance = 1e-12
            wins = int(np.sum(absrel_delta > tolerance))
            losses = int(np.sum(absrel_delta < -tolerance))
            ties = len(scenes) - wins - losses
            ci_low, ci_high = _bootstrap_mean_ci(absrel_delta, bootstrap_samples, rng)

            relative_reduction = np.divide(
                absrel_delta,
                base_absrel,
                out=np.full_like(absrel_delta, np.nan),
                where=base_absrel > 0,
            ) * 100.0
            ratios = np.divide(
                base_absrel,
                candidate_absrel,
                out=np.full_like(base_absrel, np.nan),
                where=candidate_absrel > 0,
            )
            if wilcoxon is None or np.all(np.abs(absrel_delta) <= tolerance):
                p_value = 1.0 if np.all(np.abs(absrel_delta) <= tolerance) else float("nan")
            else:
                try:
                    p_value = float(
                        wilcoxon(
                            absrel_delta,
                            zero_method="wilcox",
                            alternative="two-sided",
                            method="auto",
                        ).pvalue
                    )
                except ValueError:
                    p_value = float("nan")

            output.append(
                {
                    "baseline": baseline,
                    "baseline_label": labels[baseline],
                    "method": method,
                    "method_label": labels[method],
                    "region": region,
                    "region_label": REGION_LABELS.get(region, region),
                    "paired_scene_count": len(scenes),
                    "mean_absrel_improvement_pp": float(np.mean(absrel_delta)),
                    "median_absrel_improvement_pp": float(np.median(absrel_delta)),
                    "ci95_low_absrel_improvement_pp": ci_low,
                    "ci95_high_absrel_improvement_pp": ci_high,
                    "mean_relative_absrel_reduction_pct": float(
                        np.nanmean(relative_reduction)
                    ),
                    "median_baseline_to_candidate_error_ratio": float(np.nanmedian(ratios)),
                    "absrel_wins": wins,
                    "absrel_ties": ties,
                    "absrel_losses": losses,
                    "absrel_win_rate_pct": 100.0 * wins / len(scenes),
                    "absrel_nonloss_rate_pct": 100.0 * (wins + ties) / len(scenes),
                    "mean_rmse_improvement_m": float(np.mean(rmse_delta)),
                    "median_rmse_improvement_m": float(np.median(rmse_delta)),
                    "rmse_win_rate_pct": float(100.0 * np.mean(rmse_delta > tolerance)),
                    "mean_abs_bias_improvement_m": float(np.mean(abs_bias_delta)),
                    "abs_bias_win_rate_pct": float(
                        100.0 * np.mean(abs_bias_delta > tolerance)
                    ),
                    "wilcoxon_p_value": p_value,
                }
            )

    # Holm correction controls family-wise error across all reported paired tests.
    finite_indices = [
        index for index, row in enumerate(output) if np.isfinite(row["wilcoxon_p_value"])
    ]
    finite_indices.sort(key=lambda index: output[index]["wilcoxon_p_value"])
    running = 0.0
    total = len(finite_indices)
    for rank, index in enumerate(finite_indices):
        raw = float(output[index]["wilcoxon_p_value"])
        adjusted = min(1.0, raw * (total - rank))
        running = max(running, adjusted)
        output[index]["holm_adjusted_p_value"] = running
    for row in output:
        row.setdefault("holm_adjusted_p_value", float("nan"))
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and math.isclose(
            float(values[order[end]]),
            float(values[order[position]]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            end += 1
        average_rank = 0.5 * ((position + 1) + end)
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def scene_rankings(
    rows: Sequence[dict], methods: Sequence[str], regions: Sequence[str]
) -> List[dict]:
    lookup = {
        (row["scene"], row["method"], row["region"]): row for row in rows
    }
    labels = {row["method"]: row["method_label"] for row in rows}
    scenes_by_group: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in rows:
        scenes_by_group[(row["method"], row["region"])].add(row["scene"])

    output: List[dict] = []
    for region in regions:
        scene_sets = [scenes_by_group.get((method, region), set()) for method in methods]
        complete_scenes = sorted(set.intersection(*scene_sets)) if scene_sets else []
        rank_sum = {method: 0.0 for method in methods}
        win_share = {method: 0.0 for method in methods}
        strict_wins = {method: 0 for method in methods}
        for scene in complete_scenes:
            values = np.asarray(
                [lookup[(scene, method, region)]["absrel_pct"] for method in methods]
            )
            ranks = _average_ranks(values)
            minimum = float(np.min(values))
            winner_indices = np.flatnonzero(np.isclose(values, minimum, rtol=1e-12, atol=1e-12))
            share = 1.0 / len(winner_indices)
            for index, method in enumerate(methods):
                rank_sum[method] += float(ranks[index])
            for index in winner_indices:
                win_share[methods[int(index)]] += share
            if len(winner_indices) == 1:
                strict_wins[methods[int(winner_indices[0])]] += 1

        count = len(complete_scenes)
        for method in methods:
            output.append(
                {
                    "method": method,
                    "method_label": labels[method],
                    "region": region,
                    "region_label": REGION_LABELS.get(region, region),
                    "complete_scene_count": count,
                    "mean_rank": rank_sum[method] / count if count else float("nan"),
                    "fractional_scene_wins": win_share[method],
                    "scene_win_share_pct": 100.0 * win_share[method] / count if count else 0.0,
                    "strict_scene_wins": strict_wins[method],
                    "strict_scene_win_rate_pct": (
                        100.0 * strict_wins[method] / count if count else 0.0
                    ),
                }
            )
    return output


def threshold_failure_rates(
    rows: Sequence[dict], methods: Sequence[str], regions: Sequence[str]
) -> List[dict]:
    grouped = group_rows(rows)
    output: List[dict] = []
    for region in regions:
        for method in methods:
            subset = grouped.get((method, region), [])
            if not subset:
                continue
            values = np.asarray([row["absrel_pct"] for row in subset], dtype=float)
            for threshold in FAILURE_THRESHOLDS:
                failures = int(np.sum(values >= threshold))
                output.append(
                    {
                        "method": method,
                        "method_label": subset[0]["method_label"],
                        "region": region,
                        "region_label": subset[0]["region_label"],
                        "threshold_absrel_pct": threshold,
                        "scene_count": len(values),
                        "failure_scene_count": failures,
                        "failure_scene_rate_pct": 100.0 * failures / len(values),
                    }
                )
    return output


def worst_scenes(
    rows: Sequence[dict], methods: Sequence[str], regions: Sequence[str], top_n: int
) -> List[dict]:
    grouped = group_rows(rows)
    output: List[dict] = []
    for region in regions:
        for method in methods:
            subset = sorted(
                grouped.get((method, region), []),
                key=lambda row: row["absrel_pct"],
                reverse=True,
            )
            for rank, row in enumerate(subset[:top_n], start=1):
                output.append(
                    {
                        "method": method,
                        "method_label": row["method_label"],
                        "region": region,
                        "region_label": row["region_label"],
                        "worst_rank": rank,
                        "scene": row["scene"],
                        "absrel_pct": row["absrel_pct"],
                        "rmse_m": row["rmse_m"],
                        "mae_m": row["mae_m"],
                        "bias_m": row["bias_m"],
                    }
                )
    return output


def coverage_correlations(
    rows: Sequence[dict], fit_rows: Mapping[str, dict], regions: Sequence[str]
) -> List[dict]:
    predictors = (
        "lidar_span_m",
        "affine_design_condition",
        "anchor_count",
        "lidar_min_m",
        "lidar_max_m",
    )
    grouped = group_rows(rows)
    output: List[dict] = []
    for (method, region), subset in grouped.items():
        if region not in regions:
            continue
        for predictor in predictors:
            pairs = [
                (fit_rows[row["scene"]].get(predictor), row["absrel_pct"])
                for row in subset
                if row["scene"] in fit_rows
            ]
            pairs = [
                (float(x), float(y))
                for x, y in pairs
                if isinstance(x, (int, float)) and np.isfinite(x) and np.isfinite(y)
            ]
            x = np.asarray([pair[0] for pair in pairs])
            y = np.asarray([pair[1] for pair in pairs])
            if len(pairs) >= 3 and np.unique(x).size >= 2 and spearmanr is not None:
                result = spearmanr(x, y)
                rho, p_value = float(result.statistic), float(result.pvalue)
            else:
                rho, p_value = float("nan"), float("nan")
            output.append(
                {
                    "method": method,
                    "method_label": subset[0]["method_label"],
                    "region": region,
                    "region_label": subset[0]["region_label"],
                    "predictor": predictor,
                    "paired_scene_count": len(pairs),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    return output


def _method_colors(methods: Sequence[str], labels: Mapping[str, str]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    fallback = iter(plt.get_cmap("tab20").colors)
    for method in methods:
        label = labels[method].casefold()
        if label == "any2full":
            color = "#0072B2"
        elif label.startswith("any2full"):
            color = "#56B4E9"
        elif "median" in label:
            color = "#009E73"
        elif "poisson" in label:
            color = "#D55E00"
        elif "oasis" in label:
            color = "#CC79A7"
        elif "monotonic" in label:
            color = "#00A6A6"
        elif "log" in label:
            color = "#E6AB02"
        elif "huber" in label:
            color = "#7B3294"
        elif "affine" in label or method.endswith("_ls"):
            color = "#E69F00"
        else:
            color = matplotlib.colors.to_hex(next(fallback))
        output[method] = color
    return output


def _short_label(label: str) -> str:
    replacements = {
        "Any2Full + monotonic recalibration": "A2F + monotonic",
        "DA3 + median scale": "DA3 + median",
        "DA3 + affine LS": "DA3 + LS",
        "DA3 + positive log-LS": "DA3 + log-LS",
        "DA3 + monotonic recalibration": "DA3 + monotonic",
        "DA3 + affine LS + Poisson": "DA3 + LS + Poisson",
        "DA3 + alignment + OASIS": "DA3 + OASIS",
    }
    return replacements.get(label, label)


def _is_da3(method: str, label: str) -> bool:
    return method.casefold().startswith("da3") or label.casefold().startswith("da3")


def _summary_lookup(summary_rows: Sequence[dict]) -> Dict[Tuple[str, str], dict]:
    return {(row["method"], row["region"]): row for row in summary_rows}


def _paired_lookup(paired_rows: Sequence[dict]) -> Dict[Tuple[str, str], dict]:
    return {(row["method"], row["region"]): row for row in paired_rows}


def _ranking_lookup(ranking_rows: Sequence[dict]) -> Dict[Tuple[str, str], dict]:
    return {(row["method"], row["region"]): row for row in ranking_rows}


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    colorbar_label: str,
    cmap: str,
    fmt: str,
    norm: Optional[Normalize] = None,
) -> None:
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontweight="bold", loc="left")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(colorbar_label)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            if norm is not None:
                normalized = float(norm(value))
            else:
                vmin, vmax = image.get_clim()
                normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = "white" if normalized > 0.58 else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=color)


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_overview(
    path: Path,
    rows: Sequence[dict],
    summary_rows: Sequence[dict],
    paired_rows: Sequence[dict],
    ranking_rows: Sequence[dict],
    methods: Sequence[str],
    regions: Sequence[str],
    baseline: str,
    dpi: int,
) -> None:
    labels = {row["method"]: row["method_label"] for row in rows}
    short_labels = [_short_label(labels[method]) for method in methods]
    region_labels = [REGION_LABELS.get(region, region) for region in regions]
    summary_lookup = _summary_lookup(summary_rows)
    paired_lookup = _paired_lookup(paired_rows)
    ranking_lookup = _ranking_lookup(ranking_rows)
    colors = _method_colors(methods, labels)

    absrel = np.asarray(
        [
            [summary_lookup.get((method, region), {}).get("macro_mean_absrel_pct", np.nan)
             for region in regions]
            for method in methods
        ]
    )
    positive = absrel[np.isfinite(absrel) & (absrel > 0)]
    norm = None
    if positive.size and positive.max() / positive.min() > 8:
        norm = LogNorm(vmin=max(0.1, float(positive.min())), vmax=float(positive.max()))

    candidates = [method for method in methods if method != baseline]
    paired_matrix = np.asarray(
        [
            [paired_lookup.get((method, region), {}).get("absrel_win_rate_pct", np.nan)
             for region in regions]
            for method in candidates
        ]
    )

    fig = plt.figure(figsize=(22, 15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.22, 1.0), height_ratios=(1.08, 1.0))
    ax1 = fig.add_subplot(grid[0, 0])
    _annotated_heatmap(
        ax1,
        absrel,
        short_labels,
        region_labels,
        "A. Macro mean AbsRel — every scene has equal weight",
        "AbsRel (%)",
        "magma_r",
        ".1f",
        norm,
    )

    ax2 = fig.add_subplot(grid[0, 1])
    _annotated_heatmap(
        ax2,
        paired_matrix,
        [_short_label(labels[method]) for method in candidates],
        region_labels,
        f"B. Paired scene win rate vs {labels[baseline]}",
        "Scenes won (%)",
        "RdYlGn",
        ".0f",
        Normalize(vmin=0, vmax=100),
    )

    ax3 = fig.add_subplot(grid[1, 0])
    focus_region = "outside_support" if "outside_support" in regions else regions[0]
    grouped = group_rows(rows)
    for method in methods:
        values = np.sort(
            np.asarray(
                [row["absrel_pct"] for row in grouped.get((method, focus_region), [])],
                dtype=float,
            )
        )
        if values.size:
            percentiles = 100.0 * (np.arange(values.size) + 0.5) / values.size
            ax3.plot(
                percentiles,
                values,
                label=_short_label(labels[method]),
                color=colors[method],
                linewidth=2.0,
                marker="o" if values.size < 3 else None,
                markersize=5,
            )
    ax3.set_yscale("log")
    ax3.set_xlabel("Scene percentile (sorted independently per method)")
    ax3.set_ylabel("AbsRel (%) — log scale")
    ax3.set_title(
        f"C. Full error distribution — {REGION_LABELS.get(focus_region, focus_region)}",
        fontweight="bold",
        loc="left",
    )
    ax3.grid(True, which="both", axis="y", alpha=0.25)
    ax3.legend(fontsize=8, ncol=2, loc="upper left")

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.axis("off")
    core = [region for region in CORE_REGION_ORDER if region in regions]
    table_data = []
    for region in core:
        present = [
            method
            for method in methods
            if (method, region) in summary_lookup and (method, region) in ranking_lookup
        ]
        if not present:
            continue
        winner = min(
            present,
            key=lambda method: summary_lookup[(method, region)]["macro_mean_absrel_pct"],
        )
        summary = summary_lookup[(winner, region)]
        rank = ranking_lookup[(winner, region)]
        paired = paired_lookup.get((winner, region))
        comparison = (
            "baseline"
            if winner == baseline
            else f"{paired['absrel_win_rate_pct']:.0f}% / {paired['mean_absrel_improvement_pp']:+.1f} pp"
            if paired
            else "n/a"
        )
        table_data.append(
            [
                REGION_LABELS.get(region, region),
                _short_label(labels[winner]),
                f"{summary['macro_mean_absrel_pct']:.2f}%",
                f"{summary['macro_median_absrel_pct']:.2f}%",
                f"{rank['mean_rank']:.2f}",
                comparison,
            ]
        )
    headers = [
        "Region",
        "Lowest mean",
        "Mean",
        "Median",
        "Mean rank",
        "Win rate / Δ vs A2F",
    ]
    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.18, 0.23, 0.11, 0.11, 0.11, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.65)
    for column in range(len(headers)):
        table[(0, column)].set_facecolor("#E8EEF3")
        table[(0, column)].set_text_props(weight="bold")
    ax4.set_title(
        "D. Region champions (descriptive, not a single universal winner)",
        fontweight="bold",
        loc="left",
        pad=12,
    )

    scene_count = len({row["scene"] for row in rows})
    fig.suptitle(
        f"Any2Full vs DA3 alignment — {scene_count} matched iBims scene(s)",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.006,
        "Lower error is better. Positive Δ means the candidate improves on Any2Full. "
        "Macro means and paired wins are primary; pooled metrics are secondary.",
        ha="center",
        fontsize=10,
    )
    _save_figure(fig, path, dpi)


def plot_distributions(
    path: Path,
    rows: Sequence[dict],
    methods: Sequence[str],
    regions: Sequence[str],
    dpi: int,
) -> None:
    labels = {row["method"]: row["method_label"] for row in rows}
    colors = _method_colors(methods, labels)
    grouped = group_rows(rows)
    focus_regions = [region for region in CORE_REGION_ORDER if region in regions]
    if not focus_regions:
        focus_regions = list(regions[:4])
    rows_count = int(math.ceil(len(focus_regions) / 2))
    fig, axes = plt.subplots(rows_count, 2, figsize=(19, 5.7 * rows_count), squeeze=False)
    for ax, region in zip(axes.ravel(), focus_regions):
        datasets = [
            np.asarray([row["absrel_pct"] for row in grouped.get((method, region), [])])
            for method in methods
        ]
        positions = np.arange(1, len(methods) + 1)
        box = ax.boxplot(
            datasets,
            positions=positions,
            vert=False,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.45},
        )
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.78)
        ax.set_yticks(positions)
        ax.set_yticklabels([_short_label(labels[method]) for method in methods])
        finite_positive = np.concatenate([data[data > 0] for data in datasets if data.size])
        if finite_positive.size and finite_positive.max() / finite_positive.min() > 10:
            ax.set_xscale("log")
            scale_note = "log scale"
        else:
            scale_note = "linear scale"
        ax.set_xlabel("Per-scene AbsRel (%)")
        ax.set_title(
            f"{REGION_LABELS.get(region, region)} — {scale_note}",
            fontweight="bold",
            loc="left",
        )
        ax.grid(True, which="both", axis="x", alpha=0.25)
    for ax in axes.ravel()[len(focus_regions):]:
        ax.axis("off")
    fig.suptitle(
        "Per-scene AbsRel distributions: median, IQR, mean diamond, and outliers",
        fontsize=16,
        fontweight="bold",
    )
    _save_figure(fig, path, dpi)


def plot_paired_scatter(
    path: Path,
    rows: Sequence[dict],
    summary_rows: Sequence[dict],
    paired_rows: Sequence[dict],
    methods: Sequence[str],
    regions: Sequence[str],
    baseline: str,
    dpi: int,
) -> None:
    labels = {row["method"]: row["method_label"] for row in rows}
    summary_lookup = _summary_lookup(summary_rows)
    paired_lookup = _paired_lookup(paired_rows)
    raw_lookup = {
        (row["scene"], row["method"], row["region"]): row for row in rows
    }
    scenes_by_group: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in rows:
        scenes_by_group[(row["method"], row["region"])].add(row["scene"])
    focus_regions = [region for region in CORE_REGION_ORDER if region in regions]
    if not focus_regions:
        focus_regions = list(regions[:4])
    rows_count = int(math.ceil(len(focus_regions) / 2))
    fig, axes = plt.subplots(rows_count, 2, figsize=(15, 6.7 * rows_count), squeeze=False)
    for ax, region in zip(axes.ravel(), focus_regions):
        candidates = [
            method
            for method in methods
            if method != baseline
            and _is_da3(method, labels[method])
            and (method, region) in summary_lookup
        ]
        if not candidates:
            candidates = [
                method
                for method in methods
                if method != baseline and (method, region) in summary_lookup
            ]
        if not candidates:
            ax.axis("off")
            continue
        champion = min(
            candidates,
            key=lambda method: summary_lookup[(method, region)]["macro_mean_absrel_pct"],
        )
        scenes = sorted(
            scenes_by_group[(baseline, region)] & scenes_by_group[(champion, region)]
        )
        x = np.asarray(
            [raw_lookup[(scene, baseline, region)]["absrel_pct"] for scene in scenes]
        )
        y = np.asarray(
            [raw_lookup[(scene, champion, region)]["absrel_pct"] for scene in scenes]
        )
        better = y < x
        ax.scatter(x[better], y[better], c="#009E73", s=34, alpha=0.72, label="DA3 wins")
        ax.scatter(x[~better], y[~better], c="#D55E00", s=34, alpha=0.72, label="A2F wins/ties")
        positive = np.concatenate((x[x > 0], y[y > 0]))
        if positive.size:
            lower = max(0.01, float(positive.min()) * 0.75)
            upper = float(positive.max()) * 1.35
            ax.plot([lower, upper], [lower, upper], "--", color="#555555", linewidth=1)
            if upper / lower > 12:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        paired = paired_lookup.get((champion, region), {})
        annotation = (
            f"n = {len(scenes)}\n"
            f"win rate = {paired.get('absrel_win_rate_pct', float('nan')):.1f}%\n"
            f"mean Δ = {paired.get('mean_absrel_improvement_pp', float('nan')):+.2f} pp\n"
            f"95% CI = [{paired.get('ci95_low_absrel_improvement_pp', float('nan')):+.2f}, "
            f"{paired.get('ci95_high_absrel_improvement_pp', float('nan')):+.2f}]"
        )
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
            fontsize=9,
        )
        deltas = np.abs(x - y)
        for index in np.argsort(deltas)[-3:]:
            ax.annotate(
                scenes[int(index)],
                (x[index], y[index]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )
        ax.set_xlabel(f"{labels[baseline]} AbsRel (%)")
        ax.set_ylabel(f"{_short_label(labels[champion])} AbsRel (%)")
        ax.set_title(
            f"{REGION_LABELS.get(region, region)}: best DA3 candidate = {_short_label(labels[champion])}",
            fontweight="bold",
            loc="left",
        )
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(loc="lower right", fontsize=8)
    for ax in axes.ravel()[len(focus_regions):]:
        ax.axis("off")
    fig.suptitle(
        "Paired scene-by-scene comparison against Any2Full",
        fontsize=16,
        fontweight="bold",
    )
    _save_figure(fig, path, dpi)


def plot_metric_dashboard(
    path: Path,
    rows: Sequence[dict],
    summary_rows: Sequence[dict],
    methods: Sequence[str],
    regions: Sequence[str],
    dpi: int,
) -> None:
    labels = {row["method"]: row["method_label"] for row in rows}
    lookup = _summary_lookup(summary_rows)
    specs = (
        ("macro_mean_absrel_pct", "Macro mean AbsRel", "%", "magma_r", False),
        ("pooled_absrel_pct", "Pixel-pooled AbsRel", "%", "magma_r", False),
        ("macro_mean_rmse_m", "Macro mean RMSE", "m", "viridis_r", False),
        ("macro_mean_bias_m", "Macro mean signed bias", "m", "coolwarm", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), constrained_layout=True)
    for ax, (key, title, unit, cmap, signed) in zip(axes.ravel(), specs):
        matrix = np.asarray(
            [
                [lookup.get((method, region), {}).get(key, np.nan) for region in regions]
                for method in methods
            ]
        )
        if signed:
            max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
            norm: Optional[Normalize] = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        else:
            norm = None
        _annotated_heatmap(
            ax,
            matrix,
            [_short_label(labels[method]) for method in methods],
            [REGION_LABELS.get(region, region) for region in regions],
            title,
            unit,
            cmap,
            ".2f",
            norm,
        )
    fig.suptitle(
        "Metric dashboard: macro and pooled views answer different questions",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Macro: each scene contributes equally. Pooled: each valid pixel contributes equally. "
        "Large scenes can dominate pooled results.",
        ha="center",
        fontsize=10,
    )
    _save_figure(fig, path, dpi)


def plot_coverage(
    path: Path,
    rows: Sequence[dict],
    fit_rows: Mapping[str, dict],
    summary_rows: Sequence[dict],
    paired_rows: Sequence[dict],
    methods: Sequence[str],
    regions: Sequence[str],
    baseline: str,
    dpi: int,
) -> bool:
    region = "outside_support" if "outside_support" in regions else regions[0]
    available_predictors = {
        key
        for fit in fit_rows.values()
        for key, value in fit.items()
        if key != "scene" and isinstance(value, (int, float))
    }
    if "lidar_span_m" not in available_predictors:
        warnings.warn("Coverage plot skipped: per_scene_fits.csv has no lidar_span_m")
        return False
    labels = {row["method"]: row["method_label"] for row in rows}
    colors = _method_colors(methods, labels)
    summary_lookup = _summary_lookup(summary_rows)
    paired_lookup = _paired_lookup(paired_rows)
    raw_lookup = {
        (row["scene"], row["method"], row["region"]): row for row in rows
    }
    scenes_by_group: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in rows:
        scenes_by_group[(row["method"], row["region"])].add(row["scene"])
    ranked_methods = sorted(
        [method for method in methods if (method, region) in summary_lookup],
        key=lambda method: summary_lookup[(method, region)]["macro_mean_absrel_pct"],
    )
    plotted_methods = list(dict.fromkeys([baseline] + ranked_methods[:4]))
    candidates = [
        method
        for method in ranked_methods
        if method != baseline and _is_da3(method, labels[method])
    ]
    if not candidates:
        candidates = [method for method in ranked_methods if method != baseline]
    champion = candidates[0] if candidates else baseline

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()
    for method in plotted_methods:
        scenes = sorted(
            scene
            for scene in scenes_by_group[(method, region)]
            if scene in fit_rows and np.isfinite(fit_rows[scene].get("lidar_span_m", np.nan))
        )
        x = np.asarray([fit_rows[scene]["lidar_span_m"] for scene in scenes], dtype=float)
        y = np.asarray([raw_lookup[(scene, method, region)]["absrel_pct"] for scene in scenes])
        ax1.scatter(
            x,
            y,
            s=24,
            alpha=0.52,
            color=colors[method],
            label=_short_label(labels[method]),
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("LiDAR anchor depth span (m)")
    ax1.set_ylabel("Outside-support AbsRel (%) — log scale")
    ax1.set_title("A. Error vs anchor-range coverage", fontweight="bold", loc="left")
    ax1.grid(True, which="both", alpha=0.2)
    ax1.legend(fontsize=8, ncol=2)

    common = sorted(
        scenes_by_group[(baseline, region)]
        & scenes_by_group[(champion, region)]
        & set(fit_rows)
    )
    x = np.asarray([fit_rows[scene].get("lidar_span_m", np.nan) for scene in common])
    improvement = np.asarray(
        [
            raw_lookup[(scene, baseline, region)]["absrel_pct"]
            - raw_lookup[(scene, champion, region)]["absrel_pct"]
            for scene in common
        ]
    )
    usable = np.isfinite(x) & np.isfinite(improvement)
    ax2.scatter(x[usable], improvement[usable], c=np.where(improvement[usable] >= 0, "#009E73", "#D55E00"), s=32, alpha=0.7)
    ax2.axhline(0, color="#444444", linewidth=1)
    if usable.sum() >= 3 and spearmanr is not None and np.unique(x[usable]).size >= 2:
        correlation = spearmanr(x[usable], improvement[usable])
        note = f"Spearman ρ = {correlation.statistic:+.2f}, p = {correlation.pvalue:.3g}"
    else:
        note = "Too few usable scenes for correlation"
    ax2.text(0.03, 0.96, note, transform=ax2.transAxes, va="top", fontsize=9)
    ax2.set_xlabel("LiDAR anchor depth span (m)")
    ax2.set_ylabel("AbsRel improvement vs Any2Full (pp)")
    ax2.set_title(
        f"B. Coverage vs {_short_label(labels[champion])} improvement",
        fontweight="bold",
        loc="left",
    )
    ax2.grid(True, alpha=0.2)

    valid_spans = np.asarray(
        [fit.get("lidar_span_m", np.nan) for fit in fit_rows.values()], dtype=float
    )
    valid_spans = valid_spans[np.isfinite(valid_spans)]
    if valid_spans.size >= 4:
        boundaries = np.unique(np.quantile(valid_spans, [0, 0.25, 0.5, 0.75, 1.0]))
    else:
        boundaries = np.unique(valid_spans)
    if boundaries.size >= 2:
        bin_labels = [f"{boundaries[i]:.2f}–{boundaries[i + 1]:.2f} m" for i in range(len(boundaries) - 1)]
        matrix = np.full((len(plotted_methods), len(bin_labels)), np.nan)
        for method_index, method in enumerate(plotted_methods):
            by_bin: Dict[int, List[float]] = defaultdict(list)
            for scene in scenes_by_group[(method, region)]:
                span = fit_rows.get(scene, {}).get("lidar_span_m", np.nan)
                if not np.isfinite(span):
                    continue
                bin_index = min(np.searchsorted(boundaries, span, side="right") - 1, len(boundaries) - 2)
                if bin_index >= 0:
                    by_bin[int(bin_index)].append(raw_lookup[(scene, method, region)]["absrel_pct"])
            for bin_index, values in by_bin.items():
                matrix[method_index, bin_index] = float(np.mean(values))
        _annotated_heatmap(
            ax3,
            matrix,
            [_short_label(labels[method]) for method in plotted_methods],
            bin_labels,
            "C. Mean error within anchor-span quartiles",
            "AbsRel (%)",
            "magma_r",
            ".1f",
        )
    else:
        ax3.text(0.5, 0.5, "Not enough distinct spans for bins", ha="center", va="center")
        ax3.axis("off")

    condition_key = "affine_design_condition"
    if condition_key in available_predictors:
        for method in plotted_methods:
            scenes = sorted(
                scene
                for scene in scenes_by_group[(method, region)]
                if scene in fit_rows
                and np.isfinite(fit_rows[scene].get(condition_key, np.nan))
                and fit_rows[scene].get(condition_key, 0) > 0
            )
            condition = np.asarray([fit_rows[scene][condition_key] for scene in scenes])
            error = np.asarray([raw_lookup[(scene, method, region)]["absrel_pct"] for scene in scenes])
            ax4.scatter(condition, error, s=24, alpha=0.52, color=colors[method], label=_short_label(labels[method]))
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_xlabel("Affine design condition number — log scale")
        ax4.set_ylabel("Outside-support AbsRel (%) — log scale")
        ax4.set_title("D. Ill-conditioning vs error", fontweight="bold", loc="left")
        ax4.grid(True, which="both", alpha=0.2)
    else:
        ax4.text(0.5, 0.5, "No affine_design_condition column", ha="center", va="center")
        ax4.axis("off")

    paired = paired_lookup.get((champion, region), {})
    fig.suptitle(
        f"Coverage diagnostics — {REGION_LABELS.get(region, region)}; "
        f"best DA3 mean: {_short_label(labels[champion])} "
        f"({paired.get('absrel_win_rate_pct', float('nan')):.0f}% paired wins)",
        fontsize=16,
        fontweight="bold",
    )
    _save_figure(fig, path, dpi)
    return True


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_escape(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def write_report(
    path: Path,
    input_csv: Path,
    rows: Sequence[dict],
    summary_rows: Sequence[dict],
    paired_rows: Sequence[dict],
    ranking_rows: Sequence[dict],
    methods: Sequence[str],
    regions: Sequence[str],
    baseline: str,
    expected_scenes: int,
    fits_path: Optional[Path],
) -> None:
    labels = {row["method"]: row["method_label"] for row in rows}
    summary_lookup = _summary_lookup(summary_rows)
    paired_lookup = _paired_lookup(paired_rows)
    ranking_lookup = _ranking_lookup(ranking_rows)
    scene_count = len({row["scene"] for row in rows})
    core = [region for region in CORE_REGION_ORDER if region in regions]
    champion_rows = []
    for region in core:
        present = [method for method in methods if (method, region) in summary_lookup]
        champion = min(
            present,
            key=lambda method: summary_lookup[(method, region)]["macro_mean_absrel_pct"],
        )
        summary = summary_lookup[(champion, region)]
        rank = ranking_lookup.get((champion, region), {})
        paired = paired_lookup.get((champion, region))
        champion_rows.append(
            [
                REGION_LABELS.get(region, region),
                labels[champion],
                f"{summary['macro_mean_absrel_pct']:.3f}%",
                f"{summary['macro_median_absrel_pct']:.3f}%",
                f"{summary['pooled_absrel_pct']:.3f}%",
                f"{rank.get('mean_rank', float('nan')):.2f}",
                "baseline"
                if champion == baseline
                else f"{paired['absrel_win_rate_pct']:.1f}%" if paired else "n/a",
                "—"
                if champion == baseline
                else f"{paired['mean_absrel_improvement_pp']:+.3f} pp" if paired else "n/a",
            ]
        )

    primary_region = "outside_support" if "outside_support" in regions else core[0]
    comparison_rows = []
    for method in methods:
        if method == baseline or (method, primary_region) not in paired_lookup:
            continue
        paired = paired_lookup[(method, primary_region)]
        summary = summary_lookup[(method, primary_region)]
        comparison_rows.append(
            [
                labels[method],
                f"{summary['macro_mean_absrel_pct']:.3f}%",
                f"{paired['mean_absrel_improvement_pp']:+.3f} pp",
                f"[{paired['ci95_low_absrel_improvement_pp']:+.3f}, {paired['ci95_high_absrel_improvement_pp']:+.3f}]",
                f"{paired['absrel_win_rate_pct']:.1f}%",
                f"{paired['absrel_wins']}/{paired['absrel_ties']}/{paired['absrel_losses']}",
                f"{paired['median_baseline_to_candidate_error_ratio']:.2f}×",
                f"{paired['holm_adjusted_p_value']:.3g}",
            ]
        )
    comparison_rows.sort(key=lambda row: float(row[1].rstrip("%")))

    completeness_lines = []
    for region in regions:
        counts = {
            method: summary_lookup.get((method, region), {}).get("scene_count", 0)
            for method in methods
        }
        minimum, maximum = min(counts.values()), max(counts.values())
        completeness_lines.append(
            f"- {REGION_LABELS.get(region, region)}: {minimum}–{maximum} scenes per method"
        )

    warning_text = (
        f"**Warning:** expected {expected_scenes} scenes but found {scene_count}. "
        "Treat this as a smoke/partial result, not the final benchmark."
        if scene_count != expected_scenes
        else f"The expected {expected_scenes} unique scenes are present."
    )
    report = f"""# Any2Full vs DA3 alignment — comprehensive comparison

Generated from `{input_csv}`.

## Validation

{warning_text}

- Unique scenes: **{scene_count}**
- Methods: **{len(methods)}**
- Regions: **{len(regions)}**
- Baseline: **{labels[baseline]}**
- Coverage diagnostics: **{'enabled from ' + str(fits_path) if fits_path else 'not available'}**

Scene completeness by region:

{chr(10).join(completeness_lines)}

## How to read the result

The primary statistics are the macro mean/median and paired scene win rate. Each
scene therefore has equal influence. Pixel-pooled values are reported as a
secondary view and can be dominated by scenes or regions containing more valid
pixels. Positive improvement means the candidate has lower error than
{labels[baseline]}. The bootstrap confidence interval is paired by scene. The
Wilcoxon p-value is Holm-adjusted across the reported method-region tests.

## Descriptive champion by operational region

{_markdown_table(
    ['Region', 'Lowest macro mean', 'Mean AbsRel', 'Median', 'Pooled', 'Mean rank', 'Win rate vs A2F', 'Mean Δ'],
    champion_rows,
)}

This table does **not** assert that one universal method wins every use case.
For deployment-oriented decisions, prioritize `non_anchor`, `outside_support`,
and `near_0_2m`; anchor-only accuracy can hide poor extrapolation.

## {REGION_LABELS.get(primary_region, primary_region)}: every candidate vs Any2Full

{_markdown_table(
    ['Candidate', 'Mean AbsRel', 'Mean Δ', 'Paired 95% CI', 'Win rate', 'W/T/L', 'Median error ratio', 'Holm p'],
    comparison_rows,
)}

## Produced figures

- `01_overview_dashboard.png`: mean error, paired wins, tail distribution, and champions.
- `02_absrel_distributions.png`: per-scene medians, IQRs, means, and outliers.
- `03_paired_vs_baseline.png`: direct scene-by-scene plots against Any2Full.
- `04_metric_dashboard.png`: macro AbsRel, pooled AbsRel, RMSE, and signed bias.
- `05_coverage_analysis.png`: range support and conditioning diagnostics, when available.

## Interpretation guardrails

1. A low pooled error does not prove consistency across scenes; check macro,
   median, win rate, and the distribution plot together.
2. A high anchor score does not prove good dense depth; anchors are the pixels
   used to align or constrain several methods.
3. `outside_support` directly tests extrapolation beyond the LiDAR anchor-depth
   band; `non_anchor` tests dense prediction away from the measurements.
4. Coverage correlations are descriptive associations, not causal proof and
   not an inference-time router evaluation.
5. Confirm that Any2Full and every DA3 method used the exact same V2.1 sparse
   LiDAR inputs before interpreting the benchmark.
"""
    path.write_text(textwrap.dedent(report).strip() + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    input_csv = arguments.csv.expanduser().resolve()
    output_directory = (
        arguments.out_dir.expanduser().resolve()
        if arguments.out_dir
        else input_csv.parent / "comprehensive_analysis"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    rows = load_metric_rows(input_csv)
    methods = ordered_methods(rows)
    regions = ordered_regions(rows)
    baseline = resolve_baseline(rows, arguments.baseline)
    labels = {row["method"]: row["method_label"] for row in rows}
    scenes = sorted({row["scene"] for row in rows})
    if len(scenes) != arguments.expected_scenes:
        warnings.warn(
            f"Expected {arguments.expected_scenes} scenes, found {len(scenes)}. "
            "The report will be marked as partial."
        )

    fits_path: Optional[Path]
    if arguments.fits_csv.casefold() == "none":
        fits_path = None
    elif arguments.fits_csv.casefold() == "auto":
        candidate = input_csv.parent / "per_scene_fits.csv"
        fits_path = candidate if candidate.is_file() else None
    else:
        fits_path = Path(arguments.fits_csv).expanduser().resolve()
        if not fits_path.is_file():
            raise FileNotFoundError(f"Fit CSV does not exist: {fits_path}")
    fit_rows = load_fit_rows(fits_path) if fits_path else {}

    summary_rows = summarize(rows, methods, regions)
    paired_rows = paired_comparisons(
        rows,
        methods,
        regions,
        baseline,
        arguments.bootstrap_samples,
        arguments.seed,
    )
    ranking_rows = scene_rankings(rows, methods, regions)
    failure_rows = threshold_failure_rates(rows, methods, regions)
    worst_rows = worst_scenes(rows, methods, regions, arguments.top_worst)
    correlation_rows = (
        coverage_correlations(rows, fit_rows, regions) if fit_rows else []
    )

    write_csv(output_directory / "summary_by_method_region.csv", summary_rows)
    write_csv(output_directory / "paired_vs_baseline.csv", paired_rows)
    write_csv(output_directory / "scene_rankings.csv", ranking_rows)
    write_csv(output_directory / "threshold_failure_rates.csv", failure_rows)
    write_csv(output_directory / "worst_scenes.csv", worst_rows)
    if correlation_rows:
        write_csv(output_directory / "coverage_correlations.csv", correlation_rows)

    plot_overview(
        output_directory / "01_overview_dashboard.png",
        rows,
        summary_rows,
        paired_rows,
        ranking_rows,
        methods,
        regions,
        baseline,
        arguments.dpi,
    )
    plot_distributions(
        output_directory / "02_absrel_distributions.png",
        rows,
        methods,
        regions,
        arguments.dpi,
    )
    plot_paired_scatter(
        output_directory / "03_paired_vs_baseline.png",
        rows,
        summary_rows,
        paired_rows,
        methods,
        regions,
        baseline,
        arguments.dpi,
    )
    plot_metric_dashboard(
        output_directory / "04_metric_dashboard.png",
        rows,
        summary_rows,
        methods,
        regions,
        arguments.dpi,
    )
    coverage_written = False
    if fit_rows:
        coverage_written = plot_coverage(
            output_directory / "05_coverage_analysis.png",
            rows,
            fit_rows,
            summary_rows,
            paired_rows,
            methods,
            regions,
            baseline,
            arguments.dpi,
        )

    write_report(
        output_directory / "comparison_report.md",
        input_csv,
        rows,
        summary_rows,
        paired_rows,
        ranking_rows,
        methods,
        regions,
        baseline,
        arguments.expected_scenes,
        fits_path if coverage_written else None,
    )
    metadata = {
        "input_csv": str(input_csv),
        "fits_csv": str(fits_path) if fits_path else None,
        "scene_count": len(scenes),
        "expected_scenes": arguments.expected_scenes,
        "method_count": len(methods),
        "methods": [{"id": method, "label": labels[method]} for method in methods],
        "regions": regions,
        "baseline": baseline,
        "bootstrap_samples": arguments.bootstrap_samples,
        "seed": arguments.seed,
    }
    (output_directory / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Loaded {len(rows)} rows: {len(scenes)} scenes, {len(methods)} methods, {len(regions)} regions")
    print(f"Baseline: {labels[baseline]}")
    print(f"Wrote comprehensive analysis to: {output_directory}")
    print("Primary decision files: 01_overview_dashboard.png and comparison_report.md")


if __name__ == "__main__":
    main()
