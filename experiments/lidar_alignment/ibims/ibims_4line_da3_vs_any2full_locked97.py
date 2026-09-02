#!/usr/bin/env python3
"""Locked iBims validation: DA3 versus Any2Full with identical four-line LiDAR.

The three pilot scenes used to choose the maximum-coverage placement are read
from the completed pilot protocol and excluded.  Every remaining iBims scene
is processed with the frozen 12.5/37.5/62.5/87.5 percent row placement.

Commands
--------
``prepare``
    Simulate the frozen four-line input from iBims GT at only the selected rays,
    save that exact float32 NPY, and use the reloaded same file to produce the
    DA3-SMALL + median + validated-Poisson result.  Matching RGB and sparse NPY
    directories are prepared for the existing Any2Full runner.  This stage is
    resumable and saves each completed scene immediately.

``evaluate``
    Audit all sparse-input hashes, require one native Any2Full prediction for
    every locked scene, score both methods on identical masks, calculate paired
    bootstrap intervals and Wilcoxon tests, and apply the predeclared decision
    rule.  The primary metric is RMSE outside the shared four-line support.

The locked decision requires at least a 5 percent RMSE reduction, a paired 95%
bootstrap interval entirely above zero, p < 0.05, and a scene win rate above
50 percent.  Positive paired improvement always means DA3 is better.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import binomtest, wilcoxon


VERSION = "1.0-locked"
ROW_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
PRACTICAL_RMSE_THRESHOLD_PCT = 5.0
PRIMARY_REGION = "outside_shared_four_line"
METHODS = {
    "da3": "DA3-SMALL + median + Poisson",
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

    prepare = subparsers.add_parser(
        "prepare",
        help="Prepare locked sparse inputs and DA3 predictions; safely resumable.",
    )
    prepare.add_argument("--da3-root", type=Path, required=True)
    prepare.add_argument("--pilot-output", type=Path, required=True)
    prepare.add_argument("--prepared-data-root", type=Path, required=True)
    prepare.add_argument("--expected-total-scenes", type=positive_int, default=100)
    prepare.add_argument("--expected-locked-scenes", type=positive_int, default=97)
    prepare.add_argument("--resume", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate the complete locked set and make the predeclared decision.",
    )
    evaluate.add_argument("--pilot-output", type=Path, required=True)
    evaluate.add_argument("--prepared-data-root", type=Path, required=True)
    evaluate.add_argument("--any2full-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--expected-locked-scenes", type=positive_int, default=97)
    evaluate.add_argument("--bootstrap-samples", type=positive_int, default=20000)
    evaluate.add_argument("--seed", type=int, default=20260902)
    evaluate.add_argument("--plot-max-depth-m", type=float, default=10.0)
    evaluate.add_argument("--plot-error-max-m", type=float, default=1.0)
    evaluate.add_argument("--skip-panels", action="store_true")

    subparsers.add_parser("self-test", help="Run numerical decision-rule tests.")
    return parser.parse_args()


def import_paired_module(script_dir: Path) -> Any:
    path = script_dir / "ibims_1line_vs_4line_da3_comparison.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required sibling evaluator is missing: {path}\n"
            "Keep this script beside ibims_1line_vs_4line_da3_comparison.py."
        )
    spec = importlib.util.spec_from_file_location("ibims_paired_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "load_ibims",
        "npy_path",
        "load_npy",
        "load_poisson",
        "call_poisson",
        "sanitize_one_line",
        "simulate_four_lines",
        "median_align",
        "metrics",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"Sibling evaluator is missing helpers: {missing}")
    return module


def resolve_directory(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv_optional(path: Path) -> list[dict[str, str]]:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def configuration_hash(payload: dict[str, Any]) -> str:
    source = {key: value for key, value in payload.items() if key != "configuration_sha256"}
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_protocol_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise KeyError(f"Protocol does not contain {label}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Protocol {label} no longer exists: {path}")
    return path


def pilot_context(
    pilot_output: Path,
    expected_pilot_scenes: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    pilot = read_json(pilot_output / "protocol.json")
    fractions = tuple(float(value) for value in pilot.get("max_coverage_row_fracs", []))
    if len(fractions) != 4 or not np.allclose(fractions, ROW_FRACTIONS):
        raise RuntimeError(
            "Pilot output is not the frozen 12.5/37.5/62.5/87.5% placement: "
            f"{fractions}"
        )
    pilot_scenes = [str(value) for value in pilot.get("scenes", [])]
    if (
        len(pilot_scenes) != expected_pilot_scenes
        or len(set(pilot_scenes)) != expected_pilot_scenes
    ):
        raise RuntimeError(
            f"Expected exactly {expected_pilot_scenes} pilot scenes in the protocol: "
            f"{pilot_scenes}"
        )
    reference_output = resolve_protocol_path(pilot.get("reference_output"), "reference_output")
    reference = read_json(reference_output / "protocol.json")
    return pilot, reference, pilot_scenes


def locked_scene_context(
    pilot_output: Path,
    expected_total: int,
    expected_locked: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], Path, Path, Path]:
    expected_pilot = expected_total - expected_locked
    if expected_pilot <= 0:
        raise ValueError(
            "expected-total-scenes must be greater than expected-locked-scenes"
        )
    pilot, reference, pilot_scenes = pilot_context(pilot_output, expected_pilot)
    gt_dir = resolve_protocol_path(reference.get("gt_dir"), "gt_dir")
    one_line_dir = resolve_protocol_path(reference.get("one_line_dir"), "one_line_dir")
    da3_relative_dir = resolve_protocol_path(
        reference.get("cached_da3_dir"), "cached_da3_dir"
    )
    all_scenes = sorted(path.stem for path in gt_dir.glob("*.mat"))
    if len(all_scenes) != expected_total:
        raise RuntimeError(
            f"Found {len(all_scenes)} iBims MAT scenes, expected {expected_total}: {gt_dir}"
        )
    missing_pilot = sorted(set(pilot_scenes) - set(all_scenes))
    if missing_pilot:
        raise RuntimeError(f"Pilot scenes are absent from iBims: {missing_pilot}")
    locked_scenes = [scene for scene in all_scenes if scene not in set(pilot_scenes)]
    if len(locked_scenes) != expected_locked:
        raise RuntimeError(
            f"Locked split contains {len(locked_scenes)} scenes, expected {expected_locked}"
        )
    return (
        pilot,
        reference,
        pilot_scenes,
        locked_scenes,
        gt_dir,
        one_line_dir,
        da3_relative_dir,
    )


def protocol_payload(
    pilot_output: Path,
    pilot: dict[str, Any],
    reference: dict[str, Any],
    pilot_scenes: list[str],
    locked_scenes: list[str],
    gt_dir: Path,
    one_line_dir: Path,
    da3_relative_dir: Path,
) -> dict[str, Any]:
    payload = {
        "benchmark": "iBims locked maximum-coverage four-line DA3 versus Any2Full",
        "version": VERSION,
        "status": "locked_before_any2full_evaluation",
        "pilot_output": str(pilot_output),
        "pilot_protocol_sha256": sha256_file(pilot_output / "protocol.json"),
        "pilot_scenes_excluded": pilot_scenes,
        "locked_scenes": locked_scenes,
        "gt_dir": str(gt_dir),
        "one_line_dir": str(one_line_dir),
        "cached_da3_relative_dir": str(da3_relative_dir),
        "row_fractions": list(ROW_FRACTIONS),
        "horizontal_sampling": (
            "unique x-columns present in each scene's established v2.1 one-line map"
        ),
        "one_pixel_per_return": True,
        "sensor_min_depth_m": float(reference.get("sensor_min_depth_m", 0.10)),
        "sensor_max_depth_m": float(reference.get("sensor_max_depth_m", 32.0)),
        "eval_max_depth_m": float(reference.get("eval_max_depth_m", 0.0)),
        "outside_margin_px": int(reference.get("outside_margin_px", 10)),
        "poisson_rtol": float(reference.get("poisson_rtol", 1e-6)),
        "poisson_maxiter": int(reference.get("poisson_maxiter", 5000)),
        "da3_pipeline": "cached DA3-SMALL relative + global median scale + validated Poisson",
        "any2full_pipeline": "Any2Full-vits native metric output; no external refinement",
        "primary_region": PRIMARY_REGION,
        "primary_metric": "rmse_m",
        "secondary_metrics": [
            "absrel_pct",
            "mae_m",
            "delta1_pct",
            "bad_050_pct",
            "bad_100_pct",
        ],
        "equal_scene_weight": True,
        "practical_rmse_threshold_pct": PRACTICAL_RMSE_THRESHOLD_PCT,
        "statistical_rule": (
            "paired scene bootstrap 95% CI excludes zero; Wilcoxon two-sided p<0.05; "
            "winner has >50% scene win rate"
        ),
        "common_mask_rule": "both methods are scored on identical pixels",
        "failure_rule": "no scene may be silently skipped or repaired",
        "prediction_gt_alignment": "none",
        "noise": "none",
        "dense_gt_use": "simulate only the frozen sparse rays and evaluate dense output",
        "source_reference_configuration_sha256": reference.get("configuration_sha256"),
        "source_pilot_configuration_sha256": pilot.get("configuration_sha256"),
    }
    payload["configuration_sha256"] = configuration_hash(payload)
    return payload


def validate_or_write_protocol(path: Path, payload: dict[str, Any], resume: bool) -> None:
    if path.is_file():
        old = read_json(path)
        if old.get("configuration_sha256") != payload.get("configuration_sha256"):
            raise RuntimeError(
                f"{path.parent} contains a different experiment. Use a new directory."
            )
        if not resume:
            raise FileExistsError(
                f"Locked preparation already exists at {path.parent}; pass --resume."
            )
        return
    atomic_json(path, payload)


def preparation_paths(root: Path, scene: str) -> dict[str, Path]:
    return {
        "rgb": root / "rgb" / f"{scene}.png",
        "sparse": root / "sparse_input_m" / f"{scene}.npy",
        "gt": root / "evaluation_only" / "gt_m" / f"{scene}.npy",
        "valid": root / "evaluation_only" / "valid_mask" / f"{scene}.npy",
        "one_mask": root / "evaluation_only" / "one_line_anchor_mask" / f"{scene}.npy",
        "four_mask": root / "evaluation_only" / "four_line_anchor_mask" / f"{scene}.npy",
        "da3": root / "da3_predictions_m" / f"{scene}.npy",
    }


def relative_paths(root: Path, paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(path.relative_to(root)) for key, path in paths.items()}


def completed_manifest_row(
    row: dict[str, str],
    root: Path,
) -> bool:
    required_keys = (
        "rgb",
        "sparse",
        "gt",
        "valid",
        "one_mask",
        "four_mask",
        "da3",
    )
    for key in required_keys:
        relative = row.get(key)
        if not relative or not (root / relative).is_file():
            return False
    sparse_path = root / row["sparse"]
    da3_path = root / row["da3"]
    return (
        sha256_file(sparse_path) == row.get("sparse_sha256")
        and sha256_file(da3_path) == row.get("da3_prediction_sha256")
    )


def assert_only_expected_inputs(root: Path, scenes: list[str]) -> None:
    expected_png = {f"{scene}.png" for scene in scenes}
    expected_npy = {f"{scene}.npy" for scene in scenes}
    for directory, expected, suffix in (
        (root / "rgb", expected_png, ".png"),
        (root / "sparse_input_m", expected_npy, ".npy"),
    ):
        present = {path.name for path in directory.glob(f"*{suffix}") if path.is_file()}
        unexpected = sorted(present - expected)
        if unexpected:
            raise RuntimeError(
                f"Unexpected files in locked Any2Full input directory {directory}: {unexpected}"
            )


def prepare(args: argparse.Namespace) -> None:
    da3_root = resolve_directory(args.da3_root, "DA3 root")
    pilot_output = resolve_directory(args.pilot_output, "pilot placement output")
    prepared_root = args.prepared_data_root.expanduser().resolve()
    (
        pilot,
        reference,
        pilot_scenes,
        locked_scenes,
        gt_dir,
        one_line_dir,
        da3_relative_dir,
    ) = locked_scene_context(
        pilot_output,
        args.expected_total_scenes,
        args.expected_locked_scenes,
    )
    protocol = protocol_payload(
        pilot_output,
        pilot,
        reference,
        pilot_scenes,
        locked_scenes,
        gt_dir,
        one_line_dir,
        da3_relative_dir,
    )
    prepared_root.mkdir(parents=True, exist_ok=True)
    validate_or_write_protocol(prepared_root / "protocol.json", protocol, args.resume)
    for scene in locked_scenes:
        for path in preparation_paths(prepared_root, scene).values():
            path.parent.mkdir(parents=True, exist_ok=True)
    assert_only_expected_inputs(prepared_root, locked_scenes)

    paired = import_paired_module(Path(__file__).resolve().parent)
    poisson = paired.load_poisson(da3_root)
    sensor_min = float(protocol["sensor_min_depth_m"])
    sensor_max = float(protocol["sensor_max_depth_m"])
    eval_max = float(protocol["eval_max_depth_m"])
    rtol = float(protocol["poisson_rtol"])
    maxiter = int(protocol["poisson_maxiter"])
    manifest_path = prepared_root / "manifest.csv"
    manifest_rows = read_csv_optional(manifest_path) if args.resume else []
    old_by_scene = {row.get("scene", ""): row for row in manifest_rows}

    for index, scene in enumerate(locked_scenes, start=1):
        old = old_by_scene.get(scene)
        if args.resume and old is not None and completed_manifest_row(old, prepared_root):
            print(f"[{index:3d}/{len(locked_scenes)}] {scene} resume-skip", flush=True)
            continue
        gt, valid, rgb = paired.load_ibims(gt_dir / f"{scene}.mat")
        if eval_max > 0:
            valid &= gt <= eval_max
        raw_one_path = paired.npy_path(one_line_dir, scene)
        relative_path = paired.npy_path(da3_relative_dir, scene, da3=True)
        raw_one = paired.load_npy(raw_one_path, gt.shape)
        relative = paired.load_npy(relative_path, gt.shape)
        one_sparse, one_anchors = paired.sanitize_one_line(
            raw_one, valid, sensor_min, sensor_max
        )
        simulated, _, rows, source_column_count = paired.simulate_four_lines(
            gt,
            valid,
            one_sparse,
            ROW_FRACTIONS,
            sensor_min,
            sensor_max,
        )
        paths = preparation_paths(prepared_root, scene)
        Image.fromarray(rgb, mode="RGB").save(paths["rgb"])
        np.save(paths["sparse"], simulated.astype(np.float32))

        # Reload the frozen file.  This exact array is used by DA3 here and by
        # Any2Full later, eliminating any ambiguity about input identity.
        frozen_sparse = paired.load_npy(paths["sparse"], gt.shape)
        four_anchors = frozen_sparse > 0
        if not np.array_equal(four_anchors, simulated > 0):
            raise RuntimeError(f"{scene}: frozen sparse support changed while saving")
        da3_median, median_scale = paired.median_align(
            relative, frozen_sparse, four_anchors
        )
        da3_prediction, poisson_diagnostics = paired.call_poisson(
            poisson,
            da3_median,
            frozen_sparse,
            four_anchors,
            rtol,
            maxiter,
        )
        np.save(paths["gt"], gt.astype(np.float32))
        np.save(paths["valid"], valid.astype(np.bool_))
        np.save(paths["one_mask"], one_anchors.astype(np.bool_))
        np.save(paths["four_mask"], four_anchors.astype(np.bool_))
        np.save(paths["da3"], da3_prediction.astype(np.float32))
        loaded_da3 = paired.load_npy(paths["da3"], gt.shape)
        if not np.all(np.isfinite(loaded_da3[valid]) & (loaded_da3[valid] > 0)):
            raise RuntimeError(f"{scene}: saved DA3 prediction is invalid")

        relative_map = relative_paths(prepared_root, paths)
        new_row: dict[str, Any] = {
            "scene": scene,
            **relative_map,
            "image_height": gt.shape[0],
            "image_width": gt.shape[1],
            "physical_anchor_count": int(np.count_nonzero(four_anchors)),
            "source_column_count": source_column_count,
            "line_rows": ";".join(str(value) for value in rows),
            "median_scale": median_scale,
            "poisson_repaired_pixels": int(
                poisson_diagnostics.get("invalid_pixels_repaired_from_median", 0)
            ),
            "source_one_line_path": str(raw_one_path),
            "source_da3_relative_path": str(relative_path),
            "source_one_line_sha256": sha256_file(raw_one_path),
            "source_da3_relative_sha256": sha256_file(relative_path),
            "rgb_sha256": sha256_file(paths["rgb"]),
            "sparse_sha256": sha256_file(paths["sparse"]),
            "da3_prediction_sha256": sha256_file(paths["da3"]),
        }
        manifest_rows = [row for row in manifest_rows if row.get("scene") != scene]
        manifest_rows.append(new_row)
        manifest_rows.sort(key=lambda row: str(row["scene"]))
        write_csv(manifest_path, manifest_rows)
        old_by_scene[scene] = {key: str(value) for key, value in new_row.items()}
        print(
            f"[{index:3d}/{len(locked_scenes)}] {scene}: "
            f"{int(np.count_nonzero(four_anchors))} anchors; DA3 saved",
            flush=True,
        )

    manifest_rows = read_csv_optional(manifest_path)
    completed = {
        row.get("scene", "")
        for row in manifest_rows
        if completed_manifest_row(row, prepared_root)
    }
    if completed != set(locked_scenes):
        missing = sorted(set(locked_scenes) - completed)
        raise RuntimeError(f"Locked preparation is incomplete: {missing}")
    assert_only_expected_inputs(prepared_root, locked_scenes)
    print("\n===== LOCKED 97-SCENE PREPARATION COMPLETE =====")
    print(f"Pilot scenes excluded: {', '.join(pilot_scenes)}")
    print(f"Locked scenes prepared: {len(locked_scenes)}")
    print(f"Any2Full RGB: {prepared_root / 'rgb'}")
    print(f"Any2Full sparse depth: {prepared_root / 'sparse_input_m'}")
    print(f"Frozen protocol: {prepared_root / 'protocol.json'}")


def load_npy_2d(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.squeeze(np.load(path))
    if value.ndim != 2:
        raise ValueError(f"Expected 2D NPY at {path}, got {value.shape}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{path}: {value.shape} != {shape}")
    return value


def load_prediction(path: Path, shape: tuple[int, int], valid: np.ndarray) -> np.ndarray:
    value = load_npy_2d(path, shape).astype(np.float32)
    invalid = valid & (~np.isfinite(value) | (value <= 0))
    if np.any(invalid):
        raise RuntimeError(
            f"{path}: {int(np.count_nonzero(invalid))} invalid evaluation pixels. "
            "Locked evaluation does not repair predictions."
        )
    return value


def common_masks(
    valid: np.ndarray,
    one_anchors: np.ndarray,
    four_anchors: np.ndarray,
    margin_px: int,
) -> dict[str, np.ndarray]:
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
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def wilcoxon_pvalue(improvement: np.ndarray) -> float:
    if np.allclose(improvement, 0):
        return 1.0
    try:
        return float(
            wilcoxon(
                improvement,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except ValueError:
        return 1.0


def paired_statistics(
    metric_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["scene"]), str(row["method"]), str(row["region"])): row
        for row in metric_rows
    }
    scenes = sorted({str(row["scene"]) for row in metric_rows})
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for region in REGIONS:
        for metric in METRICS:
            da3 = np.asarray(
                [float(lookup[(scene, "da3", region)][metric]) for scene in scenes],
                dtype=np.float64,
            )
            any2full = np.asarray(
                [float(lookup[(scene, "any2full", region)][metric]) for scene in scenes],
                dtype=np.float64,
            )
            if LOWER_IS_BETTER[metric]:
                improvement = any2full - da3
                da3_wins = da3 < any2full
            else:
                improvement = da3 - any2full
                da3_wins = da3 > any2full
            low, high = bootstrap_ci(improvement, bootstrap_samples, rng)
            wins = int(np.count_nonzero(da3_wins))
            binomial = binomtest(wins, len(scenes), 0.5)
            win_ci = binomial.proportion_ci(confidence_level=0.95, method="exact")
            denominator = abs(float(np.mean(any2full)))
            result.append(
                {
                    "region": region,
                    "region_label": REGIONS[region],
                    "metric": metric,
                    "scene_count": len(scenes),
                    "da3_mean": float(np.mean(da3)),
                    "any2full_mean": float(np.mean(any2full)),
                    "da3_improvement_mean": float(np.mean(improvement)),
                    "da3_relative_improvement_pct": (
                        100.0 * float(np.mean(improvement)) / denominator
                        if denominator > 0
                        else math.nan
                    ),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "wilcoxon_two_sided_p": wilcoxon_pvalue(improvement),
                    "da3_scene_wins": wins,
                    "ties": int(np.count_nonzero(np.isclose(da3, any2full))),
                    "any2full_scene_wins": int(np.count_nonzero(any2full < da3))
                    if LOWER_IS_BETTER[metric]
                    else int(np.count_nonzero(any2full > da3)),
                    "da3_win_rate_pct": 100.0 * wins / len(scenes),
                    "da3_win_rate_ci95_low_pct": 100.0 * float(win_ci.low),
                    "da3_win_rate_ci95_high_pct": 100.0 * float(win_ci.high),
                    "positive_improvement_means": "DA3 is better",
                }
            )
    return result


def stat_lookup(rows: list[dict[str, Any]], region: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if row["region"] == region and row["metric"] == metric:
            return row
    raise KeyError((region, metric))


def summary_lookup(rows: list[dict[str, Any]], method: str, region: str) -> dict[str, Any]:
    for row in rows:
        if row["method"] == method and row["region"] == region:
            return row
    raise KeyError((method, region))


def metric_lookup(
    rows: list[dict[str, Any]], scene: str, method: str, region: str
) -> dict[str, Any]:
    for row in rows:
        if row["scene"] == scene and row["method"] == method and row["region"] == region:
            return row
    raise KeyError((scene, method, region))


def final_decision(paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = stat_lookup(paired_rows, PRIMARY_REGION, "rmse_m")
    da3_mean = float(primary["da3_mean"])
    a2f_mean = float(primary["any2full_mean"])
    if a2f_mean > 0:
        da3_reduction_pct = 100.0 * (a2f_mean - da3_mean) / a2f_mean
    else:
        da3_reduction_pct = 0.0 if da3_mean == 0 else -math.inf
    if da3_mean > 0:
        a2f_reduction_pct = 100.0 * (da3_mean - a2f_mean) / da3_mean
    else:
        a2f_reduction_pct = 0.0 if a2f_mean == 0 else -math.inf
    ci_low = float(primary["bootstrap_ci95_low"])
    ci_high = float(primary["bootstrap_ci95_high"])
    pvalue = float(primary["wilcoxon_two_sided_p"])
    da3_win_rate = float(primary["da3_win_rate_pct"])
    a2f_win_rate = 100.0 * float(primary["any2full_scene_wins"]) / float(
        primary["scene_count"]
    )
    if (
        da3_reduction_pct >= PRACTICAL_RMSE_THRESHOLD_PCT
        and ci_low > 0
        and pvalue < 0.05
        and da3_win_rate > 50.0
    ):
        code = "DA3_LOWER_RMSE_CONFIRMED"
        headline = "DA3 + median + Poisson is the confirmed RMSE winner."
        winner = "DA3"
    elif (
        a2f_reduction_pct >= PRACTICAL_RMSE_THRESHOLD_PCT
        and ci_high < 0
        and pvalue < 0.05
        and a2f_win_rate > 50.0
    ):
        code = "ANY2FULL_LOWER_RMSE_CONFIRMED"
        headline = "Any2Full is the confirmed RMSE winner."
        winner = "Any2Full"
    else:
        code = "NO_CONCLUSIVE_RMSE_WINNER"
        headline = "The locked test does not establish a conclusive RMSE winner."
        winner = "inconclusive"

    absrel = stat_lookup(paired_rows, PRIMARY_REGION, "absrel_pct")
    if float(absrel["bootstrap_ci95_low"]) > 0 and float(
        absrel["wilcoxon_two_sided_p"]
    ) < 0.05:
        absrel_winner = "DA3"
    elif float(absrel["bootstrap_ci95_high"]) < 0 and float(
        absrel["wilcoxon_two_sided_p"]
    ) < 0.05:
        absrel_winner = "Any2Full"
    else:
        absrel_winner = "inconclusive"
    return {
        "decision_code": code,
        "headline": headline,
        "primary_winner": winner,
        "primary_region": PRIMARY_REGION,
        "primary_metric": "rmse_m",
        "da3_rmse_m": da3_mean,
        "any2full_rmse_m": a2f_mean,
        "da3_rmse_reduction_vs_any2full_pct": da3_reduction_pct,
        "any2full_rmse_reduction_vs_da3_pct": a2f_reduction_pct,
        "paired_improvement_ci95_low_m": ci_low,
        "paired_improvement_ci95_high_m": ci_high,
        "wilcoxon_two_sided_p": pvalue,
        "da3_win_rate_pct": da3_win_rate,
        "any2full_win_rate_pct": a2f_win_rate,
        "practical_threshold_pct": PRACTICAL_RMSE_THRESHOLD_PCT,
        "primary_absrel_winner": absrel_winner,
        "scope": "iBims simulated noiseless maximum-coverage four-line locked scenes",
    }


def depth_panel(
    role: str,
    scene: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    four_anchors: np.ndarray,
    primary_mask: np.ndarray,
    da3: np.ndarray,
    any2full: np.ndarray,
    da3_metric: dict[str, Any],
    a2f_metric: dict[str, Any],
    output: Path,
    depth_max_m: float,
    error_max_m: float,
) -> None:
    gt_show = np.where(valid, gt, np.nan)
    da3_show = np.where(valid, da3, np.nan)
    a2f_show = np.where(valid, any2full, np.nan)
    da3_error = np.where(valid, np.abs(da3 - gt), np.nan)
    a2f_error = np.where(valid, np.abs(any2full - gt), np.nan)
    gain = a2f_error - da3_error
    figure, axes = plt.subplots(2, 4, figsize=(22, 10.5), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    y, x = np.where(four_anchors)
    axes[0, 0].scatter(x, y, s=3, c="cyan", linewidths=0)
    axes[0, 0].set_title(f"Shared 4-line input ({len(x)} physical anchors)")
    depth_image = axes[0, 1].imshow(gt_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[0, 1].set_title("iBims metric GT")
    axes[0, 2].imshow(da3_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[0, 2].set_title(
        "DA3 + median + Poisson\n"
        f"RMSE {float(da3_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(da3_metric['absrel_pct']):.2f}%"
    )
    axes[0, 3].imshow(a2f_show, cmap="turbo", vmin=0, vmax=depth_max_m)
    axes[0, 3].set_title(
        "Any2Full-vits native\n"
        f"RMSE {float(a2f_metric['rmse_m']):.3f} m | "
        f"AbsRel {float(a2f_metric['absrel_pct']):.2f}%"
    )
    axes[1, 0].imshow(
        primary_mask.astype(np.uint8),
        cmap=ListedColormap(["black", "white"]),
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Primary common mask\noutside shared 4-line support")
    error_image = axes[1, 1].imshow(
        da3_error, cmap="magma", vmin=0, vmax=error_max_m
    )
    axes[1, 1].set_title("DA3 absolute error")
    axes[1, 2].imshow(a2f_error, cmap="magma", vmin=0, vmax=error_max_m)
    axes[1, 2].set_title("Any2Full absolute error")
    gain_image = axes[1, 3].imshow(
        gain, cmap="RdBu_r", vmin=-error_max_m, vmax=error_max_m
    )
    axes[1, 3].set_title("Absolute-error difference\nred = DA3 better; blue = Any2Full better")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.colorbar(
        depth_image,
        ax=axes[0, 1:].ravel().tolist(),
        shrink=0.72,
        label="Depth (m)",
    )
    figure.colorbar(
        error_image,
        ax=[axes[1, 1], axes[1, 2]],
        shrink=0.72,
        label="Absolute error (m)",
    )
    figure.colorbar(
        gain_image,
        ax=axes[1, 3],
        shrink=0.72,
        label="|Any2Full error| - |DA3 error| (m)",
    )
    difference = float(a2f_metric["rmse_m"]) - float(da3_metric["rmse_m"])
    figure.suptitle(
        f"{role.upper()} LOCKED-SCENE DIFFERENCE — {scene}\n"
        f"Positive paired RMSE improvement favors DA3: {difference:+.3f} m"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def summary_figure(
    metric_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    scenes = sorted({str(row["scene"]) for row in metric_rows})
    da3_rmse = np.asarray(
        [
            float(metric_lookup(metric_rows, scene, "da3", PRIMARY_REGION)["rmse_m"])
            for scene in scenes
        ]
    )
    a2f_rmse = np.asarray(
        [
            float(metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION)["rmse_m"])
            for scene in scenes
        ]
    )
    da3_absrel = np.asarray(
        [
            float(metric_lookup(metric_rows, scene, "da3", PRIMARY_REGION)["absrel_pct"])
            for scene in scenes
        ]
    )
    a2f_absrel = np.asarray(
        [
            float(metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION)["absrel_pct"])
            for scene in scenes
        ]
    )
    differences = a2f_rmse - da3_rmse
    order = np.argsort(differences)
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    rmse_limit = 1.05 * max(float(np.max(da3_rmse)), float(np.max(a2f_rmse)))
    axes[0, 0].scatter(a2f_rmse, da3_rmse, alpha=0.75, s=28)
    axes[0, 0].plot([0, rmse_limit], [0, rmse_limit], "k--", linewidth=1)
    axes[0, 0].set_xlim(0, rmse_limit)
    axes[0, 0].set_ylim(0, rmse_limit)
    axes[0, 0].set_xlabel("Any2Full RMSE (m)")
    axes[0, 0].set_ylabel("DA3 RMSE (m)")
    axes[0, 0].set_title("Primary RMSE per scene\nbelow diagonal = DA3 better")
    colors = np.where(differences[order] >= 0, "tab:red", "tab:blue")
    axes[0, 1].bar(np.arange(len(scenes)), differences[order], color=colors, width=0.9)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_xlabel("Locked scenes sorted by paired difference")
    axes[0, 1].set_ylabel("Any2Full RMSE - DA3 RMSE (m)")
    axes[0, 1].set_title("Positive bars favor DA3")
    absrel_limit = 1.05 * max(float(np.max(da3_absrel)), float(np.max(a2f_absrel)))
    axes[1, 0].scatter(a2f_absrel, da3_absrel, alpha=0.75, s=28)
    axes[1, 0].plot([0, absrel_limit], [0, absrel_limit], "k--", linewidth=1)
    axes[1, 0].set_xlim(0, absrel_limit)
    axes[1, 0].set_ylim(0, absrel_limit)
    axes[1, 0].set_xlabel("Any2Full AbsRel (%)")
    axes[1, 0].set_ylabel("DA3 AbsRel (%)")
    axes[1, 0].set_title("Primary AbsRel per scene\nbelow diagonal = DA3 better")
    axes[1, 1].hist(differences, bins=min(25, max(8, len(scenes) // 4)), color="0.35")
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].axvline(float(np.mean(differences)), color="tab:red", linewidth=2)
    axes[1, 1].set_xlabel("Any2Full RMSE - DA3 RMSE (m)")
    axes[1, 1].set_ylabel("Scene count")
    axes[1, 1].set_title(f"Paired RMSE distribution\nmean {np.mean(differences):+.3f} m")
    figure.suptitle(
        "Locked iBims maximum-coverage four-line comparison\n"
        "All plots use the outside-shared-four-line primary mask"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    protocol: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    manifest_rows: list[dict[str, str]],
) -> None:
    scenes = sorted({str(row["scene"]) for row in metric_rows})
    lines = [
        "# Locked iBims four-line DA3 versus Any2Full result",
        "",
        f"## FINAL DECISION: {decision['decision_code']}",
        "",
        f"**{decision['headline']}**",
        "",
        f"Scope: {decision['scope']}.",
        "",
        "## Locked protocol",
        "",
        f"- Locked test scenes: {len(scenes)}; three pilot scenes excluded before evaluation.",
        "- Frozen four-line rows: 12.5%, 37.5%, 62.5%, and 87.5% of image height.",
        "- Both methods receive the same RGB and byte-identical float32 sparse-depth NPY for every scene.",
        "- Primary region: outside the shared four-line support.",
        "- Primary metric: equal-scene-weight RMSE.",
        "- Dense GT is used only to simulate the fixed sparse rays and evaluate; it is never used for post-inference fitting.",
        "- No scene skipping, prediction repair, GT alignment, or tuning after pilot selection is permitted.",
        "",
        "## Aggregate results",
        "",
        "A positive improvement means DA3 reduced error relative to Any2Full.",
        "",
        "| Common region | DA3 RMSE | Any2Full RMSE | DA3 RMSE improvement | DA3 AbsRel | Any2Full AbsRel | DA3 AbsRel improvement | DA3 RMSE win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        da3 = summary_lookup(summary_rows, "da3", region)
        a2f = summary_lookup(summary_rows, "any2full", region)
        rmse = stat_lookup(paired_rows, region, "rmse_m")
        absrel = stat_lookup(paired_rows, region, "absrel_pct")
        lines.append(
            f"| {REGIONS[region]} | {float(da3['mean_rmse_m']):.4f} m | "
            f"{float(a2f['mean_rmse_m']):.4f} m | "
            f"{float(rmse['da3_relative_improvement_pct']):+.2f}% | "
            f"{float(da3['mean_absrel_pct']):.3f}% | "
            f"{float(a2f['mean_absrel_pct']):.3f}% | "
            f"{float(absrel['da3_relative_improvement_pct']):+.2f}% | "
            f"{float(rmse['da3_win_rate_pct']):.1f}% |"
        )
    primary = stat_lookup(paired_rows, PRIMARY_REGION, "rmse_m")
    primary_absrel = stat_lookup(paired_rows, PRIMARY_REGION, "absrel_pct")
    lines.extend(
        [
            "",
            "## Primary statistical decision",
            "",
            "| Requirement | Predeclared rule | Result | Pass? |",
            "|---|---:|---:|---|",
            f"| Practical RMSE improvement | at least {PRACTICAL_RMSE_THRESHOLD_PCT:.1f}% | {float(decision['da3_rmse_reduction_vs_any2full_pct']):+.2f}% for DA3 | {'Yes' if float(decision['da3_rmse_reduction_vs_any2full_pct']) >= PRACTICAL_RMSE_THRESHOLD_PCT else 'No'} |",
            f"| Paired 95% CI | entirely above 0 for DA3 | [{float(primary['bootstrap_ci95_low']):+.4f}, {float(primary['bootstrap_ci95_high']):+.4f}] m | {'Yes' if float(primary['bootstrap_ci95_low']) > 0 else 'No'} |",
            f"| Wilcoxon test | p < 0.05 | p={float(primary['wilcoxon_two_sided_p']):.6g} | {'Yes' if float(primary['wilcoxon_two_sided_p']) < 0.05 else 'No'} |",
            f"| Scene win rate | above 50% | {float(primary['da3_win_rate_pct']):.1f}% ({int(primary['da3_scene_wins'])}/{int(primary['scene_count'])}) | {'Yes' if float(primary['da3_win_rate_pct']) > 50 else 'No'} |",
            "",
            f"Primary paired RMSE difference (Any2Full - DA3): **{float(primary['da3_improvement_mean']):+.4f} m**. Positive favors DA3.",
            "",
            f"Primary AbsRel: DA3 {float(primary_absrel['da3_mean']):.3f}% versus Any2Full {float(primary_absrel['any2full_mean']):.3f}%; statistical winner: **{decision['primary_absrel_winner']}**.",
            "",
            "## Distribution and failure diagnostics",
            "",
            "| Method | Primary RMSE median | RMSE p90 | Worst RMSE | AbsRel median | AbsRel p90 | Worst AbsRel |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("da3", "any2full"):
        row = summary_lookup(summary_rows, method, PRIMARY_REGION)
        lines.append(
            f"| {METHODS[method]} | {float(row['median_rmse_m']):.4f} m | "
            f"{float(row['p90_rmse_m']):.4f} m | {float(row['max_rmse_m']):.4f} m | "
            f"{float(row['median_absrel_pct']):.3f}% | "
            f"{float(row['p90_absrel_pct']):.3f}% | "
            f"{float(row['max_absrel_pct']):.3f}% |"
        )
    differences: list[tuple[float, str, float, float]] = []
    for scene in scenes:
        da3 = float(metric_lookup(metric_rows, scene, "da3", PRIMARY_REGION)["rmse_m"])
        a2f = float(
            metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION)["rmse_m"]
        )
        differences.append((a2f - da3, scene, da3, a2f))
    differences.sort()
    lines.extend(
        [
            "",
            "## Largest disagreements (automatic, not hand-selected)",
            "",
            "| Scene | DA3 RMSE | Any2Full RMSE | Any2Full - DA3 | Winner |",
            "|---|---:|---:|---:|---|",
        ]
    )
    selected_disagreements = differences[:5] + differences[-5:]
    seen_scenes: set[str] = set()
    for difference, scene, da3, a2f in selected_disagreements:
        if scene in seen_scenes:
            continue
        seen_scenes.add(scene)
        lines.append(
            f"| {scene} | {da3:.4f} m | {a2f:.4f} m | {difference:+.4f} m | "
            f"{'DA3' if da3 < a2f else 'Any2Full'} |"
        )
    anchor_counts = np.asarray(
        [int(row["physical_anchor_count"]) for row in manifest_rows], dtype=np.float64
    )
    repairs = sum(int(row.get("poisson_repaired_pixels", 0)) for row in manifest_rows)
    lines.extend(
        [
            "",
            "## Input and numerical audit",
            "",
            f"- Physical anchors per scene: mean {np.mean(anchor_counts):.1f}, median {np.median(anchor_counts):.0f}, range {int(np.min(anchor_counts))}-{int(np.max(anchor_counts))}.",
            "- Every prepared sparse NPY is the exact reloaded input used by DA3 and later supplied unchanged to Any2Full.",
            f"- DA3 Poisson invalid pixels repaired from its median prior: {repairs} across all locked scenes.",
            "- Any2Full `_rel.npy` sidecars are excluded; only native metric `<scene>.npy` predictions are scored.",
            "",
            "## Claim boundary",
            "",
            "This locked result supports a claim only for simulated, noiseless, maximum-coverage four-line iBims input. Real-sensor superiority requires a separate physical-LiDAR experiment with independent ground truth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    pilot_output = resolve_directory(args.pilot_output, "pilot placement output")
    prepared_root = resolve_directory(args.prepared_data_root, "prepared locked data")
    any2full_dir = resolve_directory(args.any2full_dir, "Any2Full prediction directory")
    output_dir = args.output_dir.expanduser().resolve()
    if args.plot_max_depth_m <= 0 or args.plot_error_max_m <= 0:
        raise ValueError("plot limits must be positive")
    protocol = read_json(prepared_root / "protocol.json")
    if protocol.get("configuration_sha256") != configuration_hash(protocol):
        raise RuntimeError("Prepared protocol hash is invalid or was edited")
    if protocol.get("pilot_protocol_sha256") != sha256_file(pilot_output / "protocol.json"):
        raise RuntimeError("Prepared data does not belong to this pilot output")
    scenes = [str(value) for value in protocol.get("locked_scenes", [])]
    if len(scenes) != args.expected_locked_scenes or len(set(scenes)) != len(scenes):
        raise RuntimeError(
            f"Prepared protocol has {len(scenes)} locked scenes; "
            f"expected {args.expected_locked_scenes}"
        )
    if protocol.get("primary_region") != PRIMARY_REGION:
        raise RuntimeError("Prepared protocol has a different primary region")
    manifest_rows = read_csv_optional(prepared_root / "manifest.csv")
    manifest_by_scene = {row.get("scene", ""): row for row in manifest_rows}
    if set(manifest_by_scene) != set(scenes):
        raise RuntimeError("Prepared manifest is incomplete or contains extra scenes")
    for scene in scenes:
        if not completed_manifest_row(manifest_by_scene[scene], prepared_root):
            raise RuntimeError(f"Prepared scene failed hash/completeness audit: {scene}")

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
            "Any2Full output must contain exactly all locked metric predictions. "
            f"missing={missing}, extras={extras}"
        )
    failed_path = any2full_dir / "failed_pairs.txt"
    if failed_path.is_file() and failed_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Any2Full reported failed scenes: {failed_path}")

    paired = import_paired_module(Path(__file__).resolve().parent)
    margin = int(protocol["outside_margin_px"])
    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, start=1):
        manifest = manifest_by_scene[scene]
        paths = {key: prepared_root / manifest[key] for key in (
            "rgb", "sparse", "gt", "valid", "one_mask", "four_mask", "da3"
        )}
        if sha256_file(paths["sparse"]) != manifest["sparse_sha256"]:
            raise RuntimeError(f"{scene}: shared sparse input hash changed")
        gt = load_npy_2d(paths["gt"]).astype(np.float32)
        valid = load_npy_2d(paths["valid"], gt.shape).astype(bool)
        one_anchors = load_npy_2d(paths["one_mask"], gt.shape).astype(bool)
        four_anchors = load_npy_2d(paths["four_mask"], gt.shape).astype(bool)
        sparse = load_npy_2d(paths["sparse"], gt.shape).astype(np.float32)
        if not np.array_equal(sparse > 0, four_anchors):
            raise RuntimeError(f"{scene}: shared sparse support differs from frozen mask")
        rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != gt.shape:
            raise ValueError(f"{scene}: RGB/GT shape mismatch")
        da3 = load_prediction(paths["da3"], gt.shape, valid)
        a2f_path = any2full_dir / f"{scene}.npy"
        any2full = load_prediction(a2f_path, gt.shape, valid)
        masks = common_masks(valid, one_anchors, four_anchors, margin)
        for method, prediction in (("da3", da3), ("any2full", any2full)):
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
        audit_rows.append(
            {
                "scene": scene,
                "physical_anchor_count": int(np.count_nonzero(four_anchors)),
                "sparse_sha256": sha256_file(paths["sparse"]),
                "sparse_hash_verified": True,
                "da3_prediction_sha256": sha256_file(paths["da3"]),
                "any2full_prediction_sha256": sha256_file(a2f_path),
                "shape": f"{gt.shape[0]}x{gt.shape[1]}",
            }
        )
        if index == 1 or index % 10 == 0 or index == len(scenes):
            print(f"[{index:3d}/{len(scenes)}] locked scenes evaluated", flush=True)

    summary_rows = aggregate_metrics(metric_rows)
    paired_rows = paired_statistics(metric_rows, args.bootstrap_samples, args.seed)
    decision = final_decision(paired_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_scene_metrics.csv", metric_rows)
    write_csv(output_dir / "summary_equal_scene_weight.csv", summary_rows)
    write_csv(output_dir / "paired_statistics.csv", paired_rows)
    write_csv(output_dir / "input_audit.csv", audit_rows)
    atomic_json(output_dir / "final_decision.json", decision)
    atomic_json(
        output_dir / "evaluation_protocol.json",
        {
            "prepared_protocol_sha256": sha256_file(prepared_root / "protocol.json"),
            "any2full_prediction_directory": str(any2full_dir),
            "scene_count": len(scenes),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "primary_region": PRIMARY_REGION,
            "primary_metric": "rmse_m",
            "decision_rule": protocol["statistical_rule"],
            "practical_rmse_threshold_pct": PRACTICAL_RMSE_THRESHOLD_PCT,
        },
    )
    summary_figure(metric_rows, output_dir / "locked_summary.png")

    if not args.skip_panels:
        differences: list[tuple[float, str]] = []
        for scene in scenes:
            da3_metric = metric_lookup(metric_rows, scene, "da3", PRIMARY_REGION)
            a2f_metric = metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION)
            differences.append(
                (float(a2f_metric["rmse_m"]) - float(da3_metric["rmse_m"]), scene)
            )
        differences.sort()
        choices = (
            ("any2full_best", differences[0][1]),
            ("typical", differences[len(differences) // 2][1]),
            ("da3_best", differences[-1][1]),
        )
        visual_root = output_dir / "automatic_examples"
        for role, scene in choices:
            manifest = manifest_by_scene[scene]
            scene_paths = {
                key: prepared_root / manifest[key]
                for key in ("rgb", "gt", "valid", "one_mask", "four_mask", "da3")
            }
            gt = load_npy_2d(scene_paths["gt"]).astype(np.float32)
            valid = load_npy_2d(scene_paths["valid"], gt.shape).astype(bool)
            one_anchors = load_npy_2d(
                scene_paths["one_mask"], gt.shape
            ).astype(bool)
            four_anchors = load_npy_2d(
                scene_paths["four_mask"], gt.shape
            ).astype(bool)
            rgb = np.asarray(
                Image.open(scene_paths["rgb"]).convert("RGB"), dtype=np.uint8
            )
            da3 = load_prediction(scene_paths["da3"], gt.shape, valid)
            any2full = load_prediction(
                any2full_dir / f"{scene}.npy", gt.shape, valid
            )
            masks = common_masks(valid, one_anchors, four_anchors, margin)
            depth_panel(
                role,
                scene,
                rgb,
                gt,
                valid,
                four_anchors,
                masks[PRIMARY_REGION],
                da3,
                any2full,
                metric_lookup(metric_rows, scene, "da3", PRIMARY_REGION),
                metric_lookup(metric_rows, scene, "any2full", PRIMARY_REGION),
                visual_root / f"{role}__{scene}.png",
                args.plot_max_depth_m,
                args.plot_error_max_m,
            )

    report_path = output_dir / "locked_comparison_report.md"
    write_report(
        report_path,
        protocol,
        summary_rows,
        paired_rows,
        metric_rows,
        decision,
        manifest_rows,
    )
    print("\n===== LOCKED 97-SCENE FINAL RESULT =====\n")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Final decision: {output_dir / 'final_decision.json'}")
    print(f"Summary chart: {output_dir / 'locked_summary.png'}")
    print(f"Full report: {report_path}")


def self_test() -> None:
    fake_rows: list[dict[str, Any]] = []
    for scene_index in range(20):
        scene = f"scene_{scene_index:02d}"
        for region in REGIONS:
            for method, rmse, absrel in (
                ("da3", 0.80 + 0.002 * scene_index, 2.00),
                ("any2full", 1.00 + 0.002 * scene_index, 2.20),
            ):
                fake_rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "region": region,
                        "rmse_m": rmse,
                        "absrel_pct": absrel,
                        "mae_m": rmse * 0.8,
                        "delta1_pct": 99.0 if method == "da3" else 98.0,
                        "bad_050_pct": 2.0 if method == "da3" else 3.0,
                        "bad_100_pct": 0.2 if method == "da3" else 0.5,
                    }
                )
    stats = paired_statistics(fake_rows, 2000, 11)
    decision = final_decision(stats)
    if decision["decision_code"] != "DA3_LOWER_RMSE_CONFIRMED":
        raise AssertionError(decision)
    primary = stat_lookup(stats, PRIMARY_REGION, "rmse_m")
    if float(primary["bootstrap_ci95_low"]) <= 0:
        raise AssertionError(primary)
    print("SELF-TEST PASSED")


def main() -> None:
    args = arguments()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "self-test":
        self_test()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
