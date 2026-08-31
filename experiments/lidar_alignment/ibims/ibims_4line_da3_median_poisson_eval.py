#!/usr/bin/env python3
"""iBims: simulated 4-line LiDAR + cached DA3 + median + existing Poisson.

The script reuses the project's validated ``existing_poisson`` function from
``compare_median_poisson_oasis_100.py``. Four lines are sampled from dense GT
at fixed normalized rows. Horizontal sample columns come from each scene's
existing one-line sparse map. No artificial noise is added.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt

METHOD = "da3_median_poisson_4line"


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--da3-root", type=Path, required=True)
    p.add_argument("--a2f-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--gt-dir", type=Path)
    p.add_argument("--one-line-dir", type=Path)
    p.add_argument("--da3-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--row-fracs", type=float, nargs=4,
                   default=(0.20, 0.40, 0.60, 0.80))
    p.add_argument("--sensor-min-depth-m", type=float, default=0.10)
    p.add_argument("--sensor-max-depth-m", type=float, default=32.0)
    p.add_argument("--eval-max-depth-m", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--limit", type=int)
    p.add_argument("--scene")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plot-max-depth-m", type=float, default=10.0)
    p.add_argument("--plot-error-max-m", type=float, default=1.0)
    return p.parse_args()


def roots(args: argparse.Namespace) -> list[Path]:
    raw = [args.a2f_root, args.data_root]
    if os.environ.get("DA3_LIDAR_DATA_ROOT"):
        raw.append(Path(os.environ["DA3_LIDAR_DATA_ROOT"]))
    result, seen = [], set()
    for item in raw:
        if item is None:
            continue
        item = item.expanduser().resolve()
        if item.exists() and str(item) not in seen:
            result.append(item); seen.add(str(item))
    return result


def resolve_gt(explicit: Path | None, search_roots: list[Path]) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    rels = [
        "datasets/ibims1/ibims1_core_mat", "datasets/ibims1_core_mat",
        "data/ibims1/ibims1_core_mat", "data/ibims1_core_mat",
        "ibims1_core_mat",
    ]
    for root in search_roots:
        for rel in rels:
            d = root / rel
            if d.is_dir() and next(d.glob("*.mat"), None):
                return d.resolve()
    counts: Counter[Path] = Counter()
    for root in search_roots:
        for f in root.rglob("*.mat"):
            if "ibims" in str(f).lower():
                counts[f.parent.resolve()] += 1
    if not counts:
        raise FileNotFoundError("Could not locate iBims MAT files")
    return max(counts, key=counts.get)


def resolve_npy_dir(explicit: Path | None, search_roots: list[Path],
                    names: list[str], label: str) -> Path:
    if explicit:
        d = explicit.expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"{label}: {d}")
        return d
    for root in search_roots:
        for name in names:
            for rel in (f"experiments/ibims_replication/{name}",
                        f"ibims_replication/{name}", name):
                d = root / rel
                if d.is_dir() and next(d.glob("*.npy"), None):
                    return d.resolve()
    found = []
    for root in search_roots:
        for name in names:
            for d in root.rglob(name):
                if d.is_dir():
                    n = sum(1 for _ in d.glob("*.npy"))
                    if n:
                        found.append((names.index(name), n, d.resolve()))
    if not found:
        raise FileNotFoundError(f"Could not locate {label}")
    return min(found, key=lambda x: (x[0], -x[1]))[2]


def field(record: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(record[name])
    except Exception as exc:
        raise KeyError(f"Missing iBims MAT field {name!r}") from exc


def load_ibims(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = loadmat(path)["data"][0, 0]
    gt = np.squeeze(field(record, "depth")).astype(np.float32)
    valid = ((np.squeeze(field(record, "mask_invalid")) == 1) &
             (np.squeeze(field(record, "mask_transp")) == 1) &
             np.isfinite(gt) & (gt > 0))
    rgb = np.squeeze(field(record, "rgb"))
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if np.nanmax(rgb) <= 1:
            rgb *= 255
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return gt, valid, rgb


def npy_path(directory: Path, scene: str, da3: bool = False) -> Path:
    names = ([f"{scene}_da3small.npy", f"{scene}_da3.npy", f"{scene}.npy"]
             if da3 else
             [f"{scene}.npy", f"{scene}_sensor.npy", f"{scene}_sparse.npy"])
    for name in names:
        p = directory / name
        if p.is_file():
            return p
    matches = sorted(directory.glob(f"{scene}*.npy"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve {scene} in {directory}: {matches}")


def load_npy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    a = np.squeeze(np.load(path)).astype(np.float32)
    if a.shape != shape:
        raise ValueError(f"{path}: {a.shape} != {shape}")
    return a


def load_poisson(da3_root: Path) -> Callable[..., Any]:
    path = (da3_root / "experiments/lidar_alignment/ibims/"
            "compare_median_poisson_oasis_100.py")
    if not path.is_file():
        raise FileNotFoundError(f"Validated Poisson source missing: {path}")
    spec = importlib.util.spec_from_file_location("validated_ibims_poisson", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "existing_poisson", None)
    if not callable(fn):
        raise AttributeError(f"{path} has no existing_poisson")
    print(f"Reusing existing_poisson{inspect.signature(fn)} from {path}")
    return fn


def call_poisson(fn: Callable[..., Any], base: np.ndarray, sparse: np.ndarray,
                 anchors: np.ndarray, rtol: float, maxiter: int) -> tuple[np.ndarray, dict]:
    aliases = {
        "base": base, "base_depth": base, "depth": base, "initial": base,
        "initial_depth": base, "aligned": base, "aligned_depth": base,
        "prediction": base, "pred": base, "sparse": sparse,
        "sparse_depth": sparse, "lidar": sparse, "lidar_depth": sparse,
        "metric_depth": sparse, "anchors": anchors, "anchor_mask": anchors,
        "sparse_mask": anchors, "valid_mask": anchors, "rtol": rtol,
        "tol": rtol, "maxiter": maxiter, "max_iter": maxiter,
    }
    sig = inspect.signature(fn)
    kwargs, unknown = {}, []
    for name, par in sig.parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif par.default is inspect.Parameter.empty and par.kind not in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            unknown.append(name)
    if unknown:
        result = fn(base, sparse, anchors, rtol, maxiter)
    else:
        result = fn(**kwargs)
    pred, diag = (result[0], result[1]) if isinstance(result, tuple) else (result, {})
    pred = np.squeeze(np.asarray(pred, np.float32))
    if pred.shape != base.shape:
        raise ValueError(f"Poisson returned {pred.shape}; expected {base.shape}")
    return pred, diag if isinstance(diag, dict) else {"value": diag}


def four_lines(gt: np.ndarray, valid: np.ndarray, old_sparse: np.ndarray,
               fracs: list[float], min_m: float, max_m: float):
    h, _ = gt.shape
    columns = np.unique(np.where(np.isfinite(old_sparse) & (old_sparse > 0))[1])
    if len(columns) < 5:
        raise RuntimeError(f"Only {len(columns)} source x-columns")
    rows = [int(round(f * (h - 1))) for f in fracs]
    if len(set(rows)) != 4:
        raise ValueError(f"Duplicate 4-line rows: {rows}")
    sparse = np.zeros_like(gt, np.float32)
    for row in rows:
        ok = valid[row, columns] & (gt[row, columns] >= min_m) & (gt[row, columns] <= max_m)
        sparse[row, columns[ok]] = gt[row, columns[ok]]
    anchors = sparse > 0
    if anchors.sum() < 12:
        raise RuntimeError(f"Only {anchors.sum()} simulated anchors")
    return sparse, anchors, rows, len(columns)


def median_align(relative: np.ndarray, sparse: np.ndarray, anchors: np.ndarray):
    ratios = sparse[anchors] / np.maximum(relative[anchors], 1e-8)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 3:
        raise RuntimeError("Too few scale ratios")
    scale = float(np.median(ratios))
    return (relative * scale).astype(np.float32), scale


def metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    m = mask & np.isfinite(pred) & (pred > 0)
    p, t = pred[m].astype(float), gt[m].astype(float)
    e, ae = p - t, np.abs(p - t)
    ratio = np.maximum(p / t, t / p)
    return {
        "n": int(m.sum()), "absrel_pct": float(100 * np.mean(ae / t)),
        "rmse_m": float(np.sqrt(np.mean(e * e))), "mae_m": float(np.mean(ae)),
        "bias_m": float(np.mean(e)), "delta1_pct": float(100 * np.mean(ratio < 1.25)),
        "bad_050_pct": float(100 * np.mean(ae > .5)),
        "bad_100_pct": float(100 * np.mean(ae > 1.0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    names, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(key); seen.add(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def patch_probe(y: int, x: int, gt: np.ndarray, pred: np.ndarray,
                valid: np.ndarray, r: int = 3):
    m = valid[y-r:y+r+1, x-r:x+r+1]
    pp, tt = pred[y-r:y+r+1, x-r:x+r+1], gt[y-r:y+r+1, x-r:x+r+1]
    use = m & np.isfinite(pp) & (pp > 0)
    if use.sum() < 36:
        return None
    g, p = float(np.median(tt[use])), float(np.median(pp[use]))
    std = float(np.std(tt[use]))
    if std > max(.08, .08 * g):
        return None
    return {"gt_m": g, "pred_m": p, "abs_error_m": abs(p-g),
            "absrel_pct": 100 * abs(p-g) / g, "gt_patch_std_m": std}


def probes(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, anchors: np.ndarray):
    dist = distance_transform_edt(~anchors)
    h, w = gt.shape
    candidate = valid & ~anchors & (dist >= max(4, np.percentile(dist[valid & ~anchors], 70)))
    candidate[:24] = False; candidate[-24:] = False
    candidate[:, :24] = False; candidate[:, -24:] = False
    yy, xx = np.where(candidate)
    step = max(1, len(yy) // 2500)
    items = []
    for y, x in zip(yy[::step], xx[::step]):
        q = patch_probe(int(y), int(x), gt, pred, valid)
        if q:
            items.append({"y": int(y), "x": int(x),
                          "distance_to_anchor_px": float(dist[y, x]), **q})
    if not items:
        return []
    depths = np.array([q["gt_m"] for q in items])
    selected = []
    for code, label, percentile in (("N", "near non-anchor surface", 25),
                                    ("F", "far non-anchor surface", 75)):
        target = float(np.percentile(depths, percentile))
        for q in sorted(items, key=lambda z: (abs(z["gt_m"]-target), z["gt_patch_std_m"])):
            if all((q["x"]-s["x"])**2+(q["y"]-s["y"])**2 >= 24**2 for s in selected):
                selected.append({"code": code, "label": label, **q}); break
    for q in sorted(items, key=lambda z: z["absrel_pct"], reverse=True):
        if all((q["x"]-s["x"])**2+(q["y"]-s["y"])**2 >= 24**2 for s in selected):
            selected.append({"code": "W", "label": "worst stable non-anchor surface", **q}); break
    return selected


def panel(role: str, scene: str, rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray,
          valid: np.ndarray, sparse: np.ndarray, rows: list[int], ps: list[dict],
          met: dict[str, float], output: Path, vmax: float, emax: float):
    g = np.where(valid, gt, np.nan)
    p = np.where(valid & np.isfinite(pred) & (pred > 0), pred, np.nan)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax[0,0].imshow(rgb)
    ay, a_x = np.where(sparse > 0)
    ax[0,0].scatter(a_x, ay, s=3, c="cyan")
    for row in rows: ax[0,0].axhline(row, c="cyan", lw=.5, alpha=.5)
    ax[0,0].set_title("RGB + simulated 4-line LiDAR")
    im = ax[0,1].imshow(g, cmap="turbo", vmin=0, vmax=vmax); ax[0,1].set_title("GT metric depth")
    ax[1,0].imshow(p, cmap="turbo", vmin=0, vmax=vmax); ax[1,0].set_title("DA3 + median + existing Poisson")
    ei = ax[1,1].imshow(np.abs(p-g), cmap="inferno", vmin=0, vmax=emax); ax[1,1].set_title("Absolute depth error")
    colors = {"N":"lime", "F":"dodgerblue", "W":"red"}
    for q in ps:
        x, y, c, code = q["x"], q["y"], colors[q["code"]], q["code"]
        for a in ax.flat:
            a.scatter([x],[y],s=80,facecolors="none",edgecolors=c,lw=2)
            a.text(x,y,code,c=c,weight="bold",ha="center",va="center",fontsize=8)
        box = dict(facecolor="black", alpha=.72, edgecolor="none", pad=2)
        ax[0,1].text(x+7,y-7,f"{code} GT {q['gt_m']:.2f} m",c=c,fontsize=8,weight="bold",bbox=box)
        ax[1,0].text(x+7,y-7,f"{code} Pred {q['pred_m']:.2f} m\n|e| {q['abs_error_m']:.2f} m ({q['absrel_pct']:.1f}%)",c=c,fontsize=8,weight="bold",bbox=box)
        ax[1,1].text(x+7,y-7,f"{code} d(anchor) {q['distance_to_anchor_px']:.1f}px",c=c,fontsize=8,weight="bold",bbox=box)
    for a in ax.flat: a.set_axis_off()
    fig.colorbar(im, ax=[ax[0,1], ax[1,0]], shrink=.8, label="Depth (m)")
    fig.colorbar(ei, ax=ax[1,1], shrink=.8, label="Absolute error (m)")
    fig.suptitle(f"{role.upper()} — {scene}\nAbsRel {met['absrel_pct']:.3f}% | RMSE {met['rmse_m']:.3f} m | MAE {met['mae_m']:.3f} m | anchors {(sparse>0).sum()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)


def summary(rows: list[dict[str, Any]], args: argparse.Namespace):
    keys = ["absrel_pct","rmse_m","mae_m","bias_m","delta1_pct","bad_050_pct","bad_100_pct"]
    s: dict[str, Any] = {"scene_count": len(rows)}
    for key in keys:
        a = np.array([float(r[key]) for r in rows])
        s[f"mean_{key}"] = float(np.mean(a)); s[f"median_{key}"] = float(np.median(a))
    lines = [
        "iBims-1 simulated four-line evaluation",
        "Method: DA3 + median + existing Poisson",
        "Primary mask: all valid, non-transparent iBims GT pixels",
        "Aggregation: per-scene metrics followed by macro mean/median",
        f"Rows: {', '.join(f'{x:.3f}' for x in args.row_fracs)}",
        "Horizontal sampling: existing one-line sensor x-columns",
        f"Sensor range: {args.sensor_min_depth_m:g}-{args.sensor_max_depth_m:g} m; noise: none",
        f"Scenes: {len(rows)}", "",
        f"AbsRel mean/median: {s['mean_absrel_pct']:.3f}% / {s['median_absrel_pct']:.3f}%",
        f"RMSE mean/median: {s['mean_rmse_m']:.3f} / {s['median_rmse_m']:.3f} m",
        f"MAE mean/median: {s['mean_mae_m']:.3f} / {s['median_mae_m']:.3f} m",
        f"Bias mean/median: {s['mean_bias_m']:+.3f} / {s['median_bias_m']:+.3f} m",
        f"delta1 mean: {s['mean_delta1_pct']:.3f}%",
        f">0.50 m mean: {s['mean_bad_050_pct']:.3f}%",
        f">1.00 m mean: {s['mean_bad_100_pct']:.3f}%", "",
        "N/F/W values on panels are 7x7 patch medians, not single pixels.",
    ]
    (args.output_dir/"summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_csv(args.output_dir/"summary.csv", [s])


def main():
    args = arguments()
    args.da3_root = args.da3_root.expanduser().resolve()
    args.a2f_root = args.a2f_root.expanduser().resolve()
    if args.data_root: args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve(); args.output_dir.mkdir(parents=True,exist_ok=True)
    if any(not 0 < f < 1 for f in args.row_fracs) or sorted(args.row_fracs) != args.row_fracs:
        raise ValueError("--row-fracs must be four increasing values between 0 and 1")
    rr = roots(args)
    gt_dir = resolve_gt(args.gt_dir, rr)
    sparse_dir = resolve_npy_dir(args.one_line_dir, rr, ["v2_1_sensor","v2_sensor"], "one-line sparse maps")
    da3_dir = resolve_npy_dir(args.da3_dir, rr, ["da3_bridge_all"], "cached DA3 predictions")
    print(f"GT: {gt_dir}\none-line: {sparse_dir}\nDA3: {da3_dir}\noutput: {args.output_dir}")
    poisson = load_poisson(args.da3_root)
    scenes = sorted(p.stem for p in gt_dir.glob("*.mat"))
    if args.scene: scenes = [s for s in scenes if s == args.scene]
    if args.limit is not None: scenes = scenes[:args.limit]
    if not scenes: raise RuntimeError("No scenes selected")
    csv_path = args.output_dir/"per_scene_metrics.csv"
    old = read_csv(csv_path) if args.resume else []
    rows: list[dict[str,Any]] = list(old); completed = {r["scene"] for r in old}
    pred_dir = args.output_dir/"predictions_m"; pred_dir.mkdir(exist_ok=True)
    for i, scene in enumerate(scenes,1):
        if args.resume and scene in completed and (pred_dir/f"{scene}.npy").is_file():
            print(f"[{i:3d}/{len(scenes)}] {scene} resume-skip"); continue
        gt, valid, _ = load_ibims(gt_dir/f"{scene}.mat")
        if args.eval_max_depth_m > 0: valid &= gt <= args.eval_max_depth_m
        old_sparse = load_npy(npy_path(sparse_dir,scene),gt.shape)
        rel = load_npy(npy_path(da3_dir,scene,True),gt.shape)
        sparse, anchors, line_rows, ncols = four_lines(gt,valid,old_sparse,args.row_fracs,args.sensor_min_depth_m,args.sensor_max_depth_m)
        base, scale = median_align(rel,sparse,anchors)
        pred, diag = call_poisson(poisson,base,sparse,anchors,args.rtol,args.maxiter)
        met = metrics(pred,gt,valid)
        row = {"scene":scene,"method":METHOD,"region":"all_valid","anchor_count":int(anchors.sum()),"source_x_column_count":ncols,"line_rows":";".join(map(str,line_rows)),"median_scale":scale,**met}
        rows = [r for r in rows if r.get("scene") != scene]; rows.append(row); rows.sort(key=lambda r:r["scene"])
        np.save(pred_dir/f"{scene}.npy",pred.astype(np.float32)); write_csv(csv_path,rows)
        print(f"[{i:3d}/{len(scenes)}] {scene} anchors={anchors.sum():4d} AbsRel={met['absrel_pct']:7.3f}% RMSE={met['rmse_m']:.3f}m MAE={met['mae_m']:.3f}m",flush=True)
    chosen_rows = [r for r in rows if r["scene"] in set(scenes)]
    summary(chosen_rows,args)
    ranked = sorted(chosen_rows,key=lambda r:float(r["absrel_pct"])); med=float(np.median([float(r["absrel_pct"]) for r in ranked])); typical=min(ranked,key=lambda r:abs(float(r["absrel_pct"])-med))
    ex = [("best",ranked[0]),("typical",typical),("worst",ranked[-1])]
    out_ex=args.output_dir/"examples_best_typical_worst"; probe_rows=[]
    for role,row in ex:
        scene=row["scene"]; gt,valid,rgb=load_ibims(gt_dir/f"{scene}.mat")
        if args.eval_max_depth_m > 0: valid &= gt <= args.eval_max_depth_m
        old_sparse=load_npy(npy_path(sparse_dir,scene),gt.shape)
        sparse,anchors,line_rows,_=four_lines(gt,valid,old_sparse,args.row_fracs,args.sensor_min_depth_m,args.sensor_max_depth_m)
        pred=load_npy(pred_dir/f"{scene}.npy",gt.shape); ps=probes(gt,pred,valid,anchors)
        for q in ps: probe_rows.append({"role":role,"scene":scene,"patch_size":7,**q})
        panel(role,scene,rgb,gt,pred,valid,sparse,line_rows,ps,{k:float(row[k]) for k in ("absrel_pct","rmse_m","mae_m")},out_ex/f"{role}__{scene}__metric_depth_and_surface_probes.png",args.plot_max_depth_m,args.plot_error_max_m)
    write_csv(out_ex/"surface_probes.csv",probe_rows)
    print("\n===== FOUR-LINE RESULT =====\n"+(args.output_dir/"summary.txt").read_text())
    print(f"Open: {out_ex}")


if __name__ == "__main__":
    main()
