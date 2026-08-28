"""
Per-surface (superpixel-level) depth evaluation for iBims-1.

For each scene: segment the RGB into ~100 superpixels using SLIC, compute
the median predicted depth and median GT depth within each superpixel, and
report per-surface error distributions across all methods.

This isolates the deployment-relevant question: "how many surfaces in the
scene are estimated accurately enough for a robot to act on?"
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.segmentation import slic

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from depth_anything_3.alignment.poisson_alignment import poisson_align


# ---------- loaders (same as compare script) ----------

def load_gt(path):
    rec = loadmat(path)["data"][0, 0]
    gt = np.squeeze(rec["depth"]).astype(np.float64)
    valid = np.squeeze(rec["mask_invalid"]).astype(bool)
    valid &= np.squeeze(rec["mask_transp"]).astype(bool)
    valid &= np.isfinite(gt) & (gt > 0)
    return gt, valid


def load_array(p, expected_shape=None):
    a = np.squeeze(np.load(p)).astype(np.float64)
    if expected_shape and a.shape != expected_shape:
        raise ValueError(f"{p} shape {a.shape} != {expected_shape}")
    return a


def load_rgb(rgb_dir, scene):
    for ext in (".png", ".jpg", ".jpeg"):
        p = rgb_dir / f"{scene}{ext}"
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    raise FileNotFoundError(f"No RGB for {scene}")


def resolve_da3_path(da3_dir, scene):
    for cand in (da3_dir / f"{scene}_da3small.npy", da3_dir / f"{scene}.npy"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No DA3 prediction for {scene}")


# ---------- alignment (mirrors compare_any2full_da3_100.py) ----------

def fit_median_scale(rel_anchor, metric_anchor):
    m = np.isfinite(rel_anchor) & (rel_anchor > 0)
    return float(np.median(metric_anchor[m] / rel_anchor[m]))


# ---------- superpixel-level evaluation ----------

def per_superpixel_errors(pred, gt, valid, segments, min_valid_pixels=30):
    """
    For each superpixel with enough valid GT pixels, compute:
        median GT depth, median predicted depth, |med_pred-med_gt|/med_gt.
    Returns list of dicts, one per usable superpixel.
    """
    records = []
    valid_pred = valid & np.isfinite(pred) & (pred > 0)

    for sp_id in np.unique(segments):
        m = (segments == sp_id) & valid_pred
        if m.sum() < min_valid_pixels:
            continue
        med_gt = float(np.median(gt[m]))
        med_pred = float(np.median(pred[m]))
        if med_gt <= 0:
            continue
        records.append({
            "superpixel_id": int(sp_id),
            "n_pixels": int(m.sum()),
            "median_gt_m": med_gt,
            "median_pred_m": med_pred,
            "absrel": abs(med_pred - med_gt) / med_gt,
            "signed_err_m": med_pred - med_gt,
        })
    return records


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path,
                    default=Path(os.environ["DA3_LIDAR_DATA_ROOT"])
                            if "DA3_LIDAR_DATA_ROOT" in os.environ else None,
                    required="DA3_LIDAR_DATA_ROOT" not in os.environ)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-superpixels", type=int, default=100,
                    help="Target number of superpixels per image (SLIC).")
    ap.add_argument("--compactness", type=float, default=10.0,
                    help="SLIC compactness: higher = more square-shaped.")
    ap.add_argument("--min-valid-pixels", type=int, default=30,
                    help="Skip superpixels with fewer valid GT pixels than this.")
    ap.add_argument("--poisson-maxiter", type=int, default=5000)
    ap.add_argument("--bad-threshold", type=float, default=0.30,
                    help="A superpixel with AbsRel > this is 'bad' (deployment-critical).")
    args = ap.parse_args()

    base = args.data_root / "experiments/ibims_replication"
    dirs = {
        "gt":       args.data_root / "datasets/ibims1/ibims1_core_mat",
        "rgb":      args.data_root / "datasets/ibims1/ibims1_core_raw/rgb",
        "sparse":   base / "v2_sensor",
        "any2full": base / "predictions_v2_sensor",
        "da3":      base / "da3_bridge_all",
    }
    for k, p in dirs.items():
        if not p.is_dir():
            raise FileNotFoundError(f"{k}: {p}")

    scenes = sorted({p.stem for p in dirs["gt"].glob("*.mat")})
    print(f"Evaluating {len(scenes)} scenes")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per-method superpixel records across all scenes.
    per_method_records = {
        "any2full":            [],
        "da3_median":          [],
        "da3_median_poisson":  [],
    }

    for i, scene in enumerate(scenes, start=1):
        try:
            gt, valid = load_gt(dirs["gt"] / f"{scene}.mat")
            rgb = load_rgb(dirs["rgb"], scene)
            sparse = load_array(dirs["sparse"] / f"{scene}.npy", gt.shape)
            any2full = load_array(dirs["any2full"] / f"{scene}.npy", gt.shape)
            da3_raw = load_array(resolve_da3_path(dirs["da3"], scene), gt.shape)
        except Exception as e:
            print(f"  [{i:3d}/{len(scenes)}] {scene}: SKIP ({e})"); continue

        anchor_mask = np.isfinite(sparse) & (sparse > 0)
        if anchor_mask.sum() == 0:
            continue

        # DA3 + median-scale.
        scale = fit_median_scale(da3_raw[anchor_mask], sparse[anchor_mask])
        da3_median = scale * da3_raw

        # DA3 + median + Poisson.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                da3_med_poi, _ = poisson_align(
                    da3_median, sparse, anchor_mask,
                    maxiter=args.poisson_maxiter, rtol=1e-6)
        except Exception:
            da3_med_poi = da3_median  # fall back; rare

        # Superpixel segmentation on the RGB (deterministic, no learning).
        segments = slic(rgb, n_segments=args.n_superpixels,
                        compactness=args.compactness,
                        start_label=0, channel_axis=-1)

        method_preds = {
            "any2full":           any2full,
            "da3_median":         da3_median,
            "da3_median_poisson": da3_med_poi,
        }
        for name, pred in method_preds.items():
            recs = per_superpixel_errors(pred, gt, valid, segments,
                                         min_valid_pixels=args.min_valid_pixels)
            for r in recs:
                r["scene"] = scene
                per_method_records[name].append(r)

        if i % 10 == 0 or i == len(scenes):
            print(f"  [{i:3d}/{len(scenes)}] done")

    # ---------- report ----------

    lines = [
        "Per-surface depth evaluation on iBims-1 (superpixel-level)",
        f"SLIC parameters: n_segments={args.n_superpixels}, "
        f"compactness={args.compactness}, "
        f"min valid pixels per surface = {args.min_valid_pixels}",
        f"'Bad surface' threshold: AbsRel > {args.bad_threshold*100:.0f}%",
        "",
    ]

    for name, recs in per_method_records.items():
        if not recs:
            lines.append(f"{name}: no records"); continue

        errs = np.array([r["absrel"] for r in recs])
        signed = np.array([r["signed_err_m"] for r in recs])

        within_5  = 100 * np.mean(errs <= 0.05)
        within_10 = 100 * np.mean(errs <= 0.10)
        within_20 = 100 * np.mean(errs <= 0.20)
        bad       = 100 * np.mean(errs >  args.bad_threshold)

        # Worst per-scene surface.
        by_scene = {}
        for r in recs:
            by_scene.setdefault(r["scene"], []).append(r["absrel"])
        worst_per_scene = np.array([max(v) for v in by_scene.values()])
        mean_worst = 100 * worst_per_scene.mean()

        lines.append(f"=== {name} ===")
        lines.append(f"  Total surfaces evaluated: {len(recs)}")
        lines.append(f"  Mean AbsRel per surface:   {100*errs.mean():.2f}%")
        lines.append(f"  Median AbsRel per surface: {100*np.median(errs):.2f}%")
        lines.append(f"  Mean signed error:         {signed.mean():+.3f} m "
                     f"(positive = predicted too far)")
        lines.append(f"  Surfaces within  5%: {within_5:.1f}%")
        lines.append(f"  Surfaces within 10%: {within_10:.1f}%")
        lines.append(f"  Surfaces within 20%: {within_20:.1f}%")
        lines.append(f"  Bad surfaces (>{args.bad_threshold*100:.0f}%): {bad:.1f}%")
        lines.append(f"  Mean worst-surface AbsRel per scene: {mean_worst:.2f}%")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    (args.output_dir / "surface_report.txt").write_text(report)

    # Also dump raw records for offline plotting / deeper analysis.
    import csv
    with (args.output_dir / "per_surface_records.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "scene", "superpixel_id", "n_pixels",
                         "median_gt_m", "median_pred_m", "absrel", "signed_err_m"])
        for method, recs in per_method_records.items():
            for r in recs:
                writer.writerow([method, r["scene"], r["superpixel_id"],
                                 r["n_pixels"], r["median_gt_m"],
                                 r["median_pred_m"], r["absrel"],
                                 r["signed_err_m"]])
    print(f"\nWrote: {args.output_dir/'surface_report.txt'}"
          f"\n       {args.output_dir/'per_surface_records.csv'}")


if __name__ == "__main__":
    main()
