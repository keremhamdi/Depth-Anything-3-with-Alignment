
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from depth_anything_3.alignment.poisson_alignment import poisson_align


def load_sparse_from_csv(csv_path, H, W):
    sparse = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("u_px") or not row.get("v_px"):
                continue
            try:
                u = int(round(float(row["u_px"])))
                v = int(round(float(row["v_px"])))
                r_mm = float(row["range_mm"])
            except ValueError:
                continue
            if not (0 <= u < W and 0 <= v < H):
                continue
            depth_m = r_mm / 1000.0
            if not (0.1 <= depth_m <= 12.0):
                continue
            sparse[v, u] = depth_m
            mask[v, u] = True
    return sparse, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path,
                    help="dataset_YYYYMMDD_HHMMSS folder")
    ap.add_argument("--nominal-dir", required=True, type=Path,
                    help="da3_median_poisson_nominal folder from previous run")
    ap.add_argument("--frame", required=True,
                    help="Frame stem, e.g. 00007_2032237800000")
    ap.add_argument("--anchor-weights", nargs="+", type=float,
                    default=[10, 100, 1000],
                    help="Anchor weights to try")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs.
    rgb_path = args.dataset / "cam_rgb" / f"{args.frame}.png"
    csv_path = args.dataset / "lidar_csv" / f"{args.frame}.csv"
    med_path = args.nominal_dir / "full_predictions_m" / f"{args.frame}.npy"

    if not med_path.exists():
        # Try other locations.
        found = list(args.nominal_dir.rglob(f"{args.frame}*.npy"))
        if not found:
            raise FileNotFoundError(f"No cached prediction found for {args.frame}")
        med_path = found[0]
        print(f"Using cached prediction: {med_path}")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    H, W = rgb.shape[:2]
    depth_med = np.load(med_path).astype(np.float64)
    if depth_med.shape != (H, W):
        raise ValueError(f"Cached prediction shape {depth_med.shape} != RGB {(H,W)}")

    sparse, anchor_mask = load_sparse_from_csv(csv_path, H, W)
    print(f"Frame {args.frame}: {anchor_mask.sum()} anchors, "
          f"depth range {sparse[anchor_mask].min():.2f}–{sparse[anchor_mask].max():.2f} m")

    # Run Poisson with each weight.
    results = {"median (no Poisson)": depth_med}
    for w in args.anchor_weights:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refined, _ = poisson_align(
                base_depth=depth_med,
                sparse_depth=sparse,
                anchor_mask=anchor_mask,
                anchor_weight=w,
                edge_aware=True,
                maxiter=5000,
                rtol=1e-6,
            )
        results[f"Poisson w={w:g}"] = refined
        anchor_rmse = np.sqrt(np.mean((refined[anchor_mask] - sparse[anchor_mask]) ** 2))
        print(f"  anchor_weight={w:>7g}: anchor RMSE {anchor_rmse:.3f} m")

    # Save comparison panel.
    n = len(results) + 1  # +1 for RGB
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    ys, xs = np.where(anchor_mask)
    axes[0].scatter(xs, ys, s=6, c="cyan", edgecolors="black", linewidths=0.3)
    axes[0].set_title(f"RGB + {anchor_mask.sum()} anchors")
    axes[0].set_axis_off()

    finite = depth_med[np.isfinite(depth_med) & (depth_med > 0)]
    vmax = float(np.clip(np.percentile(finite, 99.0), 3.0, 15.0))

    for i, (name, d) in enumerate(results.items(), start=1):
        axes[i].imshow(d, cmap="turbo", vmin=0, vmax=vmax)
        axes[i].set_title(name)
        axes[i].set_axis_off()

    fig.suptitle(f"Anchor-weight sweep — {args.frame}", fontsize=14)
    out_path = args.out_dir / f"{args.frame}_weight_sweep.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

