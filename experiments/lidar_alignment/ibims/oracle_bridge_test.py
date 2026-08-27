import numpy as np
from scipy.ndimage import distance_transform_edt

GT = "experiments/ibims_replication/oracle_inputs/lectureroom_01_gt.npy"
A2F = "experiments/ibims_replication/da3_bridge_a2f_rel/lectureroom_01_rel.npy"
DA3 = "experiments/ibims_replication/da3_bridge/lectureroom_01_da3small.npy"
SPARSE = "experiments/ibims_replication/v2_1_sensor/lectureroom_01.npy"


def fit_affine(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def metrics(pred, gt, mask):
    p = pred[mask]
    g = gt[mask]
    e = p - g

    return {
        "rmse": np.sqrt(np.mean(e ** 2)),
        "mae": np.mean(np.abs(e)),
        "absrel": 100.0 * np.mean(np.abs(e) / g),
        "rmsrel": 100.0 * np.sqrt(np.mean((e / g) ** 2)),
        "bias": np.mean(e),
    }


def print_metrics(name, pred, gt, mask):
    m = metrics(pred, gt, mask)

    print(
        f"{name:25s} "
        f"RMSE={m['rmse']:.3f} m  "
        f"MAE={m['mae']:.3f} m  "
        f"AbsRel={m['absrel']:.2f}%  "
        f"RMSRel={m['rmsrel']:.2f}%  "
        f"Bias={m['bias']:+.3f} m"
    )

    return m


gt = np.load(GT).astype(np.float64)
r = np.load(A2F).astype(np.float64)
da3 = np.load(DA3).astype(np.float64)
sparse = np.load(SPARSE).astype(np.float64)

print("Shapes:")
print(" GT    :", gt.shape)
print(" A2F   :", r.shape)
print(" DA3   :", da3.shape)
print(" Sparse:", sparse.shape)

if not (gt.shape == r.shape == da3.shape == sparse.shape):
    raise RuntimeError("Shape mismatch.")

valid = (
    np.isfinite(gt) &
    (gt > 0) &
    np.isfinite(r) &
    np.isfinite(da3) &
    (r > 1e-8)
)

print("\nCommon valid pixels:", valid.sum())
print("Coverage: %.2f%%" % (100 * valid.mean()))


# --------------------------------------------------
# ROW 2
# A2F raw disparity -> oracle affine in disparity
#
# 1 / GT = s*r + t
# --------------------------------------------------

gt_disp = 1.0 / gt[valid]

s2, t2 = fit_affine(
    r[valid],
    gt_disp,
)

aligned_disp = s2 * r + t2

row2 = np.full_like(gt, np.nan)

valid2 = (
    valid &
    np.isfinite(aligned_disp) &
    (aligned_disp > 1e-8)
)

row2[valid2] = 1.0 / aligned_disp[valid2]


# --------------------------------------------------
# ROW 3
# Same exact A2F prior, but depth-domain bridge
#
# relative depth = 1 / disparity_pre
# GT = a*relative_depth + b
# --------------------------------------------------

a2f_rel_depth = np.full_like(r, np.nan)
a2f_rel_depth[valid] = 1.0 / r[valid]

s3, t3 = fit_affine(
    a2f_rel_depth[valid],
    gt[valid],
)

row3 = s3 * a2f_rel_depth + t3


# --------------------------------------------------
# ROW 5
# DA3-Small relative depth -> GT oracle affine
#
# GT = a*DA3 + b
# --------------------------------------------------

s5, t5 = fit_affine(
    da3[valid],
    gt[valid],
)

row5 = s5 * da3 + t5


# --------------------------------------------------
# Convention sanity check
# --------------------------------------------------

corr = np.corrcoef(
    da3[valid],
    gt[valid],
)[0, 1]

print("\nDA3 / GT Pearson correlation:", f"{corr:.4f}")
print("DA3 oracle slope:", f"{s5:.6f}")

if corr <= 0 or s5 <= 0:
    print("WARNING: DA3 depth direction/convention looks suspicious.")


# --------------------------------------------------
# Global results
# --------------------------------------------------

common_eval = valid & valid2

print("\n================ GLOBAL ================")

m2 = print_metrics(
    "Row 2 A2F disparity",
    row2,
    gt,
    common_eval,
)

m3 = print_metrics(
    "Row 3 A2F depth bridge",
    row3,
    gt,
    common_eval,
)

m5 = print_metrics(
    "Row 5 DA3-Small depth",
    row5,
    gt,
    common_eval,
)


print("\nOracle parameters:")

print(
    f"Row 2: 1/D = {s2:.8f} * r + {t2:.8f}"
)

print(
    f"Row 3: D = {s3:.8f} * (1/r) + {t3:.8f}"
)

print(
    f"Row 5: D = {s5:.8f} * DA3 + {t5:.8f}"
)


# --------------------------------------------------
# GT depth stratification
# --------------------------------------------------

print("\n============= BY GT DEPTH =============")

depth_bands = [
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
]

for lo, hi in depth_bands:

    band = (
        common_eval &
        (gt >= lo) &
        (gt < hi)
    )

    if band.sum() < 20:
        continue

    print(
        f"\nGT {lo}-{hi} m   pixels={band.sum()}"
    )

    print_metrics("  Row 2 A2F disp", row2, gt, band)
    print_metrics("  Row 3 A2F depth", row3, gt, band)
    print_metrics("  Row 5 DA3", row5, gt, band)


# --------------------------------------------------
# Distance from ACTUAL projected LiDAR support
# --------------------------------------------------

lidar_mask = sparse > 0

distance_px = distance_transform_edt(
    ~lidar_mask
)

distance_norm = distance_px / gt.shape[0]

print("\n========== DISTANCE FROM LIDAR =========")

distance_bands = [
    (0.00, 0.05),
    (0.05, 0.15),
    (0.15, 0.30),
    (0.30, 2.00),
]

for lo, hi in distance_bands:

    band = (
        common_eval &
        (distance_norm >= lo) &
        (distance_norm < hi)
    )

    if band.sum() < 20:
        continue

    print(
        f"\nDistance {100*lo:.0f}-{100*hi:.0f}% image-height"
        f"   pixels={band.sum()}"
    )

    print_metrics("  Row 2 A2F disp", row2, gt, band)
    print_metrics("  Row 3 A2F depth", row3, gt, band)
    print_metrics("  Row 5 DA3", row5, gt, band)


# --------------------------------------------------
# Useful deltas
# --------------------------------------------------

print("\n================ DELTAS ================")

print(
    "Row2 - Row3 AbsRel:",
    f"{m2['absrel'] - m3['absrel']:+.2f} percentage points"
)

print(
    "Row3 - Row5 AbsRel:",
    f"{m3['absrel'] - m5['absrel']:+.2f} percentage points"
)
