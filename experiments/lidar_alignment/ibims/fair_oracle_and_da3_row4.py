import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt

GT = "experiments/ibims_replication/oracle_inputs/lectureroom_01_gt.npy"
MASK = "experiments/ibims_replication/oracle_inputs/lectureroom_01_mask_invalid.npy"
A2F = "experiments/ibims_replication/da3_bridge_a2f_rel/lectureroom_01_rel.npy"
DA3 = "experiments/ibims_replication/da3_bridge/lectureroom_01_da3small.npy"
SPARSE = "experiments/ibims_replication/v2_1_sensor/lectureroom_01.npy"


def ls_affine(x, y):
    A = np.column_stack((x, np.ones_like(x)))
    return np.linalg.lstsq(A, y, rcond=None)[0]


def metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    e = p - g
    return (
        np.sqrt(np.mean(e**2)),
        np.mean(np.abs(e)),
        100 * np.mean(np.abs(e) / g),
        np.mean(e),
    )


def show(name, pred, gt, mask):
    rmse, mae, ar, bias = metrics(pred, gt, mask)
    print(f"{name:25s} RMSE={rmse:.3f}  MAE={mae:.3f}  "
          f"AbsRel={ar:.2f}%  Bias={bias:+.3f}")
    return rmse, ar


# Positive affine function fitted with LOG-DEPTH loss.
# reciprocal=True means D = 1 / (s*x+t).
def fit_log(x, y, reciprocal=False):
    xmin, xmax = x.min(), x.max()
    u = (x - xmin) / (xmax - xmin)

    target = 1 / y if reciprocal else y
    a, b = ls_affine(x, target)

    v0 = max(a * xmin + b, 1e-6)
    v1 = max(a * xmax + b, 1e-6)

    def predict(theta):
        q0, q1 = np.exp(theta)
        q = q0 * (1-u) + q1 * u
        return 1/q if reciprocal else q

    def objective(theta):
        pred = predict(theta)
        return np.mean((np.log(pred) - np.log(y))**2)

    result = minimize(
        objective,
        np.log([v0, v1]),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12},
    )

    return xmin, xmax, np.exp(result.x), np.sqrt(result.fun)


def apply_log(x, fit, reciprocal=False):
    xmin, xmax, endpoints, _ = fit
    u = (x - xmin) / (xmax - xmin)
    q = endpoints[0] * (1-u) + endpoints[1] * u
    return 1/q if reciprocal else q


gt = np.load(GT).astype(np.float64)
mask = np.load(MASK)
r = np.load(A2F).astype(np.float64)
da3 = np.load(DA3).astype(np.float64)
sparse = np.load(SPARSE).astype(np.float64)

valid_gt = np.isfinite(gt) & (gt > 0) & (mask > 0)
valid_da3 = valid_gt & np.isfinite(da3)
valid_a2f = valid_da3 & np.isfinite(r) & (r > 1e-8)


# ============================================================
# TEST A: FAIR ROW 2 vs ROW 3
# SAME log-depth objective
# ============================================================

fit2 = fit_log(r[valid_a2f], gt[valid_a2f], reciprocal=True)
fit3 = fit_log(1/r[valid_a2f], gt[valid_a2f], reciprocal=False)

row2 = np.full_like(gt, np.nan)
row3 = np.full_like(gt, np.nan)

row2[valid_a2f] = apply_log(r[valid_a2f], fit2, reciprocal=True)
row3[valid_a2f] = apply_log(1/r[valid_a2f], fit3, reciprocal=False)

print("\n========== FAIR LOG-DEPTH ORACLE ==========")
m2 = show("Row 2 A2F disparity", row2, gt, valid_a2f)
m3 = show("Row 3 A2F depth", row3, gt, valid_a2f)

print(f"Row 2 log-RMSE: {fit2[3]:.6f}")
print(f"Row 3 log-RMSE: {fit3[3]:.6f}")
print(f"Row2-Row3 AbsRel delta: {m2[1]-m3[1]:+.2f} pp")


# ============================================================
# TEST B: DA3 ONE-LINE vs DA3 ORACLE
# ============================================================

anchors = (
    np.isfinite(sparse) &
    (sparse > 0) &
    np.isfinite(da3)
)

xL = da3[anchors]
yL = sparse[anchors]

s4, t4 = ls_affine(xL, yL)
row4 = s4 * da3 + t4

s5, t5 = ls_affine(da3[valid_da3], gt[valid_da3])
row5 = s5 * da3 + t5

A = np.column_stack((xL, np.ones_like(xL)))

print("\n============= LIDAR ANCHORS =============")
print("Anchor count:", anchors.sum())
print(f"LiDAR depth range: {yL.min():.3f} - {yL.max():.3f} m")
print(f"LiDAR depth std:   {yL.std():.6f} m")
print(f"DA3 anchor range:  {xL.min():.6f} - {xL.max():.6f}")
print(f"DA3 anchor std:    {xL.std():.6f}")
print(f"Design condition:  {np.linalg.cond(A):.3f}")
print(f"Row 4 fit: D = {s4:.6f} * DA3 + {t4:.6f}")
print(f"Row 5 fit: D = {s5:.6f} * DA3 + {t5:.6f}")

print("\n========== DA3 GLOBAL ==========")
m4 = show("Row 4 DA3 one-line", row4, gt, valid_da3)
m5 = show("Row 5 DA3 oracle", row5, gt, valid_da3)

print(f"Scaling penalty AbsRel: {m4[1]-m5[1]:+.2f} pp")
print(f"Scaling penalty RMSE:   {m4[0]-m5[0]:+.3f} m")

print("Row 4 non-positive depth:",
      f"{100*np.mean(row4[valid_da3] <= 0):.3f}%")


# ============================================================
# DEPTH BANDS
# ============================================================

print("\n========== DA3 BY GT DEPTH ==========")

for lo, hi in [(0,2), (2,5), (5,8), (8,11)]:
    b = valid_da3 & (gt >= lo) & (gt < hi)
    print(f"\nGT {lo}-{hi} m   pixels={b.sum()}")
    show("  Row 4 one-line", row4, gt, b)
    show("  Row 5 oracle", row5, gt, b)


# ============================================================
# DISTANCE FROM ACTUAL LIDAR SUPPORT
# ============================================================

dist = distance_transform_edt(~anchors) / gt.shape[0]

print("\n====== DA3 BY DISTANCE FROM LIDAR ======")

for lo, hi in [(0,.05), (.05,.15), (.15,.30), (.30,2.0)]:
    b = valid_da3 & (dist >= lo) & (dist < hi)
    print(f"\nDistance {100*lo:.0f}-{100*hi:.0f}%   pixels={b.sum()}")
    show("  Row 4 one-line", row4, gt, b)
    show("  Row 5 oracle", row5, gt, b)
