import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt

from depth_anything_3.alignment.poisson_alignment import poisson_align
from experiment_paths import parse_experiment_paths


DATA_ROOT, _ = parse_experiment_paths(
    description="Evaluate DA3 and Poisson alignment on iBims lectureroom_01."
)

GT = DATA_ROOT / "experiments/ibims_replication/oracle_inputs/lectureroom_01_gt.npy"
MASK = (
    DATA_ROOT
    / "experiments/ibims_replication/oracle_inputs/lectureroom_01_mask_invalid.npy"
)
A2F = (
    DATA_ROOT
    / "experiments/ibims_replication/da3_bridge_a2f_rel/lectureroom_01_rel.npy"
)
DA3 = (
    DATA_ROOT
    / "experiments/ibims_replication/da3_bridge/lectureroom_01_da3small.npy"
)
SPARSE = (
    DATA_ROOT
    / "experiments/ibims_replication/v2_1_sensor/lectureroom_01.npy"
)


# ============================================================
# BASIC FITTING AND EVALUATION FUNCTIONS
# ============================================================

def ls_affine(x, y):
    matrix = np.column_stack(
        (x, np.ones_like(x))
    )

    return np.linalg.lstsq(
        matrix,
        y,
        rcond=None,
    )[0]


def metrics(prediction, ground_truth, evaluation_mask):
    prediction_values = prediction[evaluation_mask]
    ground_truth_values = ground_truth[evaluation_mask]

    error = (
        prediction_values
        - ground_truth_values
    )

    return (
        np.sqrt(np.mean(error**2)),
        np.mean(np.abs(error)),
        100.0 * np.mean(
            np.abs(error) / ground_truth_values
        ),
        np.mean(error),
    )


def show(
    name,
    prediction,
    ground_truth,
    evaluation_mask,
):
    rmse, mae, absrel, bias = metrics(
        prediction,
        ground_truth,
        evaluation_mask,
    )

    print(
        f"{name:28s} "
        f"RMSE={rmse:.3f}  "
        f"MAE={mae:.3f}  "
        f"AbsRel={absrel:.2f}%  "
        f"Bias={bias:+.3f}"
    )

    return rmse, absrel


# ============================================================
# POSITIVE AFFINE FIT WITH LOG-DEPTH OBJECTIVE
# ============================================================

def fit_log(x, y, reciprocal=False):
    xmin = x.min()
    xmax = x.max()

    if xmax <= xmin:
        raise ValueError(
            "Cannot fit an affine model because x is constant."
        )

    normalized_x = (
        (x - xmin)
        / (xmax - xmin)
    )

    target = (
        1.0 / y
        if reciprocal
        else y
    )

    initial_slope, initial_shift = ls_affine(
        x,
        target,
    )

    initial_min = max(
        initial_slope * xmin + initial_shift,
        1e-6,
    )

    initial_max = max(
        initial_slope * xmax + initial_shift,
        1e-6,
    )

    def predict(theta):
        endpoint_min, endpoint_max = np.exp(theta)

        affine_value = (
            endpoint_min * (1.0 - normalized_x)
            + endpoint_max * normalized_x
        )

        if reciprocal:
            return 1.0 / affine_value

        return affine_value

    def objective(theta):
        prediction = predict(theta)

        return np.mean(
            (
                np.log(prediction)
                - np.log(y)
            ) ** 2
        )

    result = minimize(
        objective,
        np.log(
            [initial_min, initial_max]
        ),
        method="Nelder-Mead",
        options={
            "maxiter": 2000,
            "xatol": 1e-10,
            "fatol": 1e-12,
        },
    )

    return (
        xmin,
        xmax,
        np.exp(result.x),
        np.sqrt(result.fun),
    )


def apply_log(x, fit, reciprocal=False):
    xmin, xmax, endpoints, _ = fit

    normalized_x = (
        (x - xmin)
        / (xmax - xmin)
    )

    affine_value = (
        endpoints[0] * (1.0 - normalized_x)
        + endpoints[1] * normalized_x
    )

    if reciprocal:
        return 1.0 / affine_value

    return affine_value


# ============================================================
# CROSS-TABLE FUNCTIONS
# ============================================================

def cross_cell_metrics(
    prediction,
    ground_truth,
    cell_mask,
):
    count = int(cell_mask.sum())

    if count < 20:
        return {
            "n": count,
            "absrel": np.nan,
            "p90": np.nan,
            "worst": np.nan,
        }

    relative_error = (
        np.abs(
            prediction[cell_mask]
            - ground_truth[cell_mask]
        )
        / ground_truth[cell_mask]
    )

    return {
        "n": count,
        "absrel": (
            100.0 * np.mean(relative_error)
        ),
        "p90": (
            100.0
            * np.percentile(relative_error, 90)
        ),
        "worst": (
            100.0 * np.max(relative_error)
        ),
    }


def format_depth_band(lo, hi):
    if np.isinf(hi):
        return f"{lo:g}+ m"

    return f"{lo:g}-{hi:g} m"


def format_cross_cell(result):
    if np.isnan(result["absrel"]):
        return f"-- n={result['n']}"

    return (
        f"{result['absrel']:.1f}/"
        f"{result['p90']:.1f}/"
        f"{result['worst']:.0f} "
        f"n={result['n']}"
    )


# ============================================================
# LOAD INPUTS
# ============================================================

gt = np.load(GT).astype(np.float64)
mask = np.load(MASK)

relative_a2f = np.load(
    A2F
).astype(np.float64)

da3 = np.load(
    DA3
).astype(np.float64)

sparse = np.load(
    SPARSE
).astype(np.float64)


if not (
    gt.shape
    == mask.shape
    == relative_a2f.shape
    == da3.shape
    == sparse.shape
):
    raise RuntimeError(
        "GT, mask, A2F, DA3 and sparse depth shapes do not match."
    )


valid_gt = (
    np.isfinite(gt)
    & (gt > 0)
    & (mask > 0)
)

valid_da3 = (
    valid_gt
    & np.isfinite(da3)
)

valid_a2f = (
    valid_da3
    & np.isfinite(relative_a2f)
    & (relative_a2f > 1e-8)
)


# ============================================================
# TEST A: FAIR ROW 2 VERSUS ROW 3
# SAME LOG-DEPTH ORACLE OBJECTIVE
# ============================================================

fit2 = fit_log(
    relative_a2f[valid_a2f],
    gt[valid_a2f],
    reciprocal=True,
)

fit3 = fit_log(
    1.0 / relative_a2f[valid_a2f],
    gt[valid_a2f],
    reciprocal=False,
)

row2 = np.full_like(
    gt,
    np.nan,
)

row3 = np.full_like(
    gt,
    np.nan,
)

row2[valid_a2f] = apply_log(
    relative_a2f[valid_a2f],
    fit2,
    reciprocal=True,
)

row3[valid_a2f] = apply_log(
    1.0 / relative_a2f[valid_a2f],
    fit3,
    reciprocal=False,
)

print(
    "\n========== FAIR LOG-DEPTH ORACLE =========="
)

m2 = show(
    "Row 2 A2F disparity",
    row2,
    gt,
    valid_a2f,
)

m3 = show(
    "Row 3 A2F depth",
    row3,
    gt,
    valid_a2f,
)

print(
    f"Row 2 log-RMSE: {fit2[3]:.6f}"
)

print(
    f"Row 3 log-RMSE: {fit3[3]:.6f}"
)

print(
    "Row2-Row3 AbsRel delta:",
    f"{m2[1] - m3[1]:+.2f} pp",
)


# ============================================================
# TEST B: DA3 ONE-LINE, POISSON AND ORACLE
# ============================================================

anchors = (
    np.isfinite(sparse)
    & (sparse > 0)
    & np.isfinite(da3)
)

if anchors.sum() < 2:
    raise RuntimeError(
        "Fewer than two valid LiDAR anchors were found."
    )


da3_at_lidar = da3[anchors]
lidar_depth = sparse[anchors]


# Row 4: deployment-realistic global affine from LiDAR only.
scale4, shift4 = ls_affine(
    da3_at_lidar,
    lidar_depth,
)

row4 = (
    scale4 * da3
    + shift4
)


# Poisson correction after global affine calibration.
print("\nSolving uniform Poisson...")

row6_uniform, uniform_info = poisson_align(
    row4,
    sparse,
    anchors,
    edge_aware=False,
)

print("Solving edge-aware Poisson...")

row6_edge, edge_info = poisson_align(
    row4,
    sparse,
    anchors,
    edge_aware=True,
)


# Row 5: diagnostic global-affine oracle using dense GT.
scale5, shift5 = ls_affine(
    da3[valid_da3],
    gt[valid_da3],
)

row5 = (
    scale5 * da3
    + shift5
)


design_matrix = np.column_stack(
    (
        da3_at_lidar,
        np.ones_like(da3_at_lidar),
    )
)


# ============================================================
# CURRENT-FRAME GT AND LIDAR DEPTH SUPPORT
# ============================================================

gt_values = gt[valid_gt]

print(
    "\n========== GT VERSUS LIDAR DEPTH SUPPORT =========="
)

print(
    "GT depth min/median/p95/p99/max:",
    f"{gt_values.min():.3f} / "
    f"{np.median(gt_values):.3f} / "
    f"{np.percentile(gt_values, 95):.3f} / "
    f"{np.percentile(gt_values, 99):.3f} / "
    f"{gt_values.max():.3f} m",
)

print(
    "LiDAR depth min/median/p95/max:",
    f"{lidar_depth.min():.3f} / "
    f"{np.median(lidar_depth):.3f} / "
    f"{np.percentile(lidar_depth, 95):.3f} / "
    f"{lidar_depth.max():.3f} m",
)

print(
    "GT pixels nearer than LiDAR minimum:",
    f"{100.0 * np.mean(gt_values < lidar_depth.min()):.2f}%",
)

print(
    "GT pixels inside LiDAR depth support:",
    f"{100.0 * np.mean((gt_values >= lidar_depth.min()) & (gt_values <= lidar_depth.max())):.2f}%",
)

print(
    "GT pixels farther than LiDAR maximum:",
    f"{100.0 * np.mean(gt_values > lidar_depth.max()):.2f}%",
)

print(
    "GT p99 minus LiDAR maximum:",
    f"{np.percentile(gt_values, 99) - lidar_depth.max():+.3f} m",
)

print(
    "GT maximum minus LiDAR maximum:",
    f"{gt_values.max() - lidar_depth.max():+.3f} m",
)


# ============================================================
# V2.1 NUMERICAL ALIGNMENT AUDIT
# ============================================================

audit_mask = (
    anchors
    & valid_gt
)

if audit_mask.sum() > 0:
    sparse_gt_error = (
        sparse[audit_mask]
        - gt[audit_mask]
    )

    sparse_gt_absolute = np.abs(
        sparse_gt_error
    )

    print(
        "\n========== V2.1 ALIGNMENT AUDIT =========="
    )

    print(
        "Audited anchors:",
        int(audit_mask.sum()),
    )

    print(
        "Sparse-vs-GT bias:",
        f"{np.mean(sparse_gt_error):+.6f} m",
    )

    print(
        "Sparse-vs-GT RMSE:",
        f"{np.sqrt(np.mean(sparse_gt_error**2)):.6f} m",
    )

    print(
        "Sparse-vs-GT median absolute:",
        f"{np.median(sparse_gt_absolute):.6f} m",
    )

    print(
        "Sparse-vs-GT p90 absolute:",
        f"{np.percentile(sparse_gt_absolute, 90):.6f} m",
    )

    print(
        "Sparse-vs-GT maximum absolute:",
        f"{np.max(sparse_gt_absolute):.6f} m",
    )


# ============================================================
# GLOBAL RESULTS
# ============================================================

print(
    "\n============= LIDAR ANCHORS ============="
)

print(
    "Anchor count:",
    int(anchors.sum()),
)

print(
    "LiDAR depth range:",
    f"{lidar_depth.min():.3f} - "
    f"{lidar_depth.max():.3f} m",
)

print(
    "LiDAR depth std:",
    f"{lidar_depth.std():.6f} m",
)

print(
    "DA3 anchor range:",
    f"{da3_at_lidar.min():.6f} - "
    f"{da3_at_lidar.max():.6f}",
)

print(
    "DA3 anchor std:",
    f"{da3_at_lidar.std():.6f}",
)

print(
    "Raw design condition:",
    f"{np.linalg.cond(design_matrix):.3f}",
)

print(
    "Row 4 fit:",
    f"D = {scale4:.6f} * DA3 "
    f"+ {shift4:.6f}",
)

print(
    "Row 5 fit:",
    f"D = {scale5:.6f} * DA3 "
    f"+ {shift5:.6f}",
)


print("\n========== DA3 GLOBAL ==========")

m4 = show(
    "Row 4 DA3 one-line",
    row4,
    gt,
    valid_da3,
)

m6_uniform = show(
    "Row 6 uniform Poisson",
    row6_uniform,
    gt,
    valid_da3,
)

m6_edge = show(
    "Row 6 edge Poisson",
    row6_edge,
    gt,
    valid_da3,
)

m5 = show(
    "Row 5 affine oracle",
    row5,
    gt,
    valid_da3,
)

print(
    "\nUniform diagnostics:",
    uniform_info,
)

print(
    "Edge-aware diagnostics:",
    edge_info,
)

print(
    "Row 4-to-oracle AbsRel penalty:",
    f"{m4[1] - m5[1]:+.2f} pp",
)

print(
    "Row 4-to-oracle RMSE penalty:",
    f"{m4[0] - m5[0]:+.3f} m",
)

print(
    "Edge-Poisson change from Row 4:",
    f"{m6_edge[1] - m4[1]:+.2f} pp",
)

print(
    "Row 4 non-positive depth:",
    f"{100.0 * np.mean(row4[valid_da3] <= 0):.3f}%",
)

print(
    "Edge-Poisson non-positive depth:",
    f"{100.0 * np.mean(row6_edge[valid_da3] <= 0):.3f}%",
)


# ============================================================
# DEPTH BANDS
# ============================================================

depth_bands = [
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
    (11, np.inf),
]

print("\n========== DA3 BY GT DEPTH ==========")

for lo, hi in depth_bands:
    band_mask = (
        valid_da3
        & (gt >= lo)
        & (gt < hi)
    )

    if band_mask.sum() < 20:
        continue

    anchor_count = int(
        (
            anchors
            & valid_gt
            & (gt >= lo)
            & (gt < hi)
        ).sum()
    )

    label = format_depth_band(
        lo,
        hi,
    )

    print(
        f"\nGT {label}   "
        f"pixels={band_mask.sum()}   "
        f"anchors={anchor_count}"
    )

    show(
        "  Row 4 one-line",
        row4,
        gt,
        band_mask,
    )

    show(
        "  Uniform Poisson",
        row6_uniform,
        gt,
        band_mask,
    )

    show(
        "  Edge Poisson",
        row6_edge,
        gt,
        band_mask,
    )

    show(
        "  Row 5 oracle",
        row5,
        gt,
        band_mask,
    )


# ============================================================
# DISTANCE FROM ACTUAL PROJECTED LIDAR SUPPORT
# ============================================================

distance_from_lidar = (
    distance_transform_edt(~anchors)
    / gt.shape[0]
)

distance_bands = [
    (0.00, 0.05),
    (0.05, 0.15),
    (0.15, 0.30),
    (0.30, 2.00),
]

print(
    "\n====== DA3 BY DISTANCE FROM LIDAR ======"
)

for lo, hi in distance_bands:
    band_mask = (
        valid_da3
        & (distance_from_lidar >= lo)
        & (distance_from_lidar < hi)
    )

    if band_mask.sum() < 20:
        continue

    print(
        f"\nDistance "
        f"{100.0 * lo:.0f}-"
        f"{100.0 * hi:.0f}%   "
        f"pixels={band_mask.sum()}"
    )

    show(
        "  Row 4 one-line",
        row4,
        gt,
        band_mask,
    )

    show(
        "  Uniform Poisson",
        row6_uniform,
        gt,
        band_mask,
    )

    show(
        "  Edge Poisson",
        row6_edge,
        gt,
        band_mask,
    )

    show(
        "  Row 5 oracle",
        row5,
        gt,
        band_mask,
    )


# ============================================================
# GT DEPTH x DISTANCE-FROM-LIDAR CROSS-TABLES
# ============================================================

cross_methods = [
    ("Row 4 global affine", row4),
    ("Uniform Poisson", row6_uniform),
    ("Edge-aware Poisson", row6_edge),
    ("Row 5 affine oracle", row5),
]

print("\n")
print("=" * 135)
print(
    "GT DEPTH x DISTANCE-FROM-LIDAR CROSS-TABLE"
)
print(
    "Each cell: AbsRel% / p90% / worst% / pixel count"
)
print("=" * 135)

for method_name, prediction in cross_methods:
    print(f"\n{method_name}")

    print(
        f"{'GT depth':<18}"
        f"{'Near 0-5%':>28}"
        f"{'5-15%':>28}"
        f"{'15-30%':>28}"
        f"{'Far 30%+':>28}"
    )

    for depth_lo, depth_hi in depth_bands:
        cells = []

        for distance_lo, distance_hi in distance_bands:
            cell_mask = (
                valid_da3
                & (gt >= depth_lo)
                & (gt < depth_hi)
                & (
                    distance_from_lidar
                    >= distance_lo
                )
                & (
                    distance_from_lidar
                    < distance_hi
                )
            )

            result = cross_cell_metrics(
                prediction,
                gt,
                cell_mask,
            )

            cells.append(
                format_cross_cell(result)
            )

        print(
            f"{format_depth_band(depth_lo, depth_hi):<18}"
            f"{cells[0]:>28}"
            f"{cells[1]:>28}"
            f"{cells[2]:>28}"
            f"{cells[3]:>28}"
        )


# ============================================================
# CELL-BY-CELL EDGE-POISSON CHANGE
# ============================================================

print("\n")
print("=" * 105)
print(
    "EDGE-AWARE POISSON CHANGE RELATIVE TO GLOBAL AFFINE"
)
print(
    "Negative values mean that Poisson reduced AbsRel"
)
print("=" * 105)

print(
    f"{'GT depth':<18}"
    f"{'Near 0-5%':>19}"
    f"{'5-15%':>19}"
    f"{'15-30%':>19}"
    f"{'Far 30%+':>19}"
)

for depth_lo, depth_hi in depth_bands:
    changes = []

    for distance_lo, distance_hi in distance_bands:
        cell_mask = (
            valid_da3
            & (gt >= depth_lo)
            & (gt < depth_hi)
            & (
                distance_from_lidar
                >= distance_lo
            )
            & (
                distance_from_lidar
                < distance_hi
            )
        )

        global_result = cross_cell_metrics(
            row4,
            gt,
            cell_mask,
        )

        poisson_result = cross_cell_metrics(
            row6_edge,
            gt,
            cell_mask,
        )

        if (
            np.isnan(global_result["absrel"])
            or np.isnan(poisson_result["absrel"])
        ):
            changes.append("--")
        else:
            difference = (
                poisson_result["absrel"]
                - global_result["absrel"]
            )

            changes.append(
                f"{difference:+.2f} pp"
            )

    print(
        f"{format_depth_band(depth_lo, depth_hi):<18}"
        f"{changes[0]:>19}"
        f"{changes[1]:>19}"
        f"{changes[2]:>19}"
        f"{changes[3]:>19}"
    )