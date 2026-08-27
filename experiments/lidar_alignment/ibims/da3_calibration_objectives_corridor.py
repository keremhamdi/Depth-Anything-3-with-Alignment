from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import least_squares, minimize


GT_PATH = Path(
    "experiments/ibims_replication/oracle_inputs/"
    "corridor_01_gt.npy"
)

MASK_PATH = Path(
    "experiments/ibims_replication/oracle_inputs/"
    "corridor_01_mask_invalid.npy"
)
TRANSP_PATH = Path(
    "experiments/ibims_replication/oracle_inputs/"
    "corridor_01_mask_transp.npy"
)
DA3_PATH = Path(
    "experiments/ibims_replication/da3_bridge/"
    "corridor_01_da3small.npy"
)

SPARSE_PATH = Path(
    "experiments/ibims_replication/v2_1_sensor/"
    "corridor_01.npy"
)

RGB_PATH = Path(
    "datasets/ibims1/ibims1_core_raw/rgb/"
    "corridor_01.png"
)

OUTPUT_DIRECTORY = Path(
    "experiments/ibims_replication/"
    "calibration_objectives"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def fit_ls(x, y):
    matrix = np.column_stack(
        (x, np.ones_like(x))
    )

    scale, shift = np.linalg.lstsq(
        matrix,
        y,
        rcond=None,
    )[0]

    return float(scale), float(shift)


def fit_positive_log_affine(
    x,
    y,
    full_domain_min,
    full_domain_max,
):
    """
    Fit D = scale * R + shift by minimizing log-depth error.

    The affine prediction is constrained to remain positive over
    the complete DA3-value range in the image, including values
    outside the LiDAR-anchor range.
    """
    domain_size = (
        full_domain_max
        - full_domain_min
    )

    if domain_size <= 0:
        raise ValueError(
            "DA3 prediction has no usable variation."
        )

    normalized_x = (
        (x - full_domain_min)
        / domain_size
    )

    initial_scale, initial_shift = fit_ls(
        x,
        y,
    )

    initial_left = max(
        initial_scale * full_domain_min
        + initial_shift,
        1e-4,
    )

    initial_right = max(
        initial_scale * full_domain_max
        + initial_shift,
        1e-4,
    )

    def objective(log_endpoints):
        left, right = np.exp(
            log_endpoints
        )

        prediction = (
            left * (1.0 - normalized_x)
            + right * normalized_x
        )

        return np.mean(
            (
                np.log(prediction)
                - np.log(y)
            ) ** 2
        )

    result = minimize(
        objective,
        np.log(
            [initial_left, initial_right]
        ),
        method="Nelder-Mead",
        options={
            "maxiter": 3000,
            "xatol": 1e-11,
            "fatol": 1e-13,
        },
    )

    left, right = np.exp(
        result.x
    )

    scale = (
        (right - left)
        / domain_size
    )

    shift = (
        left
        - scale * full_domain_min
    )

    return (
        float(scale),
        float(shift),
        float(np.sqrt(result.fun)),
    )


def fit_huber(
    x,
    y,
    initial_scale,
    initial_shift,
    delta=0.10,
):
    """
    Robust affine fit with Huber transition delta in metres.
    """
    def residual(parameters):
        scale, shift = parameters

        return (
            scale * x
            + shift
            - y
        )

    result = least_squares(
        residual,
        x0=np.array(
            [initial_scale, initial_shift],
            dtype=np.float64,
        ),
        loss="huber",
        f_scale=delta,
        max_nfev=5000,
    )

    return (
        float(result.x[0]),
        float(result.x[1]),
    )


def calculate_metrics(
    prediction,
    ground_truth,
    evaluation_mask,
):
    predicted_values = prediction[
        evaluation_mask
    ]

    ground_truth_values = ground_truth[
        evaluation_mask
    ]

    error = (
        predicted_values
        - ground_truth_values
    )

    relative_error = (
        np.abs(error)
        / ground_truth_values
    )

    return {
        "n": int(evaluation_mask.sum()),
        "rmse": float(
            np.sqrt(np.mean(error**2))
        ),
        "mae": float(
            np.mean(np.abs(error))
        ),
        "absrel": float(
            100.0 * np.mean(relative_error)
        ),
        "p90": float(
            100.0
            * np.percentile(relative_error, 90)
        ),
        "bias": float(
            np.mean(error)
        ),
    }


def print_metrics(
    name,
    prediction,
    ground_truth,
    evaluation_mask,
):
    result = calculate_metrics(
        prediction,
        ground_truth,
        evaluation_mask,
    )

    print(
        f"{name:25s} "
        f"RMSE={result['rmse']:.3f} m  "
        f"MAE={result['mae']:.3f} m  "
        f"AbsRel={result['absrel']:.2f}%  "
        f"p90={result['p90']:.2f}%  "
        f"Bias={result['bias']:+.3f} m"
    )

    return result


# ============================================================
# LOAD DATA
# ============================================================

gt = np.load(
    GT_PATH
).astype(np.float64)

validity_mask = np.load(
    MASK_PATH
)
transparency_mask = np.load(
    TRANSP_PATH
)
da3 = np.load(
    DA3_PATH
).astype(np.float64)

sparse = np.load(
    SPARSE_PATH
).astype(np.float64)

rgb = np.array(
    Image.open(
        RGB_PATH
    ).convert("RGB")
)

if not (
    gt.shape
    == validity_mask.shape
    == da3.shape
    == sparse.shape
    == transparency_mask.shape
):
    raise RuntimeError(
        "GT, masks, DA3 and sparse map shapes do not match."
    )


valid = (
    np.isfinite(gt)
    & (gt > 0)
    & (validity_mask > 0)
    & np.isfinite(da3)
    & (transparency_mask > 0)
)

anchors = (
    np.isfinite(sparse)
    & (sparse > 0)
    & np.isfinite(da3)
)

if anchors.sum() < 2:
    raise RuntimeError(
        "Fewer than two valid LiDAR anchors."
    )


anchor_da3 = da3[anchors]
anchor_depth = sparse[anchors]

full_da3_min = float(
    da3[valid].min()
)

full_da3_max = float(
    da3[valid].max()
)


# ============================================================
# FIT THE THREE CALIBRATION OBJECTIVES
# ============================================================

ls_scale, ls_shift = fit_ls(
    anchor_da3,
    anchor_depth,
)

log_scale, log_shift, log_rmse = (
    fit_positive_log_affine(
        anchor_da3,
        anchor_depth,
        full_da3_min,
        full_da3_max,
    )
)

huber_scale, huber_shift = fit_huber(
    anchor_da3,
    anchor_depth,
    ls_scale,
    ls_shift,
    delta=0.10,
)


prediction_ls = (
    ls_scale * da3
    + ls_shift
)

prediction_log = (
    log_scale * da3
    + log_shift
)

prediction_huber = (
    huber_scale * da3
    + huber_shift
)


# Dense-GT LS oracle for diagnosis only.
oracle_scale, oracle_shift = fit_ls(
    da3[valid],
    gt[valid],
)

prediction_oracle = (
    oracle_scale * da3
    + oracle_shift
)


# ============================================================
# FIT PARAMETERS
# ============================================================

print(
    f"\n========== SAME {int(anchors.sum())} LIDAR ANCHORS =========="
)

print(
    "Anchor count:",
    int(anchors.sum()),
)

print(
    "Anchor depth range:",
    f"{anchor_depth.min():.3f} - "
    f"{anchor_depth.max():.3f} m",
)

print(
    "\nOrdinary LS:"
)

print(
    f"D = {ls_scale:.8f} * DA3 "
    f"+ {ls_shift:.8f}"
)

print(
    "\nPositive affine log-depth LS:"
)

print(
    f"D = {log_scale:.8f} * DA3 "
    f"+ {log_shift:.8f}"
)

print(
    "Anchor log-RMSE:",
    f"{log_rmse:.8f}",
)

print(
    "\nHuber, delta=0.10 m:"
)

print(
    f"D = {huber_scale:.8f} * DA3 "
    f"+ {huber_shift:.8f}"
)

print(
    "\nDense-GT LS oracle:"
)

print(
    f"D = {oracle_scale:.8f} * DA3 "
    f"+ {oracle_shift:.8f}"
)


# ============================================================
# GLOBAL RESULTS
# ============================================================

print(
    "\n========== GLOBAL RESULTS =========="
)

global_ls = print_metrics(
    "Ordinary LS",
    prediction_ls,
    gt,
    valid,
)

global_log = print_metrics(
    "Log-depth LS",
    prediction_log,
    gt,
    valid,
)

global_huber = print_metrics(
    "Huber",
    prediction_huber,
    gt,
    valid,
)

global_oracle = print_metrics(
    "Dense-GT LS oracle",
    prediction_oracle,
    gt,
    valid,
)


print(
    "\nAbsRel change relative to ordinary LS:"
)

print(
    "Log-depth LS:",
    f"{global_log['absrel'] - global_ls['absrel']:+.2f} pp",
)

print(
    "Huber:",
    f"{global_huber['absrel'] - global_ls['absrel']:+.2f} pp",
)


# ============================================================
# ANCHOR FIT QUALITY
# ============================================================

print(
    "\n========== ERROR AT LIDAR ANCHORS =========="
)

print_metrics(
    "Ordinary LS",
    prediction_ls,
    sparse,
    anchors,
)

print_metrics(
    "Log-depth LS",
    prediction_log,
    sparse,
    anchors,
)

print_metrics(
    "Huber",
    prediction_huber,
    sparse,
    anchors,
)


# ============================================================
# GT DEPTH BANDS
# ============================================================

depth_bands = [
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
    (11, 15),
    (15, 20),
    (20, 30),
    (30, 60),
]

print(
    "\n========== RESULTS BY GT DEPTH =========="
)

for lower, upper in depth_bands:
    band_mask = (
        valid
        & (gt >= lower)
        & (gt < upper)
    )

    if band_mask.sum() < 20:
        continue

    anchor_count = int(
        (
            anchors
            & valid
            & (gt >= lower)
            & (gt < upper)
        ).sum()
    )

    print(
        f"\nGT {lower}-{upper} m  "
        f"pixels={band_mask.sum()}  "
        f"anchors={anchor_count}"
    )

    print_metrics(
        "  Ordinary LS",
        prediction_ls,
        gt,
        band_mask,
    )

    print_metrics(
        "  Log-depth LS",
        prediction_log,
        gt,
        band_mask,
    )

    print_metrics(
        "  Huber",
        prediction_huber,
        gt,
        band_mask,
    )

    print_metrics(
        "  Dense-GT oracle",
        prediction_oracle,
        gt,
        band_mask,
    )


# ============================================================
# SAVE DEPTH MAPS
# ============================================================

np.save(
    OUTPUT_DIRECTORY
    / "corridor_01_da3_ls.npy",
    prediction_ls.astype(np.float32),
)

np.save(
    OUTPUT_DIRECTORY
    / "corridor_01_da3_log.npy",
    prediction_log.astype(np.float32),
)

np.save(
    OUTPUT_DIRECTORY
    / "corridor_01_da3_huber.npy",
    prediction_huber.astype(np.float32),
)


# ============================================================
# VISUALIZATION
# ============================================================

gt_display = np.where(
    valid,
    gt,
    np.nan,
)

ls_display = np.where(
    valid,
    prediction_ls,
    np.nan,
)

log_display = np.where(
    valid,
    prediction_log,
    np.nan,
)

huber_display = np.where(
    valid,
    prediction_huber,
    np.nan,
)

ls_error = np.where(
    valid,
    np.abs(prediction_ls - gt),
    np.nan,
)

log_error = np.where(
    valid,
    np.abs(prediction_log - gt),
    np.nan,
)

huber_error = np.where(
    valid,
    np.abs(prediction_huber - gt),
    np.nan,
)


depth_min = float(
    np.percentile(gt[valid], 1)
)

depth_max = float(
    np.percentile(gt[valid], 99)
)

combined_errors = np.concatenate(
    (
        ls_error[valid],
        log_error[valid],
        huber_error[valid],
    )
)

error_max = float(
    np.percentile(combined_errors, 99)
)


figure, axes = plt.subplots(
    2,
    4,
    figsize=(20, 10),
)


# RGB plus LiDAR overlay.
axes[0, 0].imshow(rgb)

anchor_y, anchor_x = np.nonzero(
    anchors
)

overlay_scatter = axes[0, 0].scatter(
    anchor_x,
    anchor_y,
    c=sparse[anchors],
    s=12,
    cmap="turbo",
)

axes[0, 0].set_title(
    f"RGB + one-line LiDAR\n"
    f"{int(anchors.sum())} anchors"
)

axes[0, 0].axis("off")

figure.colorbar(
    overlay_scatter,
    ax=axes[0, 0],
    fraction=0.046,
)


depth_images = [
    (axes[0, 1], gt_display, "GT depth"),
    (axes[0, 2], ls_display, "Ordinary LS"),
    (axes[0, 3], log_display, "Log-depth LS"),
    (axes[1, 0], huber_display, "Huber"),
]

for axis, image, title in depth_images:
    displayed = axis.imshow(
        image,
        cmap="turbo",
        vmin=depth_min,
        vmax=depth_max,
    )

    axis.set_title(title)
    axis.axis("off")

    figure.colorbar(
        displayed,
        ax=axis,
        fraction=0.046,
    )


error_images = [
    (
        axes[1, 1],
        ls_error,
        "Absolute error: LS",
    ),
    (
        axes[1, 2],
        log_error,
        "Absolute error: log LS",
    ),
    (
        axes[1, 3],
        huber_error,
        "Absolute error: Huber",
    ),
]

for axis, image, title in error_images:
    displayed = axis.imshow(
        image,
        cmap="magma",
        vmin=0,
        vmax=error_max,
    )

    axis.set_title(title)
    axis.axis("off")

    figure.colorbar(
        displayed,
        ax=axis,
        fraction=0.046,
    )

figure.suptitle(
    f"Corridor 01: DA3 calibration using the same "
    f"{int(anchors.sum())} LiDAR anchors"
)

figure.tight_layout()

visualization_path = (
    OUTPUT_DIRECTORY
    / "corridor_01_calibration_comparison.png"
)

figure.savefig(
    visualization_path,
    dpi=170,
)

plt.close(figure)

print(
    "\nSaved visualization:",
    visualization_path,
)