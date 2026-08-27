import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def poisson_align(
    base_depth,
    sparse_depth,
    anchor_mask,
    edge_aware=True,
    sigma_log=0.05,
    minimum_edge_weight=1e-3,
    anchor_weight=1000.0,
    screen_weight=1e-6,
    rtol=1e-6,
    maxiter=1000,
):
    """
    Preserve the gradients of base_depth while propagating sparse
    metric-depth residuals over the image.

    Parameters
    ----------
    base_depth : HxW array
        Globally calibrated DA3 prediction.
    sparse_depth : HxW array
        Metric LiDAR depth, zero or invalid outside anchors.
    anchor_mask : HxW bool array
        Locations containing projected LiDAR measurements.
    edge_aware : bool
        If True, reduce propagation across DA3 depth discontinuities.

    Returns
    -------
    aligned_depth : HxW array
    diagnostics : dict
    """
    base = np.asarray(base_depth, dtype=np.float64)
    sparse = np.asarray(sparse_depth, dtype=np.float64)
    anchors = np.asarray(anchor_mask, dtype=bool)

    if base.shape != sparse.shape or base.shape != anchors.shape:
        raise ValueError("base_depth, sparse_depth and anchor_mask must match.")

    if not np.isfinite(base).all():
        raise ValueError("base_depth contains non-finite values.")

    anchors &= np.isfinite(sparse) & (sparse > 0)

    if anchors.sum() == 0:
        raise ValueError("No valid LiDAR anchors were provided.")

    height, width = base.shape

    if edge_aware:
        log_base = np.log(np.maximum(base, 1e-6))

        horizontal_difference = np.abs(
            log_base[:, 1:] - log_base[:, :-1]
        )
        vertical_difference = np.abs(
            log_base[1:, :] - log_base[:-1, :]
        )

        weight_h = np.exp(
            -(horizontal_difference / sigma_log) ** 2
        )
        weight_v = np.exp(
            -(vertical_difference / sigma_log) ** 2
        )

        weight_h = np.clip(
            weight_h, minimum_edge_weight, 1.0
        )
        weight_v = np.clip(
            weight_v, minimum_edge_weight, 1.0
        )
    else:
        weight_h = np.ones(
            (height, width - 1), dtype=np.float64
        )
        weight_v = np.ones(
            (height - 1, width), dtype=np.float64
        )

    def weighted_laplacian(image):
        result = np.zeros_like(image)

        difference_h = image[:, 1:] - image[:, :-1]
        flux_h = weight_h * difference_h
        result[:, :-1] -= flux_h
        result[:, 1:] += flux_h

        difference_v = image[1:, :] - image[:-1, :]
        flux_v = weight_v * difference_v
        result[:-1, :] -= flux_v
        result[1:, :] += flux_v

        return result

    def matrix_vector(vector):
        image = vector.reshape(height, width)

        result = weighted_laplacian(image)
        result += anchor_weight * anchors * image
        result += screen_weight * image

        return result.ravel()

    right_hand_side = np.zeros_like(base)
    right_hand_side[anchors] = (
        anchor_weight
        * (sparse[anchors] - base[anchors])
    )

    diagonal = np.full_like(base, screen_weight)
    diagonal += anchor_weight * anchors

    diagonal[:, :-1] += weight_h
    diagonal[:, 1:] += weight_h
    diagonal[:-1, :] += weight_v
    diagonal[1:, :] += weight_v

    system = LinearOperator(
        (base.size, base.size),
        matvec=matrix_vector,
        dtype=np.float64,
    )

    preconditioner = LinearOperator(
        (base.size, base.size),
        matvec=lambda vector: vector / diagonal.ravel(),
        dtype=np.float64,
    )

    initial = np.zeros(base.size, dtype=np.float64)

    try:
        correction, info = cg(
            system,
            right_hand_side.ravel(),
            x0=initial,
            M=preconditioner,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
        )
    except TypeError:
        # Compatibility with older SciPy versions.
        correction, info = cg(
            system,
            right_hand_side.ravel(),
            x0=initial,
            M=preconditioner,
            tol=rtol,
            maxiter=maxiter,
        )

    if info != 0:
        warnings.warn(
            f"Poisson CG returned info={info}; "
            "the solution may not be fully converged."
        )

    correction = correction.reshape(height, width)
    aligned = base + correction

    before = base[anchors] - sparse[anchors]
    after = aligned[anchors] - sparse[anchors]

    diagnostics = {
        "cg_info": int(info),
        "anchor_count": int(anchors.sum()),
        "anchor_rmse_before": float(
            np.sqrt(np.mean(before**2))
        ),
        "anchor_rmse_after": float(
            np.sqrt(np.mean(after**2))
        ),
        "correction_min": float(correction.min()),
        "correction_max": float(correction.max()),
        "correction_mean": float(correction.mean()),
        "nonpositive_percent": float(
            100.0 * np.mean(aligned <= 0)
        ),
    }

    return aligned, diagnostics