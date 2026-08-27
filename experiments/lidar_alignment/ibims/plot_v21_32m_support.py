from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SPARSE_DIRECTORY = Path(
    "experiments/ibims_replication/v2_1_sensor_32m"
)

DA3_DIRECTORY = Path(
    "experiments/ibims_replication/da3_bridge_all"
)

OUTPUT_DIRECTORY = Path(
    "experiments/ibims_replication/analysis_100"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


records = []


for sparse_path in sorted(
    SPARSE_DIRECTORY.glob("*.npy")
):
    scene = sparse_path.stem

    da3_path = (
        DA3_DIRECTORY
        / f"{scene}_da3small.npy"
    )

    if not da3_path.exists():
        raise FileNotFoundError(
            f"Missing DA3 prediction: {da3_path}"
        )

    sparse = np.load(
        sparse_path
    ).astype(np.float64)

    da3 = np.load(
        da3_path
    ).astype(np.float64)

    if sparse.shape != da3.shape:
        raise RuntimeError(
            f"Shape mismatch: {scene}"
        )

    anchors = (
        np.isfinite(sparse)
        & (sparse > 0)
        & np.isfinite(da3)
    )

    lidar = sparse[anchors]
    da3_anchors = da3[anchors]

    design = np.column_stack(
        (
            da3_anchors,
            np.ones_like(da3_anchors),
        )
    )

    records.append(
        {
            "scene": scene,
            "anchor_count": int(anchors.sum()),
            "lidar_min": float(lidar.min()),
            "lidar_median": float(
                np.median(lidar)
            ),
            "lidar_max": float(lidar.max()),
            "support_span": float(
                lidar.max() - lidar.min()
            ),
            "lidar_std": float(lidar.std()),
            "da3_anchor_std": float(
                da3_anchors.std()
            ),
            "design_condition": float(
                np.linalg.cond(design)
            ),
        }
    )


csv_path = (
    OUTPUT_DIRECTORY
    / "v21_32m_support_all_scenes.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(records[0].keys()),
    )

    writer.writeheader()
    writer.writerows(records)


ordered = sorted(
    records,
    key=lambda row: row["lidar_max"],
)

indices = np.arange(
    len(ordered)
)

minimums = np.array(
    [row["lidar_min"] for row in ordered]
)

maximums = np.array(
    [row["lidar_max"] for row in ordered]
)

spans = np.array(
    [row["support_span"] for row in records]
)

counts = np.array(
    [row["anchor_count"] for row in records]
)

conditions = np.array(
    [
        row["design_condition"]
        for row in records
    ]
)


figure, axes = plt.subplots(
    2,
    2,
    figsize=(17, 11),
)


# Every scene's measured LiDAR interval.
axes[0, 0].vlines(
    indices,
    minimums,
    maximums,
    color="tab:blue",
    alpha=0.55,
    linewidth=1.5,
)

axes[0, 0].scatter(
    indices,
    minimums,
    s=10,
    label="Minimum",
)

axes[0, 0].scatter(
    indices,
    maximums,
    s=10,
    label="Maximum",
)

axes[0, 0].axhline(
    32,
    color="black",
    linestyle="--",
    linewidth=1,
    label="32 m sensor limit",
)

axes[0, 0].set_title(
    "LiDAR depth interval for every scene"
)

axes[0, 0].set_xlabel(
    "Scenes ordered by maximum measured depth"
)

axes[0, 0].set_ylabel(
    "Depth (m)"
)

axes[0, 0].legend()


# Distribution of support span.
axes[0, 1].hist(
    spans,
    bins=20,
    edgecolor="black",
    alpha=0.8,
)

axes[0, 1].axvline(
    np.median(spans),
    color="red",
    linestyle="--",
    label=(
        f"Median = "
        f"{np.median(spans):.2f} m"
    ),
)

axes[0, 1].set_title(
    "Distribution of LiDAR depth-support span"
)

axes[0, 1].set_xlabel(
    "Maximum depth − minimum depth (m)"
)

axes[0, 1].set_ylabel(
    "Number of scenes"
)

axes[0, 1].legend()


# Point count versus metric coverage.
axes[1, 0].scatter(
    counts,
    spans,
    alpha=0.75,
)

axes[1, 0].set_title(
    "Anchor count versus metric-depth coverage"
)

axes[1, 0].set_xlabel(
    "Projected LiDAR anchor count"
)

axes[1, 0].set_ylabel(
    "LiDAR support span (m)"
)


narrowest = min(
    records,
    key=lambda row: row["support_span"],
)

widest = max(
    records,
    key=lambda row: row["support_span"],
)

for row in [narrowest, widest]:
    axes[1, 0].annotate(
        row["scene"],
        (
            row["anchor_count"],
            row["support_span"],
        ),
        xytext=(6, 6),
        textcoords="offset points",
    )


# Conditioning versus LiDAR depth variation.
finite_condition = (
    np.isfinite(conditions)
    & (conditions > 0)
    & (spans > 0)
)

axes[1, 1].scatter(
    spans[finite_condition],
    conditions[finite_condition],
    alpha=0.75,
)

axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")

axes[1, 1].set_title(
    "Affine condition versus LiDAR support"
)

axes[1, 1].set_xlabel(
    "LiDAR support span (m, log scale)"
)

axes[1, 1].set_ylabel(
    "Raw affine design condition (log scale)"
)


figure.suptitle(
    "iBims V2.1 one-line LiDAR support across 100 scenes",
    fontsize=16,
)

figure.tight_layout()


figure_path = (
    OUTPUT_DIRECTORY
    / "v21_32m_support_summary.png"
)

figure.savefig(
    figure_path,
    dpi=180,
)

plt.close(figure)


print("Scenes:", len(records))
print("Saved CSV:", csv_path)
print("Saved graph:", figure_path)
print("Narrowest:", narrowest)
print("Widest:", widest)