# Sparse LiDAR alignment experiments

These scripts evaluate Depth Anything 3 predictions with sparse, single-plane LiDAR
measurements. Datasets, checkpoints, generated predictions, and output arrays are not
stored in this repository.

## Data root

Every script accepts `--data-root`. The directory must contain the existing
Any2Full experiment layout:

```text
<data-root>/
├── datasets/ibims1/
└── experiments/ibims_replication/
```

For the current workstation, run a script from the repository root with:

```bash
PYTHONPATH=src python experiments/lidar_alignment/ibims/da3_calibration_objectives.py --data-root "$HOME/Projects/Any2Full"
```

You can avoid repeating the argument by setting:

```bash
export DA3_LIDAR_DATA_ROOT="$HOME/Projects/Any2Full"
```

Scripts that generate files write by default to
`experiments/lidar_alignment/outputs/`, which is excluded from Git. Use
`--output-dir PATH` to select another output directory.

## Matched Any2Full versus DA3 comparison

`compare_any2full_da3_100.py` evaluates the original Any2Full V2 predictions and
five DA3 calibration strategies using the same V2 sparse-LiDAR input:

1. median scale-only alignment;
2. ordinary affine least squares;
3. positive affine log-depth least squares;
4. Huber affine alignment;
5. ordinary affine alignment followed by edge-aware Poisson correction.

Dense ground truth is used only for evaluation. The evaluator reports all valid
pixels, non-anchor pixels, the LiDAR-supported depth interval, below-support and
above-support depths, outside-support pixels, the 0–2 m near field, and anchor
pixels.

Check that all 100 scenes are matched:

```bash
PYTHONPATH=src python experiments/lidar_alignment/ibims/compare_any2full_da3_100.py --preflight-only
```

Run a complete one-scene smoke test:

```bash
PYTHONPATH=src python experiments/lidar_alignment/ibims/compare_any2full_da3_100.py --scene corridor_01
```

Run the complete 100-scene comparison:

```bash
PYTHONPATH=src python experiments/lidar_alignment/ibims/compare_any2full_da3_100.py
```

Poisson alignment is enabled by default. Use `--skip-poisson` for a faster
calibration-only run. Results are written to
`experiments/lidar_alignment/outputs/comparison_any2full_da3_v2_100/`.
