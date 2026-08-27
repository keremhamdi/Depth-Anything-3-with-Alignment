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
