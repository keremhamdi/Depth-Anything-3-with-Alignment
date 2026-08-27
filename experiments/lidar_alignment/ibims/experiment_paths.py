"""Shared command-line path handling for the LiDAR alignment experiments."""

import argparse
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "experiments" / "lidar_alignment" / "outputs"


def parse_experiment_paths(description, output_subdirectory=None):
    """Return a validated external data root and an optional local output directory."""
    environment_root = os.environ.get("DA3_LIDAR_DATA_ROOT")

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(environment_root) if environment_root else None,
        required=environment_root is None,
        help=(
            "Root containing experiments/ibims_replication and datasets/ibims1. "
            "Alternatively set DA3_LIDAR_DATA_ROOT."
        ),
    )

    if output_subdirectory is not None:
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help=(
                "Directory for generated results. The default is "
                "experiments/lidar_alignment/outputs/<experiment>."
            ),
        )

    arguments = parser.parse_args()
    data_root = arguments.data_root.expanduser().resolve()

    if not data_root.is_dir():
        parser.error(f"Data root does not exist or is not a directory: {data_root}")

    output_directory = None
    if output_subdirectory is not None:
        if arguments.output_dir is None:
            output_directory = DEFAULT_OUTPUT_ROOT / output_subdirectory
        else:
            output_directory = arguments.output_dir.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

    return data_root, output_directory
