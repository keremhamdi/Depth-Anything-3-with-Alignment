# Real RPLidar DA3 + median + Poisson test kit

This kit prepares `dataset_20260831_084934` and evaluates DA3+median against
DA3+median+existing-Poisson without leaking evaluation LiDAR anchors into the
alignment.

## 1. Extract the recording

```bash
7z x dataset_20260831_084934.7z
```

The dataset root must contain `cam_rgb`, `lidar_csv`, `lidar_mask`, `debug`, and
`metadata`.

## 2. Prepare metric sparse depth and four held-out folds

```bash
python prepare_dataset.py \
  --dataset-root /path/to/dataset_20260831_084934 \
  --output-root /path/to/rplidar_20260831_prepared
```

Use `depth_*_points` for DA3 alignment. Use the 3x3 `depth_*_splat` maps only
for models such as Any2Full whose resizing can erase isolated single pixels.

The values are camera-axis Z metres. Beams outside the exact exposure are
excluded by default.

## 3. Produce cached DA3 relative predictions

Run the same DA3-small RGB inference/bridge used in the existing RPLidar
experiment, with:

- input RGB: `/path/to/rplidar_20260831_prepared/rgb`
- output NPY: `/path/to/rplidar_20260831_da3_relative`

There must be one positive relative-depth `.npy` map per RGB stem. Do not align
these maps beforehand; the evaluator performs alignment separately inside each
held-out fold.

## 4. Run the real-sensor comparison

```bash
python evaluate_da3_median_poisson.py \
  --da3-root "$DA3_ROOT" \
  --prepared-root /path/to/rplidar_20260831_prepared \
  --da3-dir /path/to/rplidar_20260831_da3_relative \
  --output-dir /path/to/rplidar_20260831_da3_poisson_eval
```

Open `summary.csv` first. Then inspect `panels/` and
`per_scene_metrics.csv`.

## Interpretation

`summary.csv` reports four-fold blocked held-out real-LiDAR scan-line metrics.
It does not report dense full-image accuracy because the recording has no dense
independent ground truth.

The `full_predictions_m` maps use all real LiDAR anchors and are intended for
deployment-oriented qualitative inspection after the held-out comparison.
