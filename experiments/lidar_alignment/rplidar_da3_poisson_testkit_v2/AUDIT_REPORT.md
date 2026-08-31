# RPLidar A1 + Camera Module 3 capture audit

Dataset: `dataset_20260831_084934`

## Verdict

The RPLidar measurements themselves are usable for a provisional real-sensor
DA3 alignment experiment. The recording is not yet sufficient to certify
full-image deployment accuracy. Before a definitive deployment claim, refine
the camera-LiDAR calibration and capture independent dense ground truth for a
representative subset.

## What was verified

- 15 complete RGB/CSV/mask/overlay/metadata groups at 1280 x 720.
- 973 recorded beam rows; 783 valid ranges; 679 points projected into the RGB
  image.
- 100% of projected CSV centres occur in the corresponding binary mask.
- 32-69 projected anchors per image, mean 45.3.
- Radial range: 0.388-2.915 m. Camera-axis Z after conversion:
  0.332-2.777 m.
- A clean planar-board capture (`00009`) has a point-to-fitted-line residual of
  0.92 mm median and 2.31 mm at the 95th percentile. This supports strong
  internal consistency of the LiDAR returns in that capture.

## Corrections required before alignment

### Use camera-axis Z, not CSV radial range

For this session's zero yaw/pitch/roll/t_z calibration:

`z_camera = (range_mm / 1000) * cos(angle_deg)`

Across the projected points, using raw range instead of Z would overestimate
metric image depth by 6.22% on average and 4.12% at the median. 25.9% of the
points exceed 10% error and 13.4% exceed 15% error. The binary `lidar_mask` PNG
contains no metric values.

### Exclude beams outside the exact camera exposure

Exposure is 152.391 ms while the measured LiDAR revolution period is about
129.06 ms. Each image therefore mixes returns from 2-3 revolutions. The capture
code also admits a +/-4 ms timing margin: 53/679 projected points (7.8%) fall
outside the exact exposure interval. The preparation script removes those
points by default.

Long exposure is acceptable for a static calibration scene but risky for
people, arms, chairs, boards, or a moving sensor. Deployment recordings should
reduce exposure and associate measurements with a precisely defined image
time/rolling-shutter model.

### Refine extrinsic/intrinsic calibration

The photographed pink scan trace and projected green CSV dots provide an
independent visual check. They are broadly associated, but not pixel-accurate
in all captures. An automated high-confidence trace check produced a median
vertical residual near -7 px overall; only about 28% were within 5 px and 70%
within 10 px. On the clean cabinet frame `00002`, the median residual is about
-13 px and changes across the image, consistent with residual vertical
translation/roll/lens-distortion error.

This image-derived residual is a diagnostic, not a replacement for a calibrated
target. Even a 10 px mismatch can attach a foreground LiDAR distance to a
background image pixel at object boundaries, after which Poisson can propagate
the wrong correction.

## What can be measured with this dataset

The included protocol uses four spatially blocked folds. In each fold, one
quarter of angular image sectors is hidden from median alignment and Poisson;
the predictions are then evaluated on those unseen real LiDAR points. Across
four folds, every usable point is evaluated exactly once.

Report these numbers as **blocked held-out real-LiDAR scan-line metrics**.
They are useful for comparing DA3+median with DA3+median+Poisson.

Do not report them as full-image AbsRel/RMSE. The archive contains no
independent dense metric ground truth outside the RPLidar scan plane.

## Dataset coverage limitation

These 15 images show one nearby room and contain ranges below 3 m. They do not
cover long corridors, large rooms, low-light variation, glass-heavy scenes,
sensor motion, or the complete intended deployment range. More diverse
recordings and a dense-GT subset are required for a deployment conclusion.
