# Objectives

## Hardware

CPU:Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz
 | GPU: none locally; training on Google Colab, Tesla T4 | RAM: 8 GB | OS: Windows 10 | Camera: Redmi Note 14 main camera, 1x | Image resolution: 1280 x 960

## MO-1: measurement accuracy

| Field | Entry |
|---|---|
| What is measured | Absolute error in mm between the system's width/height and a ruler measurement of the same object |
| How | measurement/run_accuracy.py compares system output with measurement/ground_truth.csv, per instance |
| Target | Mean absolute error 2 mm or less, worst case 4 mm or less, over 10 or more instances |
| Conditions | Phone held roughly parallel to the surface, indoor daylight, object and marker flat on the same surface, undistorted images |
| Not included | Tilted camera, object partly outside the frame, marker covered |

Why this target: a ruler reads to about 1 mm, so errors near 1 mm cannot be told apart from ground-truth error; 2 mm is useful for sorting objects by size.

**Result: target missed.** 22 held-out photos measured, refusals switched off. Applying the system's own tilt refusal (marker skew ≤ 0.02) keeps 9 photos:

```
threshold  accepted  refused   MAE mm   worst mm
     0.01         7       15    3.570      7.052
     0.02         9       13    3.737      7.052
     0.03        10       12    3.930     10.764
     0.05        16        6    7.271     45.673
      0.1        21        1    9.103     45.673
```

All 22, no refusal (measurement/results/accuracy_summary.json):

```
raw:          n 22, MAE 9.734 mm (width 13.623, height 5.844), median 5.506, worst 44.697, mean 7.883 %
undistorted:  n 22, MAE 9.828 mm (width 13.475, height 6.181), median 6.234, worst 45.673, mean 8.048 %
```

Why it is missed:

1. Tilt. The large errors (photos 013, 028, 046, 079, 036: 20 to 45 mm) all have marker skew 0.046 or more. Measuring my own hand-made labels on those photos gives the same wrong sizes, so the error comes from the geometry, not the model. The refusal at 0.02 removes them.
2. Thickness. Even on straight photos the width comes out 3 to 6 mm too long. The calculator is about 15 mm thick, so its top outline is about 15 / 400 ≈ 4 % closer to a camera 40 cm away than the marker is, and it looks about 4 % bigger (161 mm × 4 % ≈ 6 mm). The single-marker scale assumes the object lies in the marker's plane.
3. Ground truth is one ruler reading (about 1 mm resolution), not a calliper.

## MO-2: segmentation quality

| Field | Entry |
|---|---|
| What is measured | Mean mask IoU on the test split |
| How | models/evaluate.py on the 8 test images, run once at the end |
| Target | 0.85 or more |
| Conditions | Same camera, same setup as the training photos |
| Not included | Images where the object is cut by the frame edge |

**Result: met.** Mean mask IoU 0.9477 on the 8 test images.

## Calibration (item 1)

```
images_used: 28    image size: 1280 x 960
rms_reprojection_error_px: 0.6769
per_image_error_px_mean:  0.6439
camera_matrix: [[920.845, 0, 635.701], [0, 922.968, 474.672], [0, 0, 1]]
dist_coeffs k1 k2 p1 p2 k3: [0.201066, -0.679792, -0.00286, -0.001312, 0.706842]
```

## Model metrics (item 3)

Mask R-CNN ResNet50-FPN v2, COCO-pretrained, fine-tuned 30 epochs, seed 42, Colab T4. 56 train / 16 val / 8 test.

```
test (run once, --final): map_50 1.0, map_50_95 0.946, precision 1.0, recall 1.0, mean_mask_iou 0.9477, images 8
val:                      map_50 1.0, map_50_95 1.0,   precision 0.9412, recall 1.0, mean_mask_iou 0.9723, images 16
```

These scores are high because the task is easy: one object class, one object per photo, plain backgrounds. Val precision 0.94 is one extra false detection in 16 images. With 8 test images one mistake moves a metric by about 12 points, so the test numbers are not precise.

## Raw vs undistorted (item 6)

Same 22 photos measured both ways (columns raw_* and und_* in measurement/results/accuracy.csv).

| | MAE mm | median mm | worst mm |
|---|---|---|---|
| raw | 9.734 | 5.506 | 44.697 |
| undistorted | 9.828 | 6.234 | 45.673 |

Undistortion did not reduce the error here; the difference (0.1 mm) is far smaller than the tilt and thickness errors. The calculator sits near the image centre where lens distortion is smallest, and the marker scale is taken from nearby pixels, so most distortion cancels. Undistortion would matter more for objects near the image edges.
