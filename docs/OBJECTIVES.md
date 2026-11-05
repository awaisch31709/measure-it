# Objectives

## Hardware

CPU: (to add) | GPU: none (training on Google Colab) | RAM: 8 GB | OS: Windows 10 | Camera: phone main camera | Image resolution: 1280 x 960

## MO-1: measurement accuracy

| Field | Entry |
|---|---|
| What is measured | Absolute error in mm between the system's width/height and a ruler measurement of the same object |
| How | measurement/run_accuracy.py compares system output with a hand-typed ground-truth CSV, per instance |
| Target | Mean absolute error 2 mm or less, worst case 4 mm or less, over 10 or more instances |
| Conditions | Phone held roughly parallel to the surface, indoor daylight, object and marker flat on the same surface, undistorted images |
| Not included | Tilted camera, object partly outside the frame, marker covered |

Why this target: a ruler reads to about 1 mm, so errors near 1 mm cannot be told apart from ground-truth error; 2 mm is useful for sorting objects by size.

Result: <paste raw output at the end>

## MO-2: segmentation quality

| Field | Entry |
|---|---|
| What is measured | Mean mask IoU on the test split |
| How | models/evaluate.py on the 8 test images, run once at the end |
| Target | 0.85 or more |
| Conditions | Same camera, same setup as the training photos |
| Not included | Images where the object is cut by the frame edge |

Result: <paste raw output at the end>

## Calibration (item 1)

<paste at the end>

## Model metrics on the test split (item 3)

<paste at the end>

## Raw vs undistorted (item 6)

<paste at the end>
