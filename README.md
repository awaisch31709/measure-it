# Measure It

Photograph a Casio fx-991ES PLUS calculator next to a printed ArUco marker and get its width and height in millimetres.

Pipeline: calibrated phone camera → undistort → my fine-tuned **Mask R-CNN** (torchvision) segments the calculator → OpenCV finds the ArUco marker (50 mm) for the pixel-to-mm scale → minimum-area rectangle around the mask gives width (longer side) and height.

All work is on branch `dev-test01`. Plan, objectives and done-checklist are in [`docs/`](docs).

## Results

| What | Result | File |
|---|---|---|
| Calibration | RMS reprojection error 0.68 px, 28 board images | `calibration/camera_params.json` |
| Dataset | 80 own photos, split 56 / 16 / 8 fixed before any code | `dataset/splits/` |
| Segmentation (test, 8 images) | mAP@0.5 1.0, mAP@0.5:0.95 0.946, precision 1.0, recall 1.0, mask IoU 0.948 | `models/runs/main/test_metrics.json` |
| Accuracy vs ruler (22 held-out photos) | MAE 3.7 mm on the 9 photos the system accepts; 9.8 mm on all 22 with refusals off | `measurement/results/` |
| Raw vs undistorted | MAE 9.73 mm raw vs 9.83 mm undistorted (no gain for this setup) | `measurement/results/accuracy_summary.json` |

The 2 mm accuracy target (MO-1) is **missed**. The reasons (camera tilt, and the calculator's 15 mm thickness putting its outline closer to the camera than the marker) are in [`docs/OBJECTIVES.md`](docs/OBJECTIVES.md).

## Weights

Not in git (175 MB). Download from [Release v1.0](https://github.com/awaisch31709/measure-it/releases/tag/v1.0) and place at `models/runs/main/best.pt`:

https://github.com/awaisch31709/measure-it/releases/download/v1.0/best.pt

SHA-256: `8be38c090a265630193444b0f1689f5b899c339b3efce64188b47722ad066636`

Check on Windows: `certutil -hashfile models\runs\main\best.pt SHA256`

## Setup

Python 3.13. Training versions are pinned in `requirements.txt` (Colab, CUDA build). For CPU inference:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torchmetrics pycocotools PyYAML matplotlib opencv-contrib-python
```

## One command: image to millimetres

```
python -m inference.measure photo.jpg --ckpt models/runs/main/best.pt --marker-mm 50 --out overlay.png
```

Prints one JSON line and writes `overlay.png`. A full-resolution or portrait 4:3 photo from the same phone is scaled to the calibrated 1280 x 960 first.

| Exit code | Status | Meaning |
|---|---|---|
| 0 | `ok` | width_mm and height_mm reported |
| 2 | `object_not_found` | no calculator mask above the score floor |
| 3 | `reference_not_found` | ArUco marker id 0 not detected |
| 4 | `refused_*` | low confidence, camera tilted (marker skew > 0.02) or object cut by the frame edge |

Tested outputs:

```
no_object.jpg  -> {"status": "object_not_found", ...}     exit 2
no_marker.jpg  -> {"status": "reference_not_found", ...}  exit 3
photo_060.jpg  -> {"status": "refused_tilted_plane", "skew": 0.054, "detail": "marker skew 0.054 > 0.02"}  exit 4
```

## Reproduce

```
# 1. calibration
python -m calibration.calibrate --images calibration/images --inner-cols 9 --inner-rows 7 --square-mm 20
# 2. dataset: undistort raw photos and name them per dataset/splits/all.txt
python -m dataset.prepare --raw dataset/raw_photos
# 3. train and evaluate (Colab T4)
python -m models.train --config models/config.yaml --run models/runs/main
python -m models.evaluate --ckpt models/runs/main/best.pt --split val
python -m models.evaluate --ckpt models/runs/main/best.pt --split test --final
# 5 + 6. accuracy vs ruler, raw and undistorted
python -m measurement.run_accuracy --images dataset/raw_photos --ckpt models/runs/main/best.pt --marker-mm 50
python -m measurement.threshold_sweep --column skew --values 0.01 0.02 0.03 0.05 0.1
```

## Repository layout

```
calibration/   checkerboard images, calibrate.py, undistort.py, camera_params.npz/json
dataset/       raw_photos/, images/ (undistorted), annotations/instances.json (CVAT COCO 1.0), splits/
models/        Mask R-CNN build, data loader, train.py, evaluate.py, config.yaml, runs/main/ (curves, metrics)
inference/     pipeline.py (marker, scale, mask to mm, refusals), segmenter.py, measure.py (the command)
measurement/   ground_truth.csv, run_accuracy.py, threshold_sweep.py, results/
docs/          PLAN.md, OBJECTIVES.md, DEFINITION_OF_DONE.md
```

## Notes

- Labels: polygons pre-generated with Segment Anything 3 in CVAT, each one checked and corrected by hand. SAM 3 is only a labelling aid; the measuring model is the fine-tuned Mask R-CNN.
- Ground truth is a ruler, not a calliper.
- I used an AI assistant (Claude) to help write the code scaffolding and these docs. The photos, labels, training run and all measurements are my own.
- Not done: items 7, 8 and 10; item 9 only as a skew sweep. See `docs/DEFINITION_OF_DONE.md`.
