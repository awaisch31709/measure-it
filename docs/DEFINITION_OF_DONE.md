# Definition of Done

| Done when | Evidence |
|---|---|
| [x] Calibration reprojection error reported, with the image count | RMS 0.68 px from 28 board images: `calibration/camera_params.json` |
| [x] Straight edges stay straight after undistortion | `calibration/undistortion_check.jpg` |
| [x] Plan and split file lists committed before any code (commit link) | [2401740](https://github.com/awaisch31709/measure-it/commit/2401740) is the first commit on dev-test01 |
| [x] 70+ images taken by me, undistorted, labelled; labelling tool named | 80 photos (`dataset/raw_photos`), undistorted (`dataset/images`), polygons from CVAT (SAM 3 pre-labels, every one checked by hand): `dataset/annotations/instances.json` |
| [x] Model trained on Colab, test metrics reported (IoU, mAP@0.5, mAP@0.5:0.95, precision, recall) | `models/runs/main/test_metrics.json`, copied into docs/OBJECTIVES.md |
| [x] Command runs on a new image and prints width and height in mm | `python -m inference.measure ...`, see README |
| [x] Object-not-found and reference-not-found both handled | exit codes 2 and 3, tested on a photo without the calculator and one without the marker (README) |
| [x] Accuracy table for 10+ instances with raw outputs saved | 22 instances: `measurement/results/accuracy.csv` and `accuracy_summary.json` |
| [x] Raw vs undistorted errors side by side | same files, raw and undistorted columns |
| [x] MO-1 target met, or missed with the reason written down | Missed: 3.7 mm mean error against a 2 mm target. Reason in docs/OBJECTIVES.md |
| [x] Weights linked with SHA-256 in the README | [Release v1.0](https://github.com/awaisch31709/measure-it/releases/tag/v1.0) |
| [x] Recording under 5 minutes (4:50) | Loom link: https://www.loom.com/share/7e1ebed843c249a9be46323fef108838|
| [x] Everything I did not finish is listed below | below |

## Not finished

- Items 7 (speed/accuracy target), 8 (breaking condition) and 10 (decision record) were not attempted because of time.
- Item 9 is only partly done: a skew threshold sweep, no confidence sweep.
- Ground truth comes from a ruler (about 1 mm resolution), not a calliper, and from one measurement of the calculator rather than one per photo.
- The accuracy photos are the held-out val/test photos, not a fresh set taken for this step.
- MO-1 target missed (see OBJECTIVES). The thickness bias is not corrected.
- Inference speed was not timed.
