# Plan

## Object, reference and camera

- Object: Casio fx-991ES PLUS calculator, about 161 x 80 mm, 15 mm thick (measured with a ruler). Chosen because it is rigid and has a clear outline. Its thickness puts its top edge closer to the camera than the marker, so I expect a small over-estimate (known limitation).
- Reference: printed ArUco marker (DICT_4X4_50, id 0) glued to stiff card. Black square measured with a ruler: 50 mm.
- Calibration board: printed checkerboard, 9 x 7 inner corners, square measured with a ruler: 20 mm, clipped flat on a clipboard.
- Camera: my phone's main camera, 1x, flash off, phone held in landscape about 40 cm above the surface. All images used at 1280 x 960.
- Plane: object and marker lie flat on the same surface, phone held roughly parallel to it.
- Hardware: Windows PC with 8 GB RAM and no GPU, so training runs on Google Colab (free GPU).

## Order of work and time budget (8 hours)

| Stage | Hours | Done when |
|---|---|---|
| Calibration (item 1) | 1.0 | 20+ board images, reprojection error under 1 px, straight edges stay straight |
| Photos, undistortion, labelling (item 2) | 2.0 | 80 undistorted images labelled with polygon masks, split fixed by this commit |
| Train + evaluate on Colab (item 3) | 1.5 | Test metrics reported: mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall |
| Image-to-mm command (item 4) | 1.0 | One command prints width and height in mm, handles object and reference not found |
| Accuracy vs ruler (item 5) | 1.0 | 10+ instances measured by hand and by the system |
| Raw vs undistorted (item 6) | 0.5 | Same instances measured both ways, errors side by side |
| Docs, README, recording | 1.0 | Three docs, README, Loom under 5 minutes |

## Decided up front

- Segmentation model: Mask R-CNN (torchvision, COCO-pretrained, fine-tuned on my images). Not YOLO or Roboflow. Chosen because it gives instance masks and the mAP metrics directly.
- Width and height: longer and shorter side of the minimum-area rectangle around the mask.
- Reference detection: classical OpenCV ArUco detection, not a second trained model, because the marker corners are found exactly and "reference not found" comes for free.
- Split: 80 images, 56 / 16 / 8, fixed by dataset/splits before any photo of the object was taken. Test images are only used once, at the end.

## Skipping

Items 7 to 10 (speed/accuracy target, breaking condition, refusal threshold, decision record) are likely to be skipped because of limited time. If time remains I will do item 7 first.

## Changes

<Add a dated line here whenever the plan changes, and why.>
