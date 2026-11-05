"""Calibrate the camera from checkerboard photos and save the parameters.

Usage:
    python -m calibration.calibrate --images calibration/images --inner-cols 9 --inner-rows 7 \
        --square-mm 24.8 --out calibration/camera_params.npz

--inner-cols/--inner-rows count INNER corners (where four squares meet), not squares.
--square-mm is the square side you measured with a calliper.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(image_path: Path, pattern: tuple[int, int]):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None
    found, corners = cv2.findChessboardCorners(gray, pattern)
    if not found:
        return gray.shape[::-1], None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return gray.shape[::-1], corners


def per_image_error(objp, corners, rvec, tvec, K, dist) -> float:
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    diff = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--inner-cols", type=int, required=True)
    parser.add_argument("--inner-rows", type=int, required=True)
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--out", default="calibration/camera_params.npz")
    args = parser.parse_args()

    pattern = (args.inner_cols, args.inner_rows)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * args.square_mm

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    object_points, image_points, used, rejected = [], [], [], []
    image_size = None
    for path in paths:
        size, corners = find_corners(path, pattern)
        if size is not None:
            if image_size is not None and size != image_size:
                sys.exit(f"{path.name} is {size}, expected {image_size}. Calibration images must share one resolution.")
            image_size = size
        if corners is None:
            rejected.append(path.name)
            continue
        object_points.append(objp)
        image_points.append(corners)
        used.append(path.name)

    print(f"images found: {len(paths)}  usable: {len(used)}  rejected: {len(rejected)}")
    for name in rejected:
        print(f"  no board detected: {name}")
    if len(used) < 20:
        sys.exit(f"Need 20 or more usable images, have {len(used)}.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    errors = [per_image_error(objp, c, r, t, K, dist) for c, r, t in zip(image_points, rvecs, tvecs)]

    np.savez(args.out, camera_matrix=K, dist_coeffs=dist, image_size=np.array(image_size),
             rms=rms, square_mm=args.square_mm, n_images=len(used))
    report = {
        "images_used": len(used),
        "image_size_wh": list(image_size),
        "rms_reprojection_error_px": round(float(rms), 4),
        "per_image_error_px_mean": round(float(np.mean(errors)), 4),
        "per_image_error_px_worst": {"image": used[int(np.argmax(errors))], "error": round(float(np.max(errors)), 4)},
        "camera_matrix": K.round(3).tolist(),
        "dist_coeffs_k1_k2_p1_p2_k3": dist.ravel().round(6).tolist(),
    }
    print(json.dumps(report, indent=2))
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
