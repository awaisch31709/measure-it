"""Generate a printable checkerboard and a printable ArUco reference marker.

Usage:
    python -m calibration.make_board checkerboard --inner-cols 9 --inner-rows 7 --out calibration/checkerboard.png
    python -m calibration.make_board marker --id 0 --out calibration/marker_id0.png

Print at 100% scale (no "fit to page"). Then measure a printed square and the
printed marker side with a calliper. Use the measured values, not the nominal ones.
"""
import argparse

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
PX_PER_SQUARE = 200


def checkerboard(inner_cols: int, inner_rows: int) -> np.ndarray:
    cols, rows = inner_cols + 1, inner_rows + 1
    margin = PX_PER_SQUARE
    img = np.full((rows * PX_PER_SQUARE + 2 * margin, cols * PX_PER_SQUARE + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * PX_PER_SQUARE, margin + c * PX_PER_SQUARE
                img[y:y + PX_PER_SQUARE, x:x + PX_PER_SQUARE] = 0
    return img


def marker(marker_id: int, side_px: int = 800, border_px: int = 200) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    tag = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    return cv2.copyMakeBorder(tag, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    cb = sub.add_parser("checkerboard")
    cb.add_argument("--inner-cols", type=int, default=9)
    cb.add_argument("--inner-rows", type=int, default=7)
    cb.add_argument("--out", default="calibration/checkerboard.png")
    mk = sub.add_parser("marker")
    mk.add_argument("--id", type=int, default=0)
    mk.add_argument("--out", default="calibration/marker_id0.png")
    args = parser.parse_args()

    img = checkerboard(args.inner_cols, args.inner_rows) if args.kind == "checkerboard" else marker(args.id)
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out} ({img.shape[1]}x{img.shape[0]} px)")


if __name__ == "__main__":
    main()
