"""One command, image to millimetres.

    python -m inference.measure photo.jpg --ckpt models/runs/main/best.pt --marker-mm 40.12 --out overlay.png

Prints one JSON line. Exit code: 0 measured, 2 object not found, 3 reference not found, 4 refused.
--marker-mm is the side length of your printed ArUco marker, measured with a calliper. There is no default.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

from calibration.undistort import DEFAULT_PARAMS, load_params
from inference.pipeline import EXIT_CODES, Config, draw_overlay, measure_image
from inference.segmenter import MaskRCNNSegmenter


def match_calibration_size(image, size):
    """The camera was calibrated at one resolution (width, height). A portrait photo is turned to landscape, and a
    photo from the same camera at a higher resolution with the same 4:3 shape is scaled down to the calibrated size.
    Anything else is refused, because K and the distortion terms only hold for the calibrated image geometry."""
    w, h = int(size[0]), int(size[1])
    if image.shape[0] > image.shape[1] and w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ih, iw = image.shape[:2]
    if (iw, ih) == (w, h):
        return image
    if abs(iw / ih - w / h) > 0.01:
        sys.exit(f"Image is {iw}x{ih}; the calibration is {w}x{h}. Use a photo with the same 4:3 shape.")
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--marker-mm", type=float, required=True)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--max-skew", type=float, default=0.02)
    parser.add_argument("--out", default="overlay.png")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Cannot read image: {args.image}")

    camera = load_params(Path(args.params))
    image = match_calibration_size(image, camera[2])
    cfg = Config(marker_mm=args.marker_mm, marker_id=args.marker_id,
                 min_confidence=args.min_confidence, max_skew=args.max_skew)
    result = measure_image(image, MaskRCNNSegmenter(Path(args.ckpt)), cfg, camera=camera)
    cv2.imwrite(args.out, draw_overlay(result))
    print(json.dumps(result.to_dict()))
    sys.exit(EXIT_CODES[result.status])


if __name__ == "__main__":
    main()
