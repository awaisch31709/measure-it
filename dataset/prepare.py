"""Undistort your raw dataset photos and give them their planned names.

    python -m dataset.prepare --raw raw_photos --out dataset/images

Raw photos are matched to dataset/splits/all.txt in sorted file-name order (capture order for phone photos),
so the i-th raw photo becomes img_i.jpg. Label the OUTPUT images (the undistorted ones) in CVAT.
"""
import argparse
from pathlib import Path

import cv2

from calibration.undistort import DEFAULT_PARAMS, load_params, undistort

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", default="dataset/images")
    parser.add_argument("--splits", default="dataset/splits")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    args = parser.parse_args()

    planned = (Path(args.splits) / "all.txt").read_text().split()
    raw = sorted(p for p in Path(args.raw).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(raw) != len(planned):
        raise SystemExit(f"{len(raw)} raw photos but the split plans {len(planned)}. "
                         "Re-run make_split.py with --count or fix the photo folder.")

    K, dist, size = load_params(Path(args.params))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for src, name in zip(raw, planned):
        cv2.imwrite(str(Path(args.out) / name), undistort(cv2.imread(str(src)), K, dist, size),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {len(raw)} undistorted images to {args.out}")


if __name__ == "__main__":
    main()
