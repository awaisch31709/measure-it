"""Load saved camera parameters and undistort images.

As a module:
    from calibration.undistort import load_params, undistort
As a script (writes undistorted copies, same file names):
    python -m calibration.undistort --in raw_dir --out undistorted_dir
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PARAMS = Path(__file__).with_name("camera_params.npz")


def load_params(path: Path = DEFAULT_PARAMS):
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"], tuple(int(v) for v in data["image_size"])


def undistort(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, params_size) -> np.ndarray:
    """Undistort keeping the original field of view (alpha=0 crops invalid borders; we keep K so pixel scale is unchanged)."""
    height, width = image.shape[:2]
    if (width, height) != tuple(params_size):
        raise ValueError(
            f"Image is {width}x{height} but the camera was calibrated at {params_size[0]}x{params_size[1]}. "
            "A different resolution is a different camera.")
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    args = parser.parse_args()

    K, dist, size = load_params(Path(args.params))
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(Path(args.src).iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(path))
        cv2.imwrite(str(Path(args.dst) / path.name), undistort(image, K, dist, size))
        count += 1
    print(f"undistorted {count} images -> {args.dst}")


if __name__ == "__main__":
    main()
