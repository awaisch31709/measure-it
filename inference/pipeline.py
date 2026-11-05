"""Image -> millimetres pipeline: undistort, segment, find the reference marker, convert.

The pixel-to-millimetre conversion (see `px_per_mm_from_marker` and `measure_mask`) is only valid when the
object and the reference marker lie in the same flat plane and the camera looks roughly straight down at it.
"""
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from calibration.undistort import undistort

ARUCO_DICT = cv2.aruco.DICT_4X4_50
BORDER_MARGIN_PX = 2

STATUS_OK = "ok"
STATUS_OBJECT_NOT_FOUND = "object_not_found"
STATUS_REFERENCE_NOT_FOUND = "reference_not_found"
STATUS_LOW_CONFIDENCE = "refused_low_confidence"
STATUS_TILTED = "refused_tilted_plane"
STATUS_CLIPPED = "refused_object_touches_frame_edge"
EXIT_CODES = {
    STATUS_OK: 0,
    STATUS_OBJECT_NOT_FOUND: 2,
    STATUS_REFERENCE_NOT_FOUND: 3,
    STATUS_LOW_CONFIDENCE: 4,
    STATUS_TILTED: 4,
    STATUS_CLIPPED: 4,
}

# A segmenter takes a BGR image and returns [(bool mask HxW, score), ...] sorted by descending score.
Segmenter = Callable[[np.ndarray], list]


@dataclass
class Config:
    marker_mm: float
    marker_id: int = 0
    min_confidence: float = 0.8
    max_skew: float = 0.02
    floor_score: float = 0.05
    refuse_clipped: bool = True


@dataclass
class Result:
    status: str
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    confidence: Optional[float] = None
    px_per_mm: Optional[float] = None
    skew: Optional[float] = None
    touches_border: Optional[bool] = None
    detail: str = ""
    _overlay_items: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("_overlay_items")
        if self.status != STATUS_OK:
            data["width_mm"] = data["height_mm"] = None
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in data.items()}


def detect_marker(image_bgr: np.ndarray, marker_id: int) -> Optional[np.ndarray]:
    """Return the 4 marker corners as a (4, 2) float array, or None if that marker id is not seen."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for marker_corners, found_id in zip(corners, ids.ravel()):
        if int(found_id) == marker_id:
            return marker_corners.reshape(4, 2).astype(np.float64)
    return None


def px_per_mm_from_marker(corners: np.ndarray, marker_mm: float) -> tuple[float, float]:
    """Scale (pixels per mm) from the marker's mean side length, plus a perspective-skew indicator.

    skew = largest relative difference between opposite sides. A marker seen face-on has skew ~0;
    a tilted plane makes the near side longer than the far side.
    """
    sides = np.array([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])
    skew = max(abs(sides[0] - sides[2]), abs(sides[1] - sides[3])) / sides.mean()
    return float(sides.mean() / marker_mm), float(skew)


def measure_mask(mask: np.ndarray, px_per_mm: float) -> tuple[float, float, np.ndarray]:
    """Width (longer side) and height (shorter side) of the mask's minimum-area rectangle, in mm."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return max(side_a, side_b) / px_per_mm, min(side_a, side_b) / px_per_mm, box


def touches_border(mask: np.ndarray) -> bool:
    m = BORDER_MARGIN_PX
    return bool(mask[:m].any() or mask[-m:].any() or mask[:, :m].any() or mask[:, -m:].any())


def measure_image(image_bgr: np.ndarray, segmenter: Segmenter, cfg: Config,
                  camera=None) -> Result:
    """Run the full pipeline. `camera` is (camera_matrix, dist_coeffs, size) or None to skip undistortion."""
    if camera is not None:
        image_bgr = undistort(image_bgr, *camera)

    detections = segmenter(image_bgr)
    detections = [d for d in detections if d[1] >= cfg.floor_score]
    if not detections:
        return Result(STATUS_OBJECT_NOT_FOUND, detail="no object mask above the score floor",
                      _overlay_items={"image": image_bgr})
    mask, score = detections[0]

    marker = detect_marker(image_bgr, cfg.marker_id)
    items = {"image": image_bgr, "mask": mask, "marker": marker}
    if marker is None:
        return Result(STATUS_REFERENCE_NOT_FOUND, confidence=float(score),
                      detail=f"ArUco marker id {cfg.marker_id} not detected", _overlay_items=items)

    px_per_mm, skew = px_per_mm_from_marker(marker, cfg.marker_mm)
    width_mm, height_mm, box = measure_mask(mask, px_per_mm)
    items["box"] = box
    clipped = touches_border(mask)
    base = dict(confidence=float(score), px_per_mm=px_per_mm, skew=skew, touches_border=clipped,
                width_mm=width_mm, height_mm=height_mm, _overlay_items=items)

    if score < cfg.min_confidence:
        return Result(STATUS_LOW_CONFIDENCE, detail=f"score {score:.2f} < {cfg.min_confidence}", **base)
    if skew > cfg.max_skew:
        return Result(STATUS_TILTED, detail=f"marker skew {skew:.3f} > {cfg.max_skew}", **base)
    if clipped and cfg.refuse_clipped:
        return Result(STATUS_CLIPPED, detail="object mask touches the image border", **base)
    return Result(STATUS_OK, **base)


def draw_overlay(result: Result) -> np.ndarray:
    items = result._overlay_items
    canvas = items["image"].copy()
    if "mask" in items:
        tint = np.zeros_like(canvas)
        tint[items["mask"]] = (0, 200, 0)
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.45, 0)
    if items.get("marker") is not None:
        cv2.polylines(canvas, [items["marker"].astype(np.int32)], True, (255, 120, 0), 3)
    if "box" in items:
        cv2.polylines(canvas, [items["box"].astype(np.int32)], True, (0, 0, 255), 3)
    scale = max(canvas.shape[:2]) / 1600
    if result.status == STATUS_OK:
        text = f"W {result.width_mm:.1f} mm  H {result.height_mm:.1f} mm  conf {result.confidence:.2f}"
        colour = (0, 160, 0)
    else:
        text = f"{result.status}: {result.detail}"
        colour = (0, 0, 220)
    cv2.putText(canvas, text, (int(20 * scale), int(50 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * scale, (255, 255, 255), int(6 * scale), cv2.LINE_AA)
    cv2.putText(canvas, text, (int(20 * scale), int(50 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * scale, colour, int(3 * scale), cv2.LINE_AA)
    return canvas
