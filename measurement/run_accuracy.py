"""Compare the system with calliper measurements, on raw photos and on undistorted ones.

    python -m measurement.run_accuracy --csv measurement/ground_truth.csv --images measurement/raw_images \
        --ckpt models/runs/main/best.pt --marker-mm 40.12

ground_truth.csv columns: image,width_mm,height_mm[,condition]
  width_mm = the longer side, height_mm = the shorter side, both measured by calliper.
  condition (optional) groups rows, e.g. "tilt_10deg", so error can be summarised per condition.

Refusals are switched off here so every measurement is recorded together with its confidence and skew.
measurement/threshold_sweep.py applies refusal thresholds afterwards.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

from calibration.undistort import DEFAULT_PARAMS, load_params
from inference.measure import match_calibration_size
from inference.pipeline import STATUS_OK, Config, measure_image
from inference.segmenter import MaskRCNNSegmenter


def summarise(rows: list, prefix: str) -> dict:
    """Errors over rows where BOTH the raw and the undistorted measurement succeeded."""
    abs_w = [abs(r[f"{prefix}_width_mm"] - r["true_width_mm"]) for r in rows]
    abs_h = [abs(r[f"{prefix}_height_mm"] - r["true_height_mm"]) for r in rows]
    pct = [100 * e / r[k] for e, r, k in zip(abs_w, rows, ["true_width_mm"] * len(rows))] + \
          [100 * e / r[k] for e, r, k in zip(abs_h, rows, ["true_height_mm"] * len(rows))]
    pooled = abs_w + abs_h
    return {
        "n_instances": len(rows),
        "mae_width_mm": round(statistics.mean(abs_w), 3),
        "mae_height_mm": round(statistics.mean(abs_h), 3),
        "mae_mm": round(statistics.mean(pooled), 3),
        "median_abs_error_mm": round(statistics.median(pooled), 3),
        "worst_abs_error_mm": round(max(pooled), 3),
        "mean_pct_error": round(statistics.mean(pct), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="measurement/ground_truth.csv")
    parser.add_argument("--images", default="measurement/raw_images")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--marker-mm", type=float, required=True)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--out", default="measurement/results")
    args = parser.parse_args()

    segmenter = MaskRCNNSegmenter(Path(args.ckpt))
    camera = load_params(Path(args.params))
    cfg = Config(marker_mm=args.marker_mm, marker_id=args.marker_id, min_confidence=0.0,
                 max_skew=float("inf"), refuse_clipped=False)

    records = []
    for truth in csv.DictReader(open(args.csv)):
        image = cv2.imread(str(Path(args.images) / truth["image"]))
        if image is None:
            raise SystemExit(f"Cannot read {truth['image']}")
        image = match_calibration_size(image, camera[2])
        record = {"image": truth["image"], "condition": truth.get("condition", ""),
                  "true_width_mm": float(truth["width_mm"]), "true_height_mm": float(truth["height_mm"])}
        for prefix, cam in (("raw", None), ("und", camera)):
            result = measure_image(image, segmenter, cfg, camera=cam)
            record.update({f"{prefix}_status": result.status, f"{prefix}_width_mm": result.width_mm,
                           f"{prefix}_height_mm": result.height_mm, f"{prefix}_confidence": result.confidence,
                           f"{prefix}_skew": result.skew, f"{prefix}_touches_border": result.touches_border})
        records.append(record)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "accuracy.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    usable = [r for r in records if r["raw_status"] == STATUS_OK and r["und_status"] == STATUS_OK]
    print(f"{len(records)} instances, {len(usable)} measured on both raw and undistorted images")
    for r in records:
        if r not in usable:
            print(f"  not measured: {r['image']} raw={r['raw_status']} undistorted={r['und_status']}")
    if not usable:
        raise SystemExit("Nothing measured; check the model, the marker size and the marker id.")

    print(f"\n{'image':<14}{'true W':>8}{'true H':>8}{'raw W':>8}{'raw H':>8}{'und W':>8}{'und H':>8}")
    for r in usable:
        print(f"{r['image']:<14}{r['true_width_mm']:>8.2f}{r['true_height_mm']:>8.2f}"
              f"{r['raw_width_mm']:>8.2f}{r['raw_height_mm']:>8.2f}{r['und_width_mm']:>8.2f}{r['und_height_mm']:>8.2f}")

    summary = {"raw": summarise(usable, "raw"), "undistorted": summarise(usable, "und")}
    by_condition = defaultdict(list)
    for r in usable:
        by_condition[r["condition"]].append(r)
    if len(by_condition) > 1:
        summary["by_condition_undistorted"] = {c: summarise(rs, "und") for c, rs in by_condition.items()}
    print("\n" + json.dumps(summary, indent=2))
    (out / "accuracy_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
