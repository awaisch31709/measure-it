"""Evaluate a checkpoint on a split.

    python -m models.evaluate --ckpt models/runs/main/best.pt --split val
    python -m models.evaluate --ckpt models/runs/main/best.pt --split test --final

The test split needs --final so it is never touched while you are still tuning. Run it once, at the end.

Metric definitions (all on instance masks, one object class):
  map_50, map_50_95  torchmetrics COCO mask mAP over all predictions
  precision, recall  at score >= --score-thr, a prediction counts as correct if mask IoU >= 0.5
  mean_mask_iou      for every ground-truth mask, IoU with its best-overlapping prediction (0 if none)
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torchmetrics.detection import MeanAveragePrecision

from models.build_model import build_model
from models.data import CocoMaskDataset, collate, read_split


def iou_matrix(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    gt_f, pred_f = gt.flatten(1).float(), pred.flatten(1).float()
    inter = gt_f @ pred_f.T
    union = gt_f.sum(1)[:, None] + pred_f.sum(1)[None, :] - inter
    return inter / union.clamp(min=1)


@torch.no_grad()
def evaluate_split(model, loader, device, score_thr: float = 0.5, iou_thr: float = 0.5) -> dict:
    model.eval()
    metric = MeanAveragePrecision(iou_type="segm")
    best_ious, tp, fp, fn = [], 0, 0, 0
    for images, targets in loader:
        outputs = model([img.to(device).float() / 255 for img in images])
        for out, tgt in zip(outputs, targets):
            pred_masks = (out["masks"][:, 0] > 0.5).cpu()
            scores = out["scores"].cpu()
            gt_masks = tgt["masks"].bool()
            metric.update([{"masks": pred_masks, "scores": scores, "labels": out["labels"].cpu()}],
                          [{"masks": gt_masks, "labels": tgt["labels"]}])

            keep = scores >= score_thr
            kept_masks, kept_scores = pred_masks[keep], scores[keep]
            if len(kept_masks) == 0:
                best_ious += [0.0] * len(gt_masks)
                fn += len(gt_masks)
                continue
            ious = iou_matrix(gt_masks, kept_masks)
            best_ious += ious.max(dim=1).values.tolist()
            matched_gt = set()
            for p in torch.argsort(kept_scores, descending=True).tolist():
                candidates = [(ious[g, p].item(), g) for g in range(len(gt_masks)) if g not in matched_gt]
                best = max(candidates, default=(0.0, -1))
                if best[0] >= iou_thr:
                    matched_gt.add(best[1])
                    tp += 1
                else:
                    fp += 1
            fn += len(gt_masks) - len(matched_gt)

    computed = metric.compute()
    # torchmetrics reports -1 when there is nothing to score (for example no predictions at all); count that as 0.
    return {
        "map_50": round(max(float(computed["map_50"]), 0.0), 4),
        "map_50_95": round(max(float(computed["map"]), 0.0), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "mean_mask_iou": round(sum(best_ious) / max(len(best_ious), 1), 4),
        "score_thr": score_thr,
        "images": len(loader.dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final", action="store_true", help="required to evaluate the test split")
    parser.add_argument("--config", default="models/config.yaml")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("Refusing to evaluate the test split without --final.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False, min_size=cfg["min_size"], max_size=cfg["max_size"])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device)

    dataset = CocoMaskDataset(cfg["images_dir"], cfg["annotations"], read_split(cfg["splits_dir"], args.split))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    results = evaluate_split(model, loader, device, args.score_thr)
    results.update(split=args.split, checkpoint=args.ckpt)
    print(json.dumps(results, indent=2))
    out = Path(args.out or Path(args.ckpt).with_name(f"{args.split}_metrics.json"))
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
