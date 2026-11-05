"""Fine-tune Mask R-CNN on your own labelled images.

    python -m models.train --config models/config.yaml --run models/runs/main

Trains on the train split. After every epoch it scores the val split and keeps the checkpoint with the best
val mask mAP@0.5:0.95. The test split is never loaded here.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml

from models.build_model import build_model
from models.data import CocoMaskDataset, collate, read_split
from models.evaluate import evaluate_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reseed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32) + worker_id)


def plot_history(history: list, path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.5))
    left.plot(epochs, [row["train_loss"] for row in history])
    left.set(title="train loss", xlabel="epoch")
    right.plot(epochs, [row["val_map_50_95"] for row in history], label="mAP@0.5:0.95")
    right.plot(epochs, [row["val_map_50"] for row in history], label="mAP@0.5")
    right.set(title="val mask mAP", xlabel="epoch")
    right.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="models/config.yaml")
    parser.add_argument("--run", default="models/runs/main")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = CocoMaskDataset(cfg["images_dir"], cfg["annotations"], read_split(cfg["splits_dir"], "train"),
                                augment=True, seed=cfg["seed"])
    val_set = CocoMaskDataset(cfg["images_dir"], cfg["annotations"], read_split(cfg["splits_dir"], "val"))
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=generator,
        num_workers=cfg["num_workers"], worker_init_fn=reseed_worker)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate)

    model = build_model(cfg["pretrained"], cfg["min_size"], cfg["max_size"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    info = {"config": cfg, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "opencv": cv2.__version__, "python": sys.version.split()[0], "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "train_images": len(train_set), "val_images": len(val_set)}
    (run / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    history, best = [], -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for images, targets in train_loader:
            images = [img.to(device).float() / 255 for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val = evaluate_split(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
               "val_map_50": val["map_50"], "val_map_50_95": val["map_50_95"],
               "val_precision": val["precision"], "val_recall": val["recall"]}
        history.append(row)
        print(row)
        if val["map_50_95"] > best:
            best = val["map_50_95"]
            torch.save(model.state_dict(), run / "best.pt")

    with open(run / "history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, run / "curves.png")
    print(f"best val mAP@0.5:0.95 = {best} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
