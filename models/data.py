"""COCO-format instance masks (CVAT 'COCO 1.0' export) restricted to the files named in a split list."""
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO


def read_split(splits_dir: str, name: str) -> list:
    return (Path(splits_dir) / f"{name}.txt").read_text().split()


class CocoMaskDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, annotations: str, file_names: list, augment: bool = False, seed: int = 0):
        self.images_dir = Path(images_dir)
        self.coco = COCO(annotations)
        by_name = {Path(img["file_name"]).name: img for img in self.coco.dataset["images"]}
        missing = [n for n in file_names if n not in by_name]
        if missing:
            raise SystemExit(f"{len(missing)} files in the split have no annotation entry, e.g. {missing[:3]}")
        self.items = [(n, by_name[n]) for n in file_names]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        name, info = self.items[index]
        image = cv2.cvtColor(cv2.imread(str(self.images_dir / name)), cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (info["height"], info["width"]):
            raise SystemExit(f"{name}: image is {image.shape[1]}x{image.shape[0]} but the annotation says "
                             f"{info['width']}x{info['height']}. Label the undistorted images.")
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=info["id"], iscrowd=False))
        masks = np.stack([self.coco.annToMask(a) for a in annotations]) if annotations else \
            np.zeros((0,) + image.shape[:2], np.uint8)

        if self.augment:
            image, masks = self._augment(image, masks)

        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks),
        }
        return torch.from_numpy(image).permute(2, 0, 1).contiguous(), target

    def _augment(self, image, masks):
        """Flips and brightness/contrast jitter. None of these change the object's real size."""
        if self.rng.random() < 0.5:
            image, masks = image[:, ::-1], masks[:, :, ::-1]
        if self.rng.random() < 0.5:
            image, masks = image[::-1], masks[:, ::-1]
        gain, bias = self.rng.uniform(0.75, 1.25), self.rng.uniform(-25, 25)
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image), np.ascontiguousarray(masks)


def collate(batch):
    return tuple(zip(*batch))
