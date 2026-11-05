"""Mask R-CNN segmenter that satisfies the `Segmenter` interface used by the pipeline."""
from pathlib import Path

import cv2
import numpy as np
import torch

from models.build_model import build_model


class MaskRCNNSegmenter:
    def __init__(self, checkpoint: Path, device: str = None, min_size: int = 800, max_size: int = 1333):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(pretrained=False, min_size=min_size, max_size=max_size)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray) -> list:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).float() / 255
        output = self.model([tensor])[0]
        masks = (output["masks"][:, 0] > 0.5).cpu().numpy()
        scores = output["scores"].cpu().numpy()
        order = np.argsort(-scores)
        return [(masks[i], float(scores[i])) for i in order]
