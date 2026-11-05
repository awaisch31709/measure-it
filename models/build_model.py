"""Mask R-CNN (ResNet-50 FPN v2) with a two-class head: background and the measured object."""
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2


def build_model(pretrained: bool, min_size: int = 800, max_size: int = 1333):
    """pretrained=True starts from COCO weights and fine-tunes on your images; False gives random weights."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None, min_size=min_size,
                                     max_size=max_size, box_detections_per_img=20)
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, NUM_CLASSES)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, NUM_CLASSES)
    return model
