from __future__ import annotations

import os
import sys
from collections import OrderedDict
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class CoDETRDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.inference_detector = None

    def load(self) -> "CoDETRDetector":
        if self.model is not None:
            return self
        model_cfg = self.config["models"]["codetr"]
        external_dir = Path(model_cfg["external_code_dir"]).resolve()
        if str(external_dir) not in sys.path:
            sys.path.insert(0, str(external_dir))
        try:
            import mmcv_custom  # noqa: F401
            import projects  # noqa: F401
            import mmdet.apis.inference as mmdet_inference

            mmdet_inference.load_checkpoint = _load_checkpoint_without_ema_keys
            from mmdet.apis import init_detector, inference_detector
        except ImportError as exc:
            raise RuntimeError("Co-DETR requires mmdet/mmcv installed per external/Co-DETR requirements.") from exc
        config_path = Path(model_cfg["config"])
        checkpoint_path = Path(model_cfg["checkpoint"])
        if not config_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError("Missing Co-DETR config or checkpoint.")
        self.model = init_detector(str(config_path), str(checkpoint_path), device=str(model_cfg.get("device", "cuda:0")))
        self.inference_detector = inference_detector
        return self

    def detect_images(self, images: Sequence[Any]) -> List[Dict[str, int]]:
        self.load()
        assert self.model is not None and self.inference_detector is not None
        return [self._count_detections(self.inference_detector(self.model, _to_numpy_rgb(image))) for image in images]

    def _count_detections(self, result: Any) -> Dict[str, int]:
        threshold = float(self.config["models"]["codetr"].get("confidence_threshold", 0.35))
        if isinstance(result, tuple):
            bbox_result = result[0]
        else:
            bbox_result = result
        counts: Counter[str] = Counter()
        for class_idx, detections in enumerate(bbox_result):
            if class_idx >= len(COCO_CLASSES):
                continue
            for det in detections:
                if len(det) >= 5 and float(det[4]) >= threshold:
                    counts[COCO_CLASSES[class_idx]] += 1
        return dict(counts)


def _to_numpy_rgb(image: Any):
    from PIL import Image
    import numpy as np

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image type for Co-DETR: {type(image)!r}")


def _load_checkpoint_without_ema_keys(model, filename, map_location="cpu", strict=False, logger=None):
    from mmcv.runner import _load_checkpoint, load_state_dict

    checkpoint = _load_checkpoint(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("ema_"):
            continue
        cleaned_state_dict[key] = value

    load_state_dict(model, cleaned_state_dict, strict=strict, logger=logger)
    return checkpoint


def load_codetr(config: Dict[str, Any]) -> CoDETRDetector:
    return CoDETRDetector(config).load()
