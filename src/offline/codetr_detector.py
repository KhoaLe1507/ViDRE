from __future__ import annotations

from typing import Sequence

from src.models.load_codetr import CoDETRDetector
from src.offline.keyframe_selector import SelectedKeyframe


def attach_object_counts(selected_keyframes: Sequence[SelectedKeyframe], detector: CoDETRDetector) -> None:
    if not selected_keyframes:
        return
    detections = detector.detect_images([item.image for item in selected_keyframes])
    for item, object_counts in zip(selected_keyframes, detections):
        item.record.object_counts = object_counts or {}

