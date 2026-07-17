from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


class TransNetV2ShotModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    def load(self) -> "TransNetV2ShotModel":
        if self.model is not None:
            return self
        model_cfg = self.config["models"]["transnetv2"]
        external_dir = Path(model_cfg["external_code_dir"]).resolve()
        if str(external_dir) not in sys.path:
            sys.path.insert(0, str(external_dir))
        try:
            from transnetv2 import TransNetV2
        except ImportError as exc:
            raise RuntimeError(f"Cannot import TransNetV2 from {external_dir}.") from exc
        weights_dir = Path(model_cfg["weights_dir"])
        if not (weights_dir / "saved_model.pb").exists():
            raise FileNotFoundError(f"Missing TransNetV2 SavedModel under {weights_dir}.")
        self.model = TransNetV2(str(weights_dir))
        return self

    def predict_scenes(self, video_path: str, threshold: float) -> List[Tuple[int, int]]:
        self.load()
        assert self.model is not None
        _, single_frame_predictions, _ = self.model.predict_video(video_path)
        scenes = self.model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
        return [(int(start), int(end)) for start, end in scenes.tolist()]


def load_transnetv2(config: Dict[str, Any]) -> TransNetV2ShotModel:
    return TransNetV2ShotModel(config).load()

