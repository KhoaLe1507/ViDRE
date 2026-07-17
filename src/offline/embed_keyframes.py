from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from src.models.load_openclip import OpenCLIPH14Encoder
from src.offline.keyframe_selector import SelectedKeyframe


def embed_selected_keyframes(
    selected_keyframes: Sequence[SelectedKeyframe],
    openclip_encoder: OpenCLIPH14Encoder,
    config: Dict[str, Any],
) -> Tuple[List[List[float]], List[List[float]]]:
    beit3_vectors = [item.beit3_vector for item in selected_keyframes]
    images = [item.image for item in selected_keyframes]
    openclip_vectors = openclip_encoder.encode_images(
        images,
        batch_size=int(config.get("offline", {}).get("openclip_image_batch_size", 32)),
    )
    return beit3_vectors, openclip_vectors

