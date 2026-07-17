from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.models.load_beit3 import BEiT3Encoder
from src.schemas import KeyframeRecord, ShotRecord
from src.utils.ids import make_keyframe_id
from src.utils.vector import cosine_distance
from src.utils.video_io import read_frame_map_by_index


@dataclass
class SelectedKeyframe:
    record: KeyframeRecord
    image: Any
    beit3_vector: List[float]


def select_keyframes_for_shot(
    video_path: str,
    shot: ShotRecord,
    config: Dict[str, Any],
    beit3_encoder: BEiT3Encoder,
) -> List[SelectedKeyframe]:
    candidate_indices = _candidate_frame_indices(shot, int(config["keyframe_selection"].get("sample_stride_frames", 10)))
    fallback_used = False
    if not candidate_indices:
        fallback_used = True
        candidate_indices = [int((shot.shot_start_frame + shot.shot_end_frame) // 2)]

    frame_map = read_frame_map_by_index(video_path, candidate_indices)
    actual_indices = [idx for idx in candidate_indices if idx in frame_map]
    if not actual_indices:
        fallback = int((shot.shot_start_frame + shot.shot_end_frame) // 2)
        frame_map = read_frame_map_by_index(video_path, [fallback])
        actual_indices = [fallback] if fallback in frame_map else []
        fallback_used = True
    if not actual_indices:
        return []

    images = [frame_map[idx] for idx in actual_indices]
    vectors = beit3_encoder.encode_images(images)
    selected: List[SelectedKeyframe] = []
    previous_vector: Optional[List[float]] = None
    last_selected_vector: Optional[List[float]] = None

    threshold_prev = float(config["keyframe_selection"].get("threshold_prev_candidate", 0.075))
    threshold_last = float(config["keyframe_selection"].get("threshold_last_keyframe", 0.10))

    for idx, image, vector in zip(actual_indices, images, vectors):
        if not selected:
            reason = "fallback_middle_frame" if fallback_used else "first_candidate"
            selected.append(_build_selected_keyframe(shot, idx, image, vector, reason, None, None, config))
            previous_vector = vector
            last_selected_vector = vector
            continue

        assert previous_vector is not None and last_selected_vector is not None
        dist_prev = cosine_distance(vector, previous_vector)
        dist_last = cosine_distance(vector, last_selected_vector)
        if dist_prev >= threshold_prev or dist_last >= threshold_last:
            selected.append(_build_selected_keyframe(shot, idx, image, vector, "distance_threshold", dist_prev, dist_last, config))
            last_selected_vector = vector
        previous_vector = vector
    return selected


def _candidate_frame_indices(shot: ShotRecord, stride: int) -> List[int]:
    stride = max(1, stride)
    if shot.shot_end_frame < shot.shot_start_frame:
        return []
    return list(range(shot.shot_start_frame, shot.shot_end_frame + 1, stride))


def _build_selected_keyframe(
    shot: ShotRecord,
    frame_index: int,
    image: Any,
    vector: Sequence[float],
    reason: str,
    dist_prev: Optional[float],
    dist_last: Optional[float],
    config: Dict[str, Any],
) -> SelectedKeyframe:
    timestamp_raw = frame_index / shot.fps_raw
    record = KeyframeRecord(
        keyframe_id=make_keyframe_id(shot.video_id, shot.shot_id, frame_index),
        video_id=shot.video_id,
        shot_id=shot.shot_id,
        shot_index=shot.shot_index,
        frame_index_raw=frame_index,
        timestamp_raw=timestamp_raw,
        timestamp_in_shot=timestamp_raw - shot.shot_start_time_raw,
        selection_reason=reason,
        beit3_selection_distance_prev=dist_prev,
        beit3_selection_distance_last_keyframe=dist_last,
        model_version=config["project"]["model_version"],
        config_version=config["project"]["config_version"],
    )
    return SelectedKeyframe(record=record, image=image, beit3_vector=list(vector))

