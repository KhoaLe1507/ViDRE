from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.models.load_transnetv2 import TransNetV2ShotModel
from src.schemas import ShotRecord
from src.utils.ids import make_shot_id


def detect_shots(
    video_path: str,
    video_id: str,
    video_metadata: Dict[str, Any],
    config: Dict[str, Any],
    model: TransNetV2ShotModel | None = None,
) -> List[ShotRecord]:
    fps = float(video_metadata["fps_raw"])
    if fps <= 0:
        raise ValueError(f"Cannot detect shots without valid FPS for {video_id}.")
    frame_count = int(video_metadata["frame_count"])
    threshold = float(config["transnetv2"].get("shot_boundary_threshold", 0.5))
    model = model or TransNetV2ShotModel(config)
    raw_scenes = model.predict_scenes(video_path, threshold=threshold)
    scenes = _sanitize_scenes(raw_scenes, frame_count)
    if config["transnetv2"].get("merge_short_shots", True):
        scenes = _merge_short_scenes(scenes, fps, float(config["transnetv2"].get("min_shot_duration_sec", 1.0)))
    return [_scene_to_record(video_id, idx, scene, fps, video_metadata, config) for idx, scene in enumerate(scenes)]


def _sanitize_scenes(scenes: Iterable[Tuple[int, int]], frame_count: int) -> List[Tuple[int, int]]:
    sanitized: List[Tuple[int, int]] = []
    for start, end in scenes:
        start_i = max(0, min(int(start), max(0, frame_count - 1)))
        end_i = max(0, min(int(end), max(0, frame_count - 1)))
        if end_i < start_i:
            continue
        sanitized.append((start_i, end_i))
    if not sanitized and frame_count > 0:
        return [(0, frame_count - 1)]
    return sanitized


def _merge_short_scenes(scenes: List[Tuple[int, int]], fps: float, min_duration_sec: float) -> List[Tuple[int, int]]:
    if not scenes:
        return scenes
    merged: List[Tuple[int, int]] = []
    for scene in scenes:
        duration = (scene[1] - scene[0] + 1) / fps
        if duration < min_duration_sec and merged:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, scene[1])
        else:
            merged.append(scene)
    if len(merged) > 1:
        first = merged[0]
        first_duration = (first[1] - first[0] + 1) / fps
        if first_duration < min_duration_sec:
            second = merged[1]
            merged[1] = (first[0], second[1])
            merged = merged[1:]
    return merged


def _scene_to_record(
    video_id: str,
    shot_index: int,
    scene: Tuple[int, int],
    fps: float,
    video_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> ShotRecord:
    start_frame, end_frame = scene
    shot_id = make_shot_id(video_id, shot_index, start_frame, end_frame)
    start_time = start_frame / fps
    end_time = min(float(video_metadata["duration_raw"]), (end_frame + 1) / fps)
    return ShotRecord(
        video_id=video_id,
        shot_id=shot_id,
        shot_index=shot_index,
        shot_start_frame=start_frame,
        shot_end_frame=end_frame,
        shot_start_time_raw=start_time,
        shot_end_time_raw=end_time,
        fps_raw=fps,
        duration_raw=max(0.0, end_time - start_time),
        model_version=config["project"]["model_version"],
        config_version=config["project"]["config_version"],
        processing_status="PENDING",
    )
