from __future__ import annotations


def make_video_id(video_path: str) -> str:
    name = video_path.replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def make_shot_id(video_id: str, shot_index: int, start_frame: int, end_frame: int) -> str:
    return f"{video_id}_shot_{shot_index:05d}_{start_frame}_{end_frame}"


def make_keyframe_id(video_id: str, shot_id: str, frame_index_raw: int) -> str:
    return f"{video_id}_{shot_id}_kf_{frame_index_raw}"


def make_eval_run_id(dataset_name: str, config_version: str, model_version: str, timestamp_utc: str) -> str:
    clean_timestamp = timestamp_utc.replace(":", "").replace("-", "").replace(".", "")
    return f"{dataset_name}_{config_version}_{model_version}_{clean_timestamp}"

