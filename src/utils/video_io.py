from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

from src.utils.hashing import file_sha256


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for video IO. Install dependencies from requirements.txt.") from exc
    return cv2


def get_video_metadata(video_path: str | os.PathLike[str]) -> dict:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "fps_raw": fps,
        "frame_count": frame_count,
        "duration_raw": duration,
        "width_raw": width,
        "height_raw": height,
        "checksum_raw": file_sha256(video_path),
    }


def read_frames_by_index(video_path: str | os.PathLike[str], frame_indices: Sequence[int]) -> List["Image.Image"]:
    if not frame_indices:
        return []
    cv2 = _require_cv2()
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def read_frame_map_by_index(video_path: str | os.PathLike[str], frame_indices: Sequence[int]) -> dict[int, "Image.Image"]:
    cv2 = _require_cv2()
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = {}
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames[int(frame_index)] = Image.fromarray(frame_rgb)
    cap.release()
    return frames


def iter_video_files(video_dir: str | os.PathLike[str], suffixes: Iterable[str] = (".mp4", ".mkv", ".mov", ".avi")) -> List[Path]:
    root = Path(video_dir)
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffix_set)


def reencode_shot_proxy(
    video_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    start_time: float,
    end_time: float,
    config: dict,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end_time) - float(start_time))
    if duration <= 0:
        raise ValueError(f"Invalid shot duration for proxy: {duration}")

    vf_parts = []
    max_height = int(config.get("max_height", 720))
    if max_height > 0:
        vf_parts.append(f"scale=-2:min({max_height}\\,ih)")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{float(start_time):.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264" if config.get("codec", "h264") == "h264" else str(config.get("codec")),
        "-crf",
        str(config.get("crf", 28)),
        "-preset",
        str(config.get("preset", "veryfast")),
    ]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    if not config.get("keep_audio", False):
        cmd.append("-an")
    cmd.extend([str(output)])
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {completed.stderr[-2000:]}")
    return output
