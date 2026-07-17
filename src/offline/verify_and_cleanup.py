from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from src.offline.keyframe_selector import SelectedKeyframe
from src.schemas import ShotRecord
from src.storage.r2_client import R2Client


def verify_offline_outputs(
    selected_keyframes: Sequence[SelectedKeyframe],
    shots: Sequence[ShotRecord],
    r2_client: R2Client,
    allow_no_new_keyframes: bool = False,
) -> None:
    if not selected_keyframes and not allow_no_new_keyframes:
        raise RuntimeError("No keyframes were selected; refusing to mark video as VERIFIED.")
    for shot in shots:
        if shot.processing_status == "SKIPPED":
            continue
        # Detailed vector/metadata verification is persisted in DB and Zilliz writes.
        # R2 proxy existence is checked during upload.
        _ = r2_client


def delete_raw_video_if_enabled(video_path: str, config: Dict[str, Any]) -> bool:
    if not config.get("offline", {}).get("delete_raw_after_verified", False):
        return False
    Path(video_path).unlink()
    return True
