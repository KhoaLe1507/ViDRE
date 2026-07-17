from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from src.schemas import ShotRecord
from src.storage.cockroach_client import CockroachClient
from src.storage.r2_client import R2Client
from src.utils.config import ensure_dir, get_config_value
from src.utils.hashing import file_sha256
from src.utils.video_io import reencode_shot_proxy


def reencode_upload_shot_proxies(
    video_path: str,
    shots: Sequence[ShotRecord],
    config: Dict[str, Any],
    r2_client: R2Client,
    cockroach: CockroachClient,
) -> None:
    proxy_dir = ensure_dir(get_config_value(config, "paths.local_proxy_dir", "outputs/shot_proxies"))
    container = str(config["shot_proxy"].get("container", "mp4"))
    for shot in shots:
        local_path = Path(proxy_dir) / shot.video_id / f"{shot.shot_id}.{container}"
        reencode_shot_proxy(
            video_path=video_path,
            output_path=local_path,
            start_time=shot.shot_start_time_raw,
            end_time=shot.shot_end_time_raw,
            config=config["shot_proxy"],
        )
        checksum = file_sha256(local_path)
        object_key = r2_client.make_shot_proxy_key(shot.video_id, shot.shot_id, container=container)
        r2_client.upload_file(local_path, object_key, content_type="video/mp4")
        if not r2_client.object_exists(object_key) or not r2_client.verify_checksum(object_key, checksum):
            raise RuntimeError(f"R2 verification failed for {object_key}.")
        cockroach.update_shot_proxy(shot.shot_id, object_key, checksum, status="VERIFIED")

