from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.models.load_beit3 import BEiT3Encoder
from src.models.load_codetr import CoDETRDetector
from src.models.load_openclip import OpenCLIPH14Encoder
from src.models.load_transnetv2 import TransNetV2ShotModel
from src.offline.codetr_detector import attach_object_counts
from src.offline.embed_keyframes import embed_selected_keyframes
from src.offline.keyframe_selector import SelectedKeyframe, select_keyframes_for_shot
from src.offline.reencode_shots import reencode_upload_shot_proxies
from src.offline.transnet_shot_detector import detect_shots
from src.offline.verify_and_cleanup import delete_raw_video_if_enabled, verify_offline_outputs
from src.schemas import PROCESSING, VERIFIED, VideoRecord
from src.storage.cockroach_client import CockroachClient
from src.storage.r2_client import R2Client
from src.storage.zilliz_client import ZillizClient
from src.utils.config import get_config_value, load_config
from src.utils.ids import make_video_id
from src.utils.logging import get_logger, setup_logging
from src.utils.video_io import get_video_metadata, iter_video_files


logger = get_logger(__name__)


def run_offline_phase(
    config_path: str | None = None,
    video_dir: str | None = None,
    limit: Optional[int] = None,
    fail_fast: Optional[bool] = None,
) -> Dict[str, Any]:
    setup_logging()
    config = load_config(config_path)
    video_root = video_dir or get_config_value(config, "paths.video_dir")
    videos = iter_video_files(video_root)
    if limit is not None:
        videos = videos[:limit]
    logger.info("offline_phase_start videos=%d video_dir=%s", len(videos), video_root)

    cockroach: CockroachClient | None = None
    try:
        zilliz = ZillizClient(config)
        zilliz.create_collection(drop_existing=False)
        cockroach = CockroachClient(config)
        apply_cockroach_schema_migration(cockroach, config_path)
        r2 = R2Client(config)
        beit3 = BEiT3Encoder(config).load()
        openclip = OpenCLIPH14Encoder(config).load()
        transnet = TransNetV2ShotModel(config).load()
        codetr = CoDETRDetector(config).load()

        should_fail_fast = bool(fail_fast) if fail_fast is not None else limit == 1
        processed = 0
        skipped = 0
        failed = 0
        for video_path in videos:
            try:
                outcome = process_single_video(
                    str(video_path),
                    config=config,
                    zilliz=zilliz,
                    cockroach=cockroach,
                    r2=r2,
                    beit3=beit3,
                    openclip=openclip,
                    transnet=transnet,
                    codetr=codetr,
                )
                if outcome == "skipped":
                    skipped += 1
                else:
                    processed += 1
            except Exception:
                failed += 1
                cockroach.rollback()
                logger.exception("offline_video_failed video_path=%s fail_fast=%s", video_path, should_fail_fast)
                if should_fail_fast:
                    raise
        return {"processed": processed, "skipped": skipped, "failed": failed}
    finally:
        if cockroach is not None:
            cockroach.close()


def apply_cockroach_schema_migration(cockroach: CockroachClient, config_path: str | None = None) -> None:
    migration_path = _resolve_migration_path(config_path)
    if migration_path is None:
        logger.warning("cockroach_migration_skip reason=not_found")
        return
    logger.info("cockroach_migration_apply path=%s", migration_path)
    cockroach.execute_sql_file(migration_path)


def _resolve_migration_path(config_path: str | None = None) -> Path | None:
    explicit_path = os.environ.get("COCKROACH_MIGRATION_PATH")
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if config_path:
        config_file = Path(config_path)
        candidates.append(config_file.parent.parent / "migrations" / "001_init_cockroach.sql")
    candidates.append(Path("migrations/001_init_cockroach.sql"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_single_video(
    video_path: str,
    config: Dict[str, Any],
    zilliz: ZillizClient,
    cockroach: CockroachClient,
    r2: R2Client,
    beit3: BEiT3Encoder,
    openclip: OpenCLIPH14Encoder,
    transnet: TransNetV2ShotModel,
    codetr: CoDETRDetector,
) -> str:
    video_id = make_video_id(video_path)
    model_version = config["project"]["model_version"]
    config_version = config["project"]["config_version"]
    if config.get("offline", {}).get("resume", True) and cockroach.is_video_verified(video_id, model_version, config_version):
        logger.info("offline_video_skip_verified video_id=%s", video_id)
        return "skipped"

    logger.info("offline_video_start video_id=%s path=%s", video_id, video_path)
    metadata = get_video_metadata(video_path)
    logger.info(
        "offline_video_metadata video_id=%s frames=%s fps=%.3f duration=%.3f size=%sx%s",
        video_id,
        metadata.get("frame_count"),
        float(metadata.get("fps_raw") or 0.0),
        float(metadata.get("duration_raw") or 0.0),
        metadata.get("width_raw"),
        metadata.get("height_raw"),
    )
    video_record = VideoRecord(
        video_id=video_id,
        source_path=video_path,
        duration_raw=float(metadata["duration_raw"]),
        fps_raw=float(metadata["fps_raw"]),
        width_raw=int(metadata["width_raw"]),
        height_raw=int(metadata["height_raw"]),
        checksum_raw=metadata["checksum_raw"],
        model_version=model_version,
        config_version=config_version,
        processing_status=PROCESSING,
    )
    cockroach.upsert_video(video_record)

    shots = detect_shots(video_path, video_id, metadata, config, model=transnet)
    cockroach.upsert_shots(shots)
    logger.info("offline_shots_detected video_id=%s shots=%d", video_id, len(shots))

    selected: list[SelectedKeyframe] = []
    shots_to_process = []
    for shot in shots:
        if config.get("offline", {}).get("resume", True) and cockroach.is_shot_verified(shot.shot_id, model_version, config_version):
            logger.info("offline_shot_skip_verified shot_id=%s", shot.shot_id)
            continue
        shots_to_process.append(shot)
        selected.extend(select_keyframes_for_shot(video_path, shot, config, beit3))

    if selected:
        logger.info("offline_keyframes_selected video_id=%s keyframes=%d", video_id, len(selected))
        beit3_vectors, openclip_vectors = embed_selected_keyframes(selected, openclip, config)
        logger.info("offline_keyframes_embedded video_id=%s keyframes=%d", video_id, len(selected))
        attach_object_counts(selected, codetr)
        logger.info("offline_objects_detected video_id=%s keyframes=%d", video_id, len(selected))
        for item in selected:
            item.record.zilliz_inserted = True
            item.record.processing_status = VERIFIED
        zilliz.insert_keyframes([item.record for item in selected], beit3_vectors, openclip_vectors)
        logger.info("offline_zilliz_inserted video_id=%s keyframes=%d", video_id, len(selected))
        cockroach.upsert_keyframes([item.record for item in selected])
        logger.info("offline_keyframes_upserted video_id=%s keyframes=%d", video_id, len(selected))
    else:
        logger.warning("offline_no_new_keyframes video_id=%s", video_id)

    if shots_to_process:
        reencode_upload_shot_proxies(video_path, shots_to_process, config, r2, cockroach)
    verify_offline_outputs(selected, shots, r2, allow_no_new_keyframes=not shots_to_process)
    raw_deleted = delete_raw_video_if_enabled(video_path, config)
    cockroach.mark_video_status(video_id, VERIFIED, r2_raw_deleted_at=raw_deleted)
    logger.info("offline_video_verified video_id=%s keyframes=%d raw_deleted=%s", video_id, len(selected), raw_deleted)
    return "processed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_offline_phase(config_path=args.config, video_dir=args.video_dir, limit=args.limit, fail_fast=args.fail_fast), indent=2))
