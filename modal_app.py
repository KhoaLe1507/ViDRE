from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict

try:
    import modal
except ImportError:
    modal = None


class _LocalApp:
    def function(self, *args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator


if modal is not None:
    MODAL_CLOUD = "aws"
    MODAL_REGION = "eu-central-1"
    placement_kwargs = {
        "cloud": MODAL_CLOUD,
        "region": MODAL_REGION,
    }
    debug_image = modal.Image.debian_slim(python_version="3.10")
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
        .run_commands(
            "pip install --upgrade pip setuptools wheel",
            "pip install --extra-index-url https://download.pytorch.org/whl/cu117 torch==1.13.1+cu117 torchvision==0.14.1+cu117",
            "pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html",
        )
        .pip_install_from_requirements("requirements.txt")
        .add_local_dir("src", remote_path="/root/src", copy=True)
        .add_local_dir("configs", remote_path="/root/configs", copy=True)
        .add_local_dir("migrations", remote_path="/root/migrations", copy=True)
        .add_local_dir("external/unilm/beit3", remote_path="/root/external/unilm/beit3", copy=True)
        .add_local_dir("external/TransNetV2/inference", remote_path="/root/external/TransNetV2/inference", copy=True)
        .add_local_dir("external/Co-DETR", remote_path="/root/external/Co-DETR", copy=True)
    )
    app = modal.App("vidre-text-to-keyframe", image=image)
    volume = modal.Volume.from_name("vidre-data", create_if_missing=True)
    function_kwargs = {
        "volumes": {"/data": volume},
        "secrets": [modal.Secret.from_name("vidre-secrets")],
        **placement_kwargs,
    }
else:
    app = _LocalApp()
    function_kwargs = {}
    placement_kwargs = {}
    debug_image = None

temporal_generation_function_kwargs = dict(function_kwargs)
temporal_generation_function_kwargs["cloud"] = "aws"
temporal_generation_function_kwargs["region"] = ["eu-west-3", "eu-west-1", "eu-west-2"]


def _prepare_runtime() -> None:
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    os.environ.setdefault("MODEL_DIR", "/data/models")
    os.environ.setdefault("CONFIG_PATH", "/root/configs/default.yaml")


def _default_config_path(config_path: str | None) -> str | None:
    if config_path:
        return config_path
    if modal is not None:
        return "/root/configs/default.yaml"
    return None



def _region_payload() -> Dict[str, str | None]:
    import os
    import socket

    return {
        "cloud_provider": os.getenv("MODAL_CLOUD_PROVIDER"),
        "region": os.getenv("MODAL_REGION"),
        "environment": os.getenv("MODAL_ENVIRONMENT"),
        "hostname": socket.gethostname(),
    }


@app.function(image=debug_image, **placement_kwargs)
def debug_region():
    return _region_payload()


@app.function(image=debug_image, gpu="L4", cpu=1, memory=4096, timeout=300, **placement_kwargs)
def debug_gpu_region():
    return _region_payload()


@app.local_entrypoint(name="debug_region_cli")
def debug_region_cli():
    print(json.dumps(debug_region.remote(), indent=2))


@app.local_entrypoint(name="debug_gpu_region_cli")
def debug_gpu_region_cli():
    print(json.dumps(debug_gpu_region.remote(), indent=2))


@app.function(cpu=1, memory=1024, timeout=300, **function_kwargs)
def apply_cockroach_migration(config_path: str | None = None, migration_path: str | None = None) -> Dict[str, Any]:
    _prepare_runtime()
    from src.storage.cockroach_client import CockroachClient
    from src.utils.config import load_config

    config_path = _default_config_path(config_path)
    migration_path = migration_path or ("/root/migrations/001_init_cockroach.sql" if modal is not None else "migrations/001_init_cockroach.sql")
    client = CockroachClient(load_config(config_path))
    try:
        client.execute_sql_file(migration_path)
    finally:
        client.close()
    return {"applied": migration_path}


@app.local_entrypoint(name="apply_cockroach_migration_cli")
def apply_cockroach_migration_cli(config_path: str | None = None, migration_path: str | None = None):
    print(json.dumps(apply_cockroach_migration.remote(config_path=config_path, migration_path=migration_path), indent=2))


@app.function(gpu="L4", cpu=8, memory=65536, timeout=86400, **function_kwargs)
def run_offline_phase(
    config_path: str | None = None,
    video_dir: str | None = None,
    limit: int | None = None,
    fail_fast: bool | None = None,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.offline.run_offline_phase import run_offline_phase as _run_offline_phase

    video_dir = video_dir or ("/data/TimeLens-Bench/videos/charades" if modal is not None else None)
    config_path = _default_config_path(config_path)
    return _run_offline_phase(config_path=config_path, video_dir=video_dir, limit=limit, fail_fast=fail_fast)


@app.local_entrypoint(name="run_offline_phase_cli")
def run_offline_phase_cli(
    config_path: str | None = None,
    video_dir: str | None = None,
    limit: int | None = None,
    fail_fast: bool | None = None,
):
    result = run_offline_phase.remote(config_path=config_path, video_dir=video_dir, limit=limit, fail_fast=fail_fast)
    print(json.dumps(result, indent=2))


def _json_default(value: Any) -> str:
    return repr(value)


@app.function(cpu=1, memory=1024, timeout=300, **function_kwargs)
def inspect_storage_state(config_path: str | None = None) -> str:
    _prepare_runtime()
    from src.storage.cockroach_client import CockroachClient
    from src.storage.zilliz_client import ZillizClient
    from src.utils.config import load_config

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    zilliz = ZillizClient(config)
    cockroach = CockroachClient(config)
    try:
        collection_name = zilliz.collection_name
        collections = zilliz.client.list_collections()
        zilliz_stats: Dict[str, Any] = {}
        zilliz_sample = []
        if zilliz.client.has_collection(collection_name):
            try:
                zilliz_stats = dict(zilliz.client.get_collection_stats(collection_name=collection_name))
            except Exception as exc:
                zilliz_stats = {"error": repr(exc)}
            try:
                zilliz_sample = zilliz.client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["keyframe_id", "video_id", "shot_id", "timestamp_raw", "frame_index_raw"],
                    limit=5,
                )
            except Exception as exc:
                zilliz_sample = [{"error": repr(exc)}]

        with cockroach.conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM videos")
            videos = int(cur.fetchone()["n"])
            cur.execute("SELECT count(*) AS n FROM shots")
            shots = int(cur.fetchone()["n"])
            cur.execute("SELECT count(*) AS n FROM keyframes")
            keyframes = int(cur.fetchone()["n"])
            cur.execute(
                """
                SELECT processing_status, count(*) AS n
                FROM videos
                GROUP BY processing_status
                ORDER BY processing_status
                """
            )
            videos_by_status = {str(row["processing_status"]): int(row["n"]) for row in cur.fetchall()}
            cur.execute(
                """
                SELECT processing_status, count(*) AS n
                FROM shots
                GROUP BY processing_status
                ORDER BY processing_status
                """
            )
            shots_by_status = {str(row["processing_status"]): int(row["n"]) for row in cur.fetchall()}
            cur.execute(
                """
                SELECT processing_status, count(*) AS n
                FROM keyframes
                GROUP BY processing_status
                ORDER BY processing_status
                """
            )
            keyframes_by_status = {str(row["processing_status"]): int(row["n"]) for row in cur.fetchall()}
            cur.execute(
                """
                SELECT video_id, processing_status, updated_at
                FROM videos
                ORDER BY updated_at DESC
                LIMIT 10
                """
            )
            recent_videos = [
                {
                    "video_id": str(row["video_id"]),
                    "processing_status": str(row["processing_status"]),
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
                for row in cur.fetchall()
            ]

        payload = {
            "zilliz_collection_from_secret": collection_name,
            "zilliz_collections": collections,
            "zilliz_stats": zilliz_stats,
            "zilliz_sample": zilliz_sample,
            "cockroach_counts": {
                "videos": videos,
                "shots": shots,
                "keyframes": keyframes,
            },
            "cockroach_status_counts": {
                "videos": videos_by_status,
                "shots": shots_by_status,
                "keyframes": keyframes_by_status,
            },
            "recent_videos": recent_videos,
        }
        return json.dumps(payload, indent=2, default=_json_default)
    finally:
        cockroach.close()


@app.local_entrypoint(name="inspect_storage_state_cli")
def inspect_storage_state_cli(config_path: str | None = None):
    print(inspect_storage_state.remote(config_path=config_path))


@app.function(cpu=1, memory=1024, timeout=900, **function_kwargs)
def list_processed_videos(
    config_path: str | None = None,
    status: str = "VERIFIED",
    limit: int | None = None,
    offset: int = 0,
    preview_limit: int = 50,
    output_path: str | None = None,
) -> str:
    _prepare_runtime()
    from datetime import datetime, timezone
    from pathlib import Path

    from src.storage.cockroach_client import CockroachClient
    from src.utils.config import load_config

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    model_version = str(config["project"]["model_version"])
    config_version = str(config["project"]["config_version"])
    requested_status = str(status or "VERIFIED")
    statuses = [item.strip().upper() for item in requested_status.split(",") if item.strip()]
    include_all_statuses = not statuses or "ALL" in statuses

    where_clauses = ["v.model_version = %s", "v.config_version = %s"]
    where_params: list[Any] = [model_version, config_version]
    if not include_all_statuses:
        placeholders = ", ".join(["%s"] * len(statuses))
        where_clauses.append(f"v.processing_status IN ({placeholders})")
        where_params.extend(statuses)
    where_sql = " AND ".join(where_clauses)

    limit_sql = ""
    query_params: list[Any] = [
        model_version,
        config_version,
        model_version,
        config_version,
        *where_params,
    ]
    if limit is not None:
        limit_sql += " LIMIT %s"
        query_params.append(int(limit))
    if offset:
        limit_sql += " OFFSET %s"
        query_params.append(int(offset))

    cockroach = CockroachClient(config)
    try:
        with cockroach.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT count(*) AS n
                FROM videos v
                WHERE {where_sql}
                """,
                tuple(where_params),
            )
            selected_count = int(cur.fetchone()["n"])
            cur.execute(
                """
                SELECT processing_status, count(*) AS n
                FROM videos
                WHERE model_version = %s AND config_version = %s
                GROUP BY processing_status
                ORDER BY processing_status
                """,
                (model_version, config_version),
            )
            status_counts = {str(row["processing_status"]): int(row["n"]) for row in cur.fetchall()}
            cur.execute(
                f"""
                SELECT
                    v.video_id,
                    v.processing_status,
                    v.source_path,
                    v.duration_raw,
                    v.fps_raw,
                    v.updated_at,
                    COALESCE(s.shot_count, 0) AS shot_count,
                    COALESCE(k.keyframe_count, 0) AS keyframe_count
                FROM videos v
                LEFT JOIN (
                    SELECT video_id, count(*) AS shot_count
                    FROM shots
                    WHERE model_version = %s AND config_version = %s
                    GROUP BY video_id
                ) s ON s.video_id = v.video_id
                LEFT JOIN (
                    SELECT video_id, count(*) AS keyframe_count
                    FROM keyframes
                    WHERE model_version = %s AND config_version = %s
                    GROUP BY video_id
                ) k ON k.video_id = v.video_id
                WHERE {where_sql}
                ORDER BY v.video_id
                {limit_sql}
                """,
                tuple(query_params),
            )
            rows = [
                {
                    "video_id": str(row["video_id"]),
                    "processing_status": str(row["processing_status"]),
                    "source_path": str(row["source_path"]),
                    "duration_raw": float(row["duration_raw"]),
                    "fps_raw": float(row["fps_raw"]),
                    "shot_count": int(row["shot_count"]),
                    "keyframe_count": int(row["keyframe_count"]),
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
                for row in cur.fetchall()
            ]

        payload = {
            "status_filter": "ALL" if include_all_statuses else statuses,
            "model_version": model_version,
            "config_version": config_version,
            "selected_count": selected_count,
            "returned_count": len(rows),
            "offset": int(offset),
            "limit": int(limit) if limit is not None else None,
            "status_counts": status_counts,
            "videos": rows,
        }
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output_root = Path("/data/outputs/diagnostics" if modal is not None else "outputs/diagnostics")
            output_path_obj = output_root / f"processed_videos_{timestamp}.json"
        else:
            output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        if modal is not None:
            volume.commit()

        summary = {
            "status_filter": payload["status_filter"],
            "selected_count": selected_count,
            "returned_count": len(rows),
            "status_counts": status_counts,
            "output_path": str(output_path_obj),
            "preview": rows[: max(0, int(preview_limit))],
        }
        return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
    finally:
        cockroach.close()


@app.local_entrypoint(name="list_processed_videos_cli")
def list_processed_videos_cli(
    config_path: str | None = None,
    status: str = "VERIFIED",
    limit: int | None = None,
    offset: int = 0,
    preview_limit: int = 50,
    output_path: str | None = None,
):
    print(
        list_processed_videos.remote(
            config_path=config_path,
            status=status,
            limit=limit,
            offset=offset,
            preview_limit=preview_limit,
            output_path=output_path,
        )
    )


@app.function(cpu=1, memory=1024, timeout=900, **function_kwargs)
def inspect_eval_oracle_coverage(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    require_verified_gt_video: bool = False,
) -> str:
    _prepare_runtime()
    from src.eval.run_evaluation import load_query_samples, load_verified_video_ids
    from src.storage.cockroach_client import CockroachClient
    from src.utils.config import get_config_value, load_config

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    dataset_path = dataset_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples.json"
        if modal is not None
        else get_config_value(config, "paths.dataset_query_samples")
    )
    samples = load_query_samples(dataset_path)
    if require_verified_gt_video:
        verified_video_ids = load_verified_video_ids(config)
        samples = [sample for sample in samples if sample.video_id in verified_video_ids]
    if offset:
        samples = samples[int(offset) :]
    if limit is not None:
        samples = samples[:limit]

    cockroach = CockroachClient(config)
    try:
        oracle_hits = 0
        missing_gt_span = 0
        missing_gt_video = 0
        total_gt_video_keyframes = 0
        total_gt_span_keyframes = 0
        sample_misses = []
        video_status_cache: Dict[str, str | None] = {}
        video_status_counts: Dict[str, int] = {}

        with cockroach.conn.cursor() as cur:
            for sample in samples:
                gt_start, gt_end = float(sample.gt_span[0]), float(sample.gt_span[1])
                if sample.video_id not in video_status_cache:
                    cur.execute(
                        """
                        SELECT processing_status
                        FROM videos
                        WHERE video_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (sample.video_id,),
                    )
                    status_row = cur.fetchone()
                    video_status_cache[sample.video_id] = str(status_row["processing_status"]) if status_row else None
                status = video_status_cache[sample.video_id]
                video_status_counts[str(status)] = video_status_counts.get(str(status), 0) + 1

                cur.execute(
                    """
                    SELECT
                        count(*) AS gt_video_keyframes,
                        sum(CASE WHEN timestamp_raw >= %s AND timestamp_raw <= %s THEN 1 ELSE 0 END) AS gt_span_keyframes,
                        min(least(abs(timestamp_raw - %s), abs(timestamp_raw - %s))) AS nearest_gap_sec
                    FROM keyframes
                    WHERE video_id = %s AND processing_status = 'VERIFIED'
                    """,
                    (gt_start, gt_end, gt_start, gt_end, sample.video_id),
                )
                row = cur.fetchone()
                gt_video_keyframes = int(row["gt_video_keyframes"] or 0)
                gt_span_keyframes = int(row["gt_span_keyframes"] or 0)
                nearest_gap_sec = row["nearest_gap_sec"]
                total_gt_video_keyframes += gt_video_keyframes
                total_gt_span_keyframes += gt_span_keyframes

                if gt_video_keyframes == 0:
                    missing_gt_video += 1
                if gt_span_keyframes > 0:
                    oracle_hits += 1
                else:
                    missing_gt_span += 1
                    if len(sample_misses) < 20:
                        sample_misses.append(
                            {
                                "query_id": sample.query_id,
                                "video_id": sample.video_id,
                                "video_status": status,
                                "gt_span": sample.gt_span,
                                "gt_video_keyframes": gt_video_keyframes,
                                "nearest_gap_sec": float(nearest_gap_sec) if nearest_gap_sec is not None else None,
                                "query_text": sample.query_text,
                            }
                        )

        n = len(samples)
        payload = {
            "num_queries": n,
            "offset": int(offset),
            "limit": limit,
            "require_verified_gt_video": bool(require_verified_gt_video),
            "oracle_hit_queries": oracle_hits,
            "oracle_recall_upper_bound": (oracle_hits / n) if n else 0.0,
            "queries_without_any_keyframe_in_gt_span": missing_gt_span,
            "queries_without_any_keyframe_in_gt_video": missing_gt_video,
            "avg_gt_video_keyframes": (total_gt_video_keyframes / n) if n else 0.0,
            "avg_gt_span_keyframes": (total_gt_span_keyframes / n) if n else 0.0,
            "video_status_counts": video_status_counts,
            "sample_misses": sample_misses,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    finally:
        cockroach.close()


@app.local_entrypoint(name="inspect_eval_oracle_coverage_cli")
def inspect_eval_oracle_coverage_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    require_verified_gt_video: bool = False,
):
    print(
        inspect_eval_oracle_coverage.remote(
            config_path=config_path,
            dataset_path=dataset_path,
            limit=limit,
            offset=offset,
            require_verified_gt_video=require_verified_gt_video,
        )
    )


@app.function(cpu=2, memory=8192, timeout=21600, **function_kwargs)
def generate_scene_moment_queries(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_path: str | None = None,
    limit: int = 100,
    offset: int = 0,
    frames_per_query: int = 4,
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
) -> str:
    _prepare_runtime()
    from pathlib import Path

    from src.eval.run_evaluation import (
        filter_samples_with_gt_span_keyframes,
        load_query_samples,
        load_verified_video_ids,
    )
    from src.utils.config import get_config_value, load_config
    from src.utils.video_io import get_video_metadata, read_frames_by_index

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    dataset_path = dataset_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples.json"
        if modal is not None
        else get_config_value(config, "paths.dataset_query_samples")
    )
    output_path = output_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples-scene-moment-first100.json"
        if modal is not None
        else "outputs/query_sets/charades-timelens-query-samples-scene-moment-first100.json"
    )

    samples = load_query_samples(dataset_path)
    if require_verified_gt_video:
        verified_video_ids = load_verified_video_ids(config)
        samples = [sample for sample in samples if sample.video_id in verified_video_ids]
    if require_gt_span_keyframe:
        samples = filter_samples_with_gt_span_keyframes(config, samples)
    if offset:
        samples = samples[int(offset) :]
    samples = samples[: int(limit)]

    api_key_env = get_config_value(config, "models.gemini.api_key_env", "GEMINI_API_KEY")
    model_env = get_config_value(config, "models.gemini.model_env", "GEMINI_MODEL")
    api_key = os.environ.get(api_key_env, "")
    model_name = os.environ.get(model_env, "")
    if not api_key or not model_name:
        raise RuntimeError(f"{api_key_env} and {model_env} must be set.")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name)

    rows = []
    failures = []
    for index, sample in enumerate(samples, start=1):
        try:
            source_path, checked_paths = _resolve_video_path("", sample.video_id, config)
            if source_path is None:
                raise RuntimeError(f"Cannot resolve video path for {sample.video_id}: {checked_paths}")
            metadata = get_video_metadata(source_path)
            fps = float(metadata["fps_raw"] or 0.0)
            duration = float(metadata["duration_raw"] or sample.duration or 0.0)
            frame_indices = _sample_frame_indices_for_span(
                sample.gt_span,
                fps=fps,
                duration=duration,
                frames_per_query=frames_per_query,
            )
            frames = read_frames_by_index(source_path, frame_indices)
            if not frames:
                raise RuntimeError(f"Cannot read sampled frames for {sample.query_id}: {frame_indices}")
            contact_sheet = _make_candidate_contact_sheet(frames, thumb_size=(256, 192), columns=len(frames))
            prompt = (
                "You are rewriting an action-centric video retrieval query into a scene/moment/event query.\n"
                "Use the provided frames from the ground-truth video moment as visual evidence.\n"
                "Write one natural English query that describes the visible scene, setting, people, clothing, objects, "
                "spatial relations, and the broader moment. It should be useful for text-to-video/keyframe retrieval.\n"
                "Do not mention frame numbers, timestamps, dataset names, or that you are looking at images.\n"
                "Keep it one sentence, specific but not overly long, about 20-35 words.\n"
                "Return valid JSON only in this format: "
                "{\"scene_moment_query\": \"...\", \"visible_cues\": [\"...\", \"...\"]}.\n\n"
                f"Original action query: {sample.query_text}\n"
                f"Video id: {sample.video_id}\n"
                f"Ground-truth span: {sample.gt_span[0]} to {sample.gt_span[1]} seconds\n"
            )
            response = gemini_model.generate_content([prompt, contact_sheet])
            payload = _parse_json_from_text(response.text or "{}")
            scene_query = str(payload.get("scene_moment_query") or "").strip()
            if not scene_query:
                raise RuntimeError(f"Gemini returned empty scene_moment_query for {sample.query_id}")
            rows.append(
                {
                    "query_id": sample.query_id,
                    "video_id": sample.video_id,
                    "query_index": index - 1 + int(offset),
                    "duration": sample.duration,
                    "query": scene_query,
                    "query_text": scene_query,
                    "span": sample.gt_span,
                    "gt_span": sample.gt_span,
                    "original_query": sample.query_text,
                    "query_style": "scene_moment_event",
                    "generation_model": model_name,
                    "sampled_frame_indices": frame_indices,
                    "visible_cues": payload.get("visible_cues") or [],
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "query_id": sample.query_id,
                    "video_id": sample.video_id,
                    "original_query": sample.query_text,
                    "error": repr(exc),
                }
            )

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    diagnostics_path = output_path_obj.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(
        json.dumps(
            {
                "source_dataset_path": str(dataset_path),
                "output_path": str(output_path_obj),
                "requested_limit": int(limit),
                "offset": int(offset),
                "generated_count": len(rows),
                "failure_count": len(failures),
                "failures": failures[:50],
                "require_verified_gt_video": bool(require_verified_gt_video),
                "require_gt_span_keyframe": bool(require_gt_span_keyframe),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    if modal is not None:
        volume.commit()

    summary = {
        "output_path": str(output_path_obj),
        "diagnostics_path": str(diagnostics_path),
        "generated_count": len(rows),
        "failure_count": len(failures),
        "preview": rows[:5],
    }
    return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)


TEMPORAL_QUERY_TYPES: Dict[int, Dict[str, str]] = {
    1: {
        "name": "two_consecutive_scenes",
        "description": "Two different scenes or visual states appear one after the other in the same short moment.",
    },
    2: {
        "name": "multi_scene_sequence",
        "description": "Three or more scenes or events appear in a specific order in the same moment.",
    },
    3: {
        "name": "central_scene_between_context",
        "description": "The target moment is best identified by what happens immediately before and after it.",
    },
    4: {
        "name": "immediate_transition",
        "description": "The second scene appears right after the first with little or no visual gap.",
    },
    5: {
        "name": "simultaneous_events",
        "description": "Two events happen at the same time or overlap in the same time span.",
    },
    6: {
        "name": "interrupted_scene",
        "description": "An ongoing scene is interrupted by a new event that changes the action.",
    },
}


@app.function(cpu=2, memory=2048, timeout=21600, nonpreemptible=True, **temporal_generation_function_kwargs)
def generate_temporal_scene_moment_queries(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_path: str | None = None,
    target_count: int = 1000,
    offset: int = 0,
    max_candidates: int | None = None,
    frames_in_span: int = 6,
    context_frames: int = 2,
    context_window_seconds: float = 3.0,
    checkpoint_every: int = 5,
    gemini_timeout_seconds: int = 120,
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
) -> str:
    _prepare_runtime()
    from collections import Counter
    from pathlib import Path

    from src.eval.run_evaluation import (
        filter_samples_with_gt_span_keyframes,
        load_query_samples,
        load_verified_video_ids,
    )
    from src.utils.config import get_config_value, load_config
    from src.utils.video_io import get_video_metadata, read_frame_map_by_index

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    dataset_path = dataset_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples.json"
        if modal is not None
        else get_config_value(config, "paths.dataset_query_samples")
    )
    output_path = output_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples-temporal-scene-moment-first1000.json"
        if modal is not None
        else "outputs/query_sets/charades-timelens-query-samples-temporal-scene-moment-first1000.json"
    )
    output_path_obj = Path(output_path)
    diagnostics_path = output_path_obj.with_suffix(".diagnostics.json")
    rows = []
    failures = []
    skipped = []
    if output_path_obj.exists():
        loaded_rows = json.loads(output_path_obj.read_text(encoding="utf-8"))
        if not isinstance(loaded_rows, list):
            raise ValueError(f"Expected existing output to contain a JSON list: {output_path_obj}")
        rows = [dict(row) for row in loaded_rows]
    if diagnostics_path.exists():
        diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        failures = list(diagnostics_payload.get("failures") or [])
        skipped = list(diagnostics_payload.get("skipped") or [])

    samples = load_query_samples(dataset_path)
    if require_verified_gt_video:
        verified_video_ids = load_verified_video_ids(config)
        samples = [sample for sample in samples if sample.video_id in verified_video_ids]
    if require_gt_span_keyframe:
        samples = filter_samples_with_gt_span_keyframes(config, samples)
    if offset:
        samples = samples[int(offset) :]
    if max_candidates is not None:
        samples = samples[: int(max_candidates)]

    api_key_env = get_config_value(config, "models.gemini.api_key_env", "GEMINI_API_KEY")
    model_env = get_config_value(config, "models.gemini.model_env", "GEMINI_MODEL")
    api_key = os.environ.get(api_key_env, "")
    model_name = os.environ.get(model_env, "")
    if not api_key or not model_name:
        raise RuntimeError(f"{api_key_env} and {model_env} must be set.")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name)
    print(
        "temporal_generation_gemini_ready "
        f"model={model_name} dataset={dataset_path} output={output_path_obj} "
        f"existing_rows={len(rows)} existing_failures={len(failures)} existing_skipped={len(skipped)} "
        f"samples_after_filter={len(samples)}",
        flush=True,
    )

    target_count = max(1, int(target_count))
    checkpoint_every = max(1, int(checkpoint_every))
    gemini_timeout_seconds = max(10, int(gemini_timeout_seconds))
    type_counts: Counter[str] = Counter()
    for row in rows:
        type_name = str(row.get("temporal_query_type") or "")
        if type_name:
            type_counts[type_name] += 1
    processed_query_ids = {str(row.get("query_id")) for row in rows if row.get("query_id")}
    processed_query_ids.update(str(row.get("query_id")) for row in skipped if row.get("query_id"))
    taxonomy_text = json.dumps(TEMPORAL_QUERY_TYPES, ensure_ascii=False, indent=2)

    def write_progress() -> tuple[Dict[str, int], Dict[str, Any]]:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        type_counts_by_id = {
            str(type_id): int(type_counts[TEMPORAL_QUERY_TYPES[type_id]["name"]])
            for type_id in sorted(TEMPORAL_QUERY_TYPES)
        }
        diagnostics_payload = {
            "source_dataset_path": str(dataset_path),
            "output_path": str(output_path_obj),
            "requested_target_count": target_count,
            "generated_count": len(rows),
            "processed_candidate_count": len(rows) + len(skipped) + len(failures),
            "failure_count": len(failures),
            "skipped_count": len(skipped),
            "temporal_query_types": TEMPORAL_QUERY_TYPES,
            "temporal_type_counts": dict(type_counts),
            "temporal_type_counts_by_id": type_counts_by_id,
            "failures": failures[:200],
            "skipped": skipped[:200],
            "offset": int(offset),
            "max_candidates": max_candidates,
            "frames_in_span": int(frames_in_span),
            "context_frames": int(context_frames),
            "context_window_seconds": float(context_window_seconds),
            "checkpoint_every": checkpoint_every,
            "gemini_timeout_seconds": gemini_timeout_seconds,
            "require_verified_gt_video": bool(require_verified_gt_video),
            "require_gt_span_keyframe": bool(require_gt_span_keyframe),
        }
        diagnostics_path.write_text(
            json.dumps(diagnostics_payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        if modal is not None:
            volume.commit()
        return type_counts_by_id, diagnostics_payload

    if len(rows) >= target_count:
        type_counts_by_id, _ = write_progress()
        summary = {
            "output_path": str(output_path_obj),
            "diagnostics_path": str(diagnostics_path),
            "generated_count": len(rows),
            "failure_count": len(failures),
            "skipped_count": len(skipped),
            "temporal_type_counts_by_id": type_counts_by_id,
            "temporal_type_counts": dict(type_counts),
            "preview": rows[:5],
            "resume_status": "already_complete",
        }
        return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)

    for source_index, sample in enumerate(samples, start=1 + int(offset)):
        if len(rows) >= target_count:
            break
        if sample.query_id in processed_query_ids:
            continue
        try:
            print(
                "temporal_generation_sample_start "
                f"source_index={source_index} generated={len(rows)}/{target_count} "
                f"skipped={len(skipped)} failures={len(failures)} "
                f"query_id={sample.query_id} video_id={sample.video_id}",
                flush=True,
            )
            source_path, checked_paths = _resolve_video_path("", sample.video_id, config)
            if source_path is None:
                raise RuntimeError(f"Cannot resolve video path for {sample.video_id}: {checked_paths}")
            metadata = get_video_metadata(source_path)
            fps = float(metadata["fps_raw"] or 0.0)
            duration = float(metadata["duration_raw"] or sample.duration or 0.0)
            frame_groups = _sample_temporal_context_frame_indices(
                sample.gt_span,
                fps=fps,
                duration=duration,
                frames_in_span=frames_in_span,
                context_frames=context_frames,
                context_window_seconds=context_window_seconds,
            )
            all_frame_indices = frame_groups["all"]
            frame_map = read_frame_map_by_index(source_path, all_frame_indices)
            available_indices = [index for index in all_frame_indices if index in frame_map]
            if len(available_indices) < 2:
                raise RuntimeError(f"Cannot read enough sampled frames for {sample.query_id}: {all_frame_indices}")
            frames = [frame_map[index] for index in available_indices]
            contact_sheet = _make_candidate_contact_sheet(frames, thumb_size=(256, 192), columns=min(len(frames), 5))
            available_groups = {
                group_name: [index for index in indices if index in frame_map]
                for group_name, indices in frame_groups.items()
                if group_name != "all"
            }
            prompt = (
                "You are creating an English benchmark query for temporal-relation video/keyframe retrieval.\n"
                "You are given sampled frames in chronological order from the target ground-truth moment, with a small "
                "amount of before/after context when available.\n\n"
                "Your job:\n"
                "1. Inspect the visible sequence carefully.\n"
                "2. Choose the single most suitable temporal query type from this taxonomy, only if the frames support it.\n"
                "3. Write one natural English query that describes the scene/moment with an explicit temporal relation "
                "such as before, after, then, while, during, immediately after, or interrupted by.\n"
                "4. Keep the query visually grounded, specific, and useful for retrieval, about 25-45 words.\n\n"
                "Do not invent people, objects, places, colors, or events that are not visible or strongly implied.\n"
                "Do not mention frame numbers, timestamps, dataset names, or that you are looking at images.\n"
                "If the sampled moment does not contain a clear temporal relation, return applicable=false.\n\n"
                f"Temporal query taxonomy:\n{taxonomy_text}\n\n"
                "Return valid JSON only in this format:\n"
                "{\n"
                '  "applicable": true,\n'
                '  "temporal_query_type_id": 1,\n'
                '  "temporal_query": "...",\n'
                '  "temporal_relation_summary": "...",\n'
                '  "visible_cues": ["...", "..."]\n'
                "}\n\n"
                f"Original action query: {sample.query_text}\n"
                f"Video id: {sample.video_id}\n"
                f"Ground-truth span: {sample.gt_span[0]} to {sample.gt_span[1]} seconds\n"
                f"Frame groups: {json.dumps(available_groups, ensure_ascii=False)}\n"
                "The contact sheet frames are ordered left-to-right, top-to-bottom in chronological order.\n"
            )
            print(
                "temporal_generation_gemini_request "
                f"query_id={sample.query_id} frames={len(available_indices)} timeout={gemini_timeout_seconds}",
                flush=True,
            )
            response = gemini_model.generate_content(
                [prompt, contact_sheet],
                request_options={"timeout": gemini_timeout_seconds},
            )
            print(f"temporal_generation_gemini_response query_id={sample.query_id}", flush=True)
            payload = _parse_json_from_text(response.text or "{}")
            if not bool(payload.get("applicable", False)):
                skipped.append(
                    {
                        "query_id": sample.query_id,
                        "video_id": sample.video_id,
                        "original_query": sample.query_text,
                        "reason": "no_clear_temporal_relation",
                    }
                )
                processed_query_ids.add(sample.query_id)
                if (len(rows) + len(skipped) + len(failures)) % checkpoint_every == 0:
                    write_progress()
                    print(
                        "temporal_generation_checkpoint "
                        f"generated={len(rows)} skipped={len(skipped)} failures={len(failures)}",
                        flush=True,
                    )
                continue
            type_id = int(payload.get("temporal_query_type_id"))
            if type_id not in TEMPORAL_QUERY_TYPES:
                raise RuntimeError(f"Unsupported temporal_query_type_id={type_id!r} for {sample.query_id}")
            temporal_query = str(payload.get("temporal_query") or "").strip()
            if not temporal_query:
                raise RuntimeError(f"Gemini returned empty temporal_query for {sample.query_id}")
            type_name = TEMPORAL_QUERY_TYPES[type_id]["name"]
            rows.append(
                {
                    "query_id": sample.query_id,
                    "video_id": sample.video_id,
                    "query_index": len(rows),
                    "source_query_index": source_index - 1,
                    "duration": sample.duration,
                    "query": temporal_query,
                    "query_text": temporal_query,
                    "span": sample.gt_span,
                    "gt_span": sample.gt_span,
                    "original_query": sample.query_text,
                    "query_style": "temporal_scene_moment_event",
                    "temporal_query_type_id": type_id,
                    "temporal_query_type": type_name,
                    "temporal_relation_summary": str(payload.get("temporal_relation_summary") or "").strip(),
                    "generation_model": model_name,
                    "sampled_frame_indices": available_indices,
                    "sampled_frame_groups": available_groups,
                    "visible_cues": payload.get("visible_cues") or [],
                }
            )
            type_counts[type_name] += 1
            processed_query_ids.add(sample.query_id)
            print(
                "temporal_generation_sample_done "
                f"query_id={sample.query_id} type_id={type_id} type={type_name} generated={len(rows)}",
                flush=True,
            )
            if (len(rows) + len(skipped) + len(failures)) % checkpoint_every == 0:
                write_progress()
                print(
                    "temporal_generation_checkpoint "
                    f"generated={len(rows)} skipped={len(skipped)} failures={len(failures)}",
                    flush=True,
                )
        except Exception as exc:
            failures.append(
                {
                    "query_id": sample.query_id,
                    "video_id": sample.video_id,
                    "original_query": sample.query_text,
                    "error": repr(exc),
                }
            )
            processed_query_ids.add(sample.query_id)
            print(
                "temporal_generation_sample_failed "
                f"query_id={sample.query_id} error={repr(exc)} generated={len(rows)} "
                f"skipped={len(skipped)} failures={len(failures)}",
                flush=True,
            )
            if (len(rows) + len(skipped) + len(failures)) % checkpoint_every == 0:
                write_progress()
                print(
                    "temporal_generation_checkpoint "
                    f"generated={len(rows)} skipped={len(skipped)} failures={len(failures)}",
                    flush=True,
                )

    type_counts_by_id, _ = write_progress()

    summary = {
        "output_path": str(output_path_obj),
        "diagnostics_path": str(diagnostics_path),
        "generated_count": len(rows),
        "failure_count": len(failures),
        "skipped_count": len(skipped),
        "temporal_type_counts_by_id": type_counts_by_id,
        "temporal_type_counts": dict(type_counts),
        "preview": rows[:5],
    }
    return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)


def _sample_temporal_context_frame_indices(
    gt_span: list[float],
    fps: float,
    duration: float,
    frames_in_span: int,
    context_frames: int,
    context_window_seconds: float,
) -> Dict[str, list[int]]:
    if fps <= 0.0:
        return {"before": [], "in_span": [], "after": [], "all": []}
    start = max(0.0, float(gt_span[0]))
    end = min(float(duration), max(start, float(gt_span[1])))
    if end <= start:
        end = min(float(duration), start + 1.0)
    context_window_seconds = max(0.0, float(context_window_seconds))
    before_start = max(0.0, start - context_window_seconds)
    after_end = min(float(duration), end + context_window_seconds)
    before = _sample_frame_indices_for_interval(before_start, start, fps, duration, context_frames, include_end=False)
    in_span = _sample_frame_indices_for_interval(start, end, fps, duration, frames_in_span, include_end=True)
    after = _sample_frame_indices_for_interval(end, after_end, fps, duration, context_frames, include_end=True)
    after = [index for index in after if index not in set(in_span)]
    all_indices = sorted(dict.fromkeys(before + in_span + after))
    return {"before": before, "in_span": in_span, "after": after, "all": all_indices}


def _sample_frame_indices_for_interval(
    start: float,
    end: float,
    fps: float,
    duration: float,
    count: int,
    include_end: bool,
) -> list[int]:
    count = max(0, int(count))
    if count == 0 or fps <= 0.0 or duration <= 0.0:
        return []
    start = max(0.0, min(float(duration), float(start)))
    end = max(0.0, min(float(duration), float(end)))
    if end <= start:
        return []
    if count == 1:
        times = [(start + end) / 2.0]
    else:
        denominator = float(count - 1) if include_end else float(count)
        times = [start + ((end - start) * position / denominator) for position in range(count)]
    max_frame_index = max(0, int(round(float(duration) * fps)) - 1)
    return sorted({max(0, min(max_frame_index, int(round(time_sec * fps)))) for time_sec in times})


def _sample_frame_indices_for_span(
    gt_span: list[float],
    fps: float,
    duration: float,
    frames_per_query: int,
) -> list[int]:
    if fps <= 0.0:
        return []
    frames_per_query = max(1, int(frames_per_query))
    start = max(0.0, float(gt_span[0]))
    end = min(float(duration), max(start, float(gt_span[1])))
    if end <= start:
        end = min(float(duration), start + 1.0)
    if end <= start:
        end = max(start, float(duration))
    if frames_per_query == 1:
        times = [(start + end) / 2.0]
    else:
        times = [
            start + ((end - start) * position / float(frames_per_query - 1))
            for position in range(frames_per_query)
        ]
    max_frame_index = max(0, int(round(float(duration) * fps)) - 1)
    indices = sorted({max(0, min(max_frame_index, int(round(time_sec * fps)))) for time_sec in times})
    return indices


@app.local_entrypoint(name="generate_scene_moment_queries_cli")
def generate_scene_moment_queries_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_path: str | None = None,
    limit: int = 100,
    offset: int = 0,
    frames_per_query: int = 4,
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
):
    print(
        generate_scene_moment_queries.remote(
            config_path=config_path,
            dataset_path=dataset_path,
            output_path=output_path,
            limit=limit,
            offset=offset,
            frames_per_query=frames_per_query,
            require_verified_gt_video=require_verified_gt_video,
            require_gt_span_keyframe=require_gt_span_keyframe,
        )
    )


@app.local_entrypoint(name="generate_temporal_scene_moment_queries_cli")
def generate_temporal_scene_moment_queries_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_path: str | None = None,
    target_count: int = 1000,
    offset: int = 0,
    max_candidates: int | None = None,
    frames_in_span: int = 6,
    context_frames: int = 2,
    context_window_seconds: float = 3.0,
    checkpoint_every: int = 5,
    gemini_timeout_seconds: int = 120,
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
):
    print(
        generate_temporal_scene_moment_queries.remote(
            config_path=config_path,
            dataset_path=dataset_path,
            output_path=output_path,
            target_count=target_count,
            offset=offset,
            max_candidates=max_candidates,
            frames_in_span=frames_in_span,
            context_frames=context_frames,
            context_window_seconds=context_window_seconds,
            checkpoint_every=checkpoint_every,
            gemini_timeout_seconds=gemini_timeout_seconds,
            require_verified_gt_video=require_verified_gt_video,
            require_gt_span_keyframe=require_gt_span_keyframe,
        )
    )


def _manual_scene_moment_query_rows() -> list[Dict[str, Any]]:
    return [
        {
            "query_id": "N0NLE_001",
            "video_id": "N0NLE",
            "query_index": 1,
            "duration": 29.529500000000002,
            "query": (
                "In a carpeted hallway outside a bedroom, a blonde girl in a black-and-white striped shirt "
                "and pink shorts looks down at a phone near an open doorway and a crutch."
            ),
            "query_text": (
                "In a carpeted hallway outside a bedroom, a blonde girl in a black-and-white striped shirt "
                "and pink shorts looks down at a phone near an open doorway and a crutch."
            ),
            "span": [0.0, 24.0],
            "gt_span": [0.0, 24.0],
            "original_query": "A person is playing with a mobile phone.",
            "query_style": "scene_moment_event",
            "generation_model": "manual_video_inspection",
            "sampled_frame_indices": [0, 240, 480, 719],
            "visible_cues": [
                "Blonde girl holding a mobile phone",
                "Black-and-white striped shirt with a pink heart",
                "Pink shorts",
                "Carpeted hallway and open bedroom doorway",
                "Crutch leaning near the wall",
            ],
        },
        {
            "query_id": "FRSBQ_000",
            "video_id": "FRSBQ",
            "query_index": 0,
            "duration": 32.0,
            "query": (
                "In a purple-walled bedroom with red curtains and a patterned bed, a young person in red pants "
                "stands beside the bed while holding a striped shirt."
            ),
            "query_text": (
                "In a purple-walled bedroom with red curtains and a patterned bed, a young person in red pants "
                "stands beside the bed while holding a striped shirt."
            ),
            "span": [21.0, 32.0],
            "gt_span": [21.0, 32.0],
            "original_query": "A person is taking off clothes.",
            "query_style": "scene_moment_event",
            "generation_model": "manual_video_inspection",
            "sampled_frame_indices": [630, 740, 850, 960],
            "visible_cues": [
                "Purple bedroom wall",
                "Red curtains with a window behind them",
                "Patterned bed with blankets",
                "Young person wearing red pants",
                "Striped shirt being held near the bed",
            ],
        },
        {
            "query_id": "PCNUP_005",
            "video_id": "PCNUP",
            "query_index": 5,
            "duration": 23.6,
            "query": (
                "Inside a small closet filled with hanging clothes and items on the floor, a girl in a bright pink "
                "shirt stands near the doorway while handling clothing."
            ),
            "query_text": (
                "Inside a small closet filled with hanging clothes and items on the floor, a girl in a bright pink "
                "shirt stands near the doorway while handling clothing."
            ),
            "span": [10.0, 12.0],
            "gt_span": [10.0, 12.0],
            "original_query": "A person takes clothes off the hanger.",
            "query_style": "scene_moment_event",
            "generation_model": "manual_video_inspection",
            "sampled_frame_indices": [300, 320, 340, 360],
            "visible_cues": [
                "Small closet space",
                "Many hanging clothes on the left",
                "Girl wearing a bright pink shirt and jeans",
                "Cluttered closet floor",
                "Doorway and pink wall nearby",
            ],
        },
    ]


@app.function(cpu=1, memory=2048, timeout=900, **function_kwargs)
def patch_scene_moment_first1000_manual_queries(
    config_path: str | None = None,
    dataset_path: str | None = None,
    scene_query_path: str = "/data/TimeLens-Bench/charades-timelens-query-samples-scene-moment-first1000.json",
    diagnostics_path: str = "/data/TimeLens-Bench/charades-timelens-query-samples-scene-moment-first1000.diagnostics.json",
) -> str:
    _prepare_runtime()
    from pathlib import Path

    from src.eval.run_evaluation import (
        filter_samples_with_gt_span_keyframes,
        load_query_samples,
        load_verified_video_ids,
    )
    from src.utils.config import get_config_value, load_config

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    dataset_path = dataset_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples.json"
        if modal is not None
        else get_config_value(config, "paths.dataset_query_samples")
    )

    scene_path = Path(scene_query_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene query file not found: {scene_path}")
    rows = json.loads(scene_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected scene query file to contain a JSON list: {scene_path}")

    manual_rows = _manual_scene_moment_query_rows()
    rows_by_id = {str(row["query_id"]): dict(row) for row in rows}
    inserted_or_updated = []
    for row in manual_rows:
        rows_by_id[str(row["query_id"])] = dict(row)
        inserted_or_updated.append(str(row["query_id"]))

    samples = load_query_samples(dataset_path)
    verified_video_ids = load_verified_video_ids(config)
    samples = [sample for sample in samples if sample.video_id in verified_video_ids]
    samples = filter_samples_with_gt_span_keyframes(config, samples)
    ordered_query_ids = [sample.query_id for sample in samples[:1000]]
    ordered_rows = []
    missing_after_patch = []
    for query_index, query_id in enumerate(ordered_query_ids):
        row = rows_by_id.get(query_id)
        if row is None:
            missing_after_patch.append(query_id)
            continue
        row["query_index"] = query_index
        ordered_rows.append(row)

    ordered_set = set(ordered_query_ids)
    extras = [row for query_id, row in rows_by_id.items() if query_id not in ordered_set]
    ordered_rows.extend(extras)
    scene_path.write_text(json.dumps(ordered_rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    diagnostics_payload: Dict[str, Any] = {}
    diagnostics_path_obj = Path(diagnostics_path)
    if diagnostics_path_obj.exists():
        diagnostics_payload = json.loads(diagnostics_path_obj.read_text(encoding="utf-8"))
    original_failures = list(diagnostics_payload.get("failures") or [])
    fixed_ids = set(inserted_or_updated)
    remaining_failures = [failure for failure in original_failures if str(failure.get("query_id")) not in fixed_ids]
    diagnostics_payload.update(
        {
            "output_path": str(scene_path),
            "generated_count": len(ordered_rows),
            "failure_count": len(remaining_failures),
            "failures": remaining_failures,
            "manual_fix_count": len(inserted_or_updated),
            "manual_fixed_query_ids": inserted_or_updated,
            "missing_after_patch": missing_after_patch,
        }
    )
    diagnostics_path_obj.write_text(
        json.dumps(diagnostics_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    if modal is not None:
        volume.commit()

    summary = {
        "scene_query_path": str(scene_path),
        "diagnostics_path": str(diagnostics_path_obj),
        "inserted_or_updated": inserted_or_updated,
        "total_rows": len(ordered_rows),
        "remaining_failure_count": len(remaining_failures),
        "missing_after_patch": missing_after_patch,
    }
    return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)


@app.local_entrypoint(name="patch_scene_moment_first1000_manual_queries_cli")
def patch_scene_moment_first1000_manual_queries_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    scene_query_path: str = "/data/TimeLens-Bench/charades-timelens-query-samples-scene-moment-first1000.json",
    diagnostics_path: str = "/data/TimeLens-Bench/charades-timelens-query-samples-scene-moment-first1000.diagnostics.json",
):
    print(
        patch_scene_moment_first1000_manual_queries.remote(
            config_path=config_path,
            dataset_path=dataset_path,
            scene_query_path=scene_query_path,
            diagnostics_path=diagnostics_path,
        )
    )


def _zilliz_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _first_rank(candidates: list[Any], predicate) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if predicate(candidate):
            return rank
    return None


def _first_unique_video_rank(candidates: list[Any], video_id: str) -> int | None:
    seen = set()
    unique_rank = 0
    for candidate in candidates:
        if candidate.video_id in seen:
            continue
        seen.add(candidate.video_id)
        unique_rank += 1
        if candidate.video_id == video_id:
            return unique_rank
    return None


def _candidate_preview(candidates: list[Any], limit: int = 5) -> list[Dict[str, Any]]:
    return [
        {
            "rank": int(candidate.rank),
            "keyframe_id": candidate.keyframe_id,
            "video_id": candidate.video_id,
            "timestamp_raw": float(candidate.timestamp_raw),
            "score": float(candidate.score),
        }
        for candidate in candidates[:limit]
    ]


def _make_candidate_contact_sheet(candidate_images: list[Any], thumb_size: tuple[int, int] = (256, 192), columns: int = 5) -> Any:
    from PIL import Image, ImageDraw

    columns = max(1, int(columns))
    rows = max(1, (len(candidate_images) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(candidate_images, start=1):
        tile = image.convert("RGB").resize(thumb_size)
        x = ((index - 1) % columns) * thumb_size[0]
        y = ((index - 1) // columns) * thumb_size[1]
        sheet.paste(tile, (x, y))
        label = f"{index}"
        draw.rectangle((x + 6, y + 6, x + 42, y + 36), fill="black")
        draw.text((x + 16, y + 12), label, fill="white")
    return sheet


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    import json
    import re

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _candidate_to_eval_dict(candidate: Any, rank: int) -> Dict[str, Any]:
    return {
        "rank": int(rank),
        "keyframe_id": candidate.keyframe_id,
        "video_id": candidate.video_id,
        "shot_id": candidate.shot_id,
        "timestamp_raw": float(candidate.timestamp_raw),
        "frame_index_raw": int(candidate.frame_index_raw),
        "score": float(candidate.score),
    }


def _ranked_videos(video_scores: Dict[str, float]) -> list[Dict[str, Any]]:
    return [
        {"video_id": video_id, "score": float(score), "rank": rank}
        for rank, (video_id, score) in enumerate(
            sorted(video_scores.items(), key=lambda item: (-float(item[1]), item[0])),
            start=1,
        )
    ]


def _rank_of_video(ranked_videos: list[Dict[str, Any]], video_id: str) -> int | None:
    for item in ranked_videos:
        if item["video_id"] == video_id:
            return int(item["rank"])
    return None


def _video_score_strategies(results_by_model: Dict[str, list[Any]], rrf_k: int = 60) -> Dict[str, list[Dict[str, Any]]]:
    from collections import defaultdict

    single_model_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, candidates in results_by_model.items():
        per_video = defaultdict(list)
        for rank, candidate in enumerate(candidates, start=1):
            per_video[candidate.video_id].append((rank, float(candidate.score)))

        strategies = {
            f"{model_name}_first_rrf": {},
            f"{model_name}_max_score": {},
            f"{model_name}_mean_top3_score": {},
            f"{model_name}_sum_top3_score": {},
            f"{model_name}_rrf_sum_top3_keyframes": {},
        }
        for video_id, rank_scores in per_video.items():
            ranks = sorted(rank for rank, _ in rank_scores)
            scores = sorted((score for _, score in rank_scores), reverse=True)
            top3_scores = scores[:3]
            top3_ranks = ranks[:3]
            strategies[f"{model_name}_first_rrf"][video_id] = 1.0 / (rrf_k + ranks[0])
            strategies[f"{model_name}_max_score"][video_id] = top3_scores[0]
            strategies[f"{model_name}_mean_top3_score"][video_id] = sum(top3_scores) / len(top3_scores)
            strategies[f"{model_name}_sum_top3_score"][video_id] = sum(top3_scores)
            strategies[f"{model_name}_rrf_sum_top3_keyframes"][video_id] = sum(1.0 / (rrf_k + rank) for rank in top3_ranks)
        single_model_scores[model_name] = strategies

    all_scores: Dict[str, Dict[str, float]] = {}
    for strategies in single_model_scores.values():
        all_scores.update(strategies)

    combined_names = [
        "first_rrf",
        "max_score",
        "mean_top3_score",
        "sum_top3_score",
        "rrf_sum_top3_keyframes",
    ]
    for combined_name in combined_names:
        combined_scores: Dict[str, float] = {}
        for model_name, strategies in single_model_scores.items():
            scores = strategies.get(f"{model_name}_{combined_name}", {})
            for video_id, score in scores.items():
                combined_scores[video_id] = combined_scores.get(video_id, 0.0) + float(score)
        all_scores[f"combined_{combined_name}"] = combined_scores

    return {name: _ranked_videos(scores) for name, scores in all_scores.items()}


def _resolve_video_path(source_path: str, video_id: str, config: Dict[str, Any]) -> tuple[Any | None, list[Dict[str, Any]]]:
    from pathlib import Path

    from src.utils.config import get_config_value

    configured_video_dir = Path(get_config_value(config, "paths.video_dir", "/data/TimeLens-Bench/videos/charades"))
    candidates = [
        Path(source_path),
        configured_video_dir / f"{video_id}.mp4",
        Path("/data/TimeLens-Bench/videos/charades") / f"{video_id}.mp4",
        Path("/data/data/TimeLens-Bench/videos/charades") / f"{video_id}.mp4",
    ]

    checked = []
    seen = set()
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        exists = path.exists()
        checked.append(
            {
                "path": path_str,
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists and path.is_file() else None,
            }
        )
        if exists and path.is_file():
            return path, checked
    return None, checked


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def diagnose_retrieval_failure_modes(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = 100,
    offset: int = 0,
    model_space: str = "openclip",
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
    global_depth: int = 500,
    in_video_depth: int = 500,
    output_path: str | None = None,
) -> str:
    _prepare_runtime()
    from datetime import datetime, timezone
    from pathlib import Path

    from src.eval.metrics import compute_hits, is_correct_keyframe
    from src.eval.run_evaluation import (
        filter_samples_with_gt_span_keyframes,
        load_query_samples,
        load_verified_video_ids,
    )
    from src.models.load_beit3 import BEiT3Encoder
    from src.models.load_openclip import OpenCLIPH14Encoder
    from src.storage.zilliz_client import ZillizClient
    from src.utils.config import get_config_value, load_config

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    dataset_path = dataset_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples.json"
        if modal is not None
        else get_config_value(config, "paths.dataset_query_samples")
    )
    samples = load_query_samples(dataset_path)
    if require_verified_gt_video:
        verified_video_ids = load_verified_video_ids(config)
        samples = [sample for sample in samples if sample.video_id in verified_video_ids]
    if require_gt_span_keyframe:
        samples = filter_samples_with_gt_span_keyframes(config, samples)
    if offset:
        samples = samples[int(offset) :]
    if limit is not None:
        samples = samples[:limit]

    zilliz = ZillizClient(config)
    if model_space == "openclip":
        encoder = OpenCLIPH14Encoder(config).load()
        vector_field = zilliz.openclip_field
    elif model_space == "beit3":
        encoder = BEiT3Encoder(config).load()
        vector_field = zilliz.beit3_field
    else:
        raise ValueError("model_space must be 'openclip' or 'beit3'.")

    model_version = config["project"]["model_version"]
    config_version = config["project"]["config_version"]
    rows = []
    global_first_gt_video_ranks = []
    global_first_gt_video_unique_ranks = []
    global_first_correct_ranks = []
    in_video_first_correct_ranks = []
    sample_failures = []

    for sample in samples:
        vector = encoder.encode_texts([sample.query_text])[0]
        global_results = zilliz.search(vector_field, vector, int(global_depth))
        filter_expr = (
            f"video_id == {_zilliz_string(sample.video_id)} "
            f"and model_version == {_zilliz_string(model_version)} "
            f"and config_version == {_zilliz_string(config_version)}"
        )
        in_video_results = zilliz.search(vector_field, vector, int(in_video_depth), filter_expr=filter_expr)

        gt_start, gt_end = float(sample.gt_span[0]), float(sample.gt_span[1])
        global_hits = compute_hits(global_results, sample.video_id, sample.gt_span)
        in_video_hits = compute_hits(in_video_results, sample.video_id, sample.gt_span)
        first_gt_video_rank = _first_rank(global_results, lambda candidate: candidate.video_id == sample.video_id)
        first_correct_global_rank = _first_rank(
            global_results,
            lambda candidate: is_correct_keyframe(candidate, sample.video_id, gt_start, gt_end),
        )
        first_correct_in_video_rank = _first_rank(
            in_video_results,
            lambda candidate: is_correct_keyframe(candidate, sample.video_id, gt_start, gt_end),
        )

        if first_gt_video_rank is not None:
            global_first_gt_video_ranks.append(first_gt_video_rank)
        first_gt_video_unique_rank = _first_unique_video_rank(global_results, sample.video_id)
        if first_gt_video_unique_rank is not None:
            global_first_gt_video_unique_ranks.append(first_gt_video_unique_rank)
        if first_correct_global_rank is not None:
            global_first_correct_ranks.append(first_correct_global_rank)
        if first_correct_in_video_rank is not None:
            in_video_first_correct_ranks.append(first_correct_in_video_rank)

        row = {
            "query_id": sample.query_id,
            "query_text": sample.query_text,
            "video_id": sample.video_id,
            "gt_span": sample.gt_span,
            "global": global_hits,
            "in_video": {
                "hit_at_1": in_video_hits["hit_at_1"],
                "hit_at_5": in_video_hits["hit_at_5"],
                "hit_at_10": in_video_hits["hit_at_10"],
            },
            "first_gt_video_rank_global": first_gt_video_rank,
            "first_gt_video_unique_rank_global": first_gt_video_unique_rank,
            "first_correct_rank_global": first_correct_global_rank,
            "first_correct_rank_in_video": first_correct_in_video_rank,
        }
        rows.append(row)

        if len(sample_failures) < 20 and (
            not global_hits["video_hit_at_10"] or not global_hits["hit_at_10"] or not in_video_hits["hit_at_10"]
        ):
            sample_failures.append(
                {
                    **row,
                    "global_top": _candidate_preview(global_results, limit=5),
                    "in_video_top": _candidate_preview(in_video_results, limit=10),
                }
            )

    n = len(rows)

    def ratio(key: str, scope: str = "global") -> float:
        return (sum(1 for row in rows if row[scope].get(key)) / n) if n else 0.0

    def avg(values: list[int]) -> float | None:
        return (sum(values) / len(values)) if values else None

    def unique_video_recall_at(k: int) -> float:
        if not n:
            return 0.0
        return sum(
            1
            for row in rows
            if row["first_gt_video_unique_rank_global"] is not None
            and int(row["first_gt_video_unique_rank_global"]) <= k
        ) / n

    def estimated_block_recall_at(k: int, block_size: int) -> float:
        if not n:
            return 0.0
        hits = 0
        for row in rows:
            unique_rank = row["first_gt_video_unique_rank_global"]
            in_video_rank = row["first_correct_rank_in_video"]
            if unique_rank is None or in_video_rank is None:
                continue
            if int(in_video_rank) > block_size:
                continue
            estimated_position = (int(unique_rank) - 1) * block_size + int(in_video_rank)
            if estimated_position <= k:
                hits += 1
        return hits / n

    two_stage_estimates = {}
    for block_size in [1, 2, 3, 5]:
        two_stage_estimates[f"block_{block_size}_per_video"] = {
            "estimated_recall_at_1": estimated_block_recall_at(1, block_size),
            "estimated_recall_at_5": estimated_block_recall_at(5, block_size),
            "estimated_recall_at_10": estimated_block_recall_at(10, block_size),
        }

    summary = {
        "num_queries": n,
        "offset": int(offset),
        "limit": limit,
        "model_space": model_space,
        "vector_field": vector_field,
        "require_verified_gt_video": bool(require_verified_gt_video),
        "require_gt_span_keyframe": bool(require_gt_span_keyframe),
        "global_depth": int(global_depth),
        "in_video_depth": int(in_video_depth),
        "global_recall_at_1": ratio("hit_at_1"),
        "global_recall_at_5": ratio("hit_at_5"),
        "global_recall_at_10": ratio("hit_at_10"),
        "global_video_recall_at_1": ratio("video_hit_at_1"),
        "global_video_recall_at_5": ratio("video_hit_at_5"),
        "global_video_recall_at_10": ratio("video_hit_at_10"),
        "global_unique_video_recall_at_1": unique_video_recall_at(1),
        "global_unique_video_recall_at_5": unique_video_recall_at(5),
        "global_unique_video_recall_at_10": unique_video_recall_at(10),
        "in_video_temporal_recall_at_1": ratio("hit_at_1", scope="in_video"),
        "in_video_temporal_recall_at_5": ratio("hit_at_5", scope="in_video"),
        "in_video_temporal_recall_at_10": ratio("hit_at_10", scope="in_video"),
        "found_gt_video_within_global_depth": (len(global_first_gt_video_ranks) / n) if n else 0.0,
        "found_correct_keyframe_within_global_depth": (len(global_first_correct_ranks) / n) if n else 0.0,
        "found_correct_keyframe_within_in_video_depth": (len(in_video_first_correct_ranks) / n) if n else 0.0,
        "avg_first_gt_video_rank_global": avg(global_first_gt_video_ranks),
        "avg_first_gt_video_unique_rank_global": avg(global_first_gt_video_unique_ranks),
        "avg_first_correct_rank_global": avg(global_first_correct_ranks),
        "avg_first_correct_rank_in_video": avg(in_video_first_correct_ranks),
        "estimated_two_stage_block_rerank": two_stage_estimates,
    }
    payload = {
        **summary,
        "sample_failures": sample_failures,
        "per_query": rows,
    }

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_root = Path("/data/outputs/diagnostics" if modal is not None else "outputs/diagnostics")
        output_path_obj = output_root / f"retrieval_failure_modes_{model_space}_{timestamp}.json"
    else:
        output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    if modal is not None:
        volume.commit()

    summary["output_path"] = str(output_path_obj)
    summary["sample_failures_count"] = len(sample_failures)
    summary["per_query_count"] = len(rows)
    return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)


@app.local_entrypoint(name="diagnose_retrieval_failure_modes_cli")
def diagnose_retrieval_failure_modes_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = 100,
    offset: int = 0,
    model_space: str = "openclip",
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
    global_depth: int = 500,
    in_video_depth: int = 500,
    output_path: str | None = None,
):
    print(
        diagnose_retrieval_failure_modes.remote(
            config_path=config_path,
            dataset_path=dataset_path,
            limit=limit,
            offset=offset,
            model_space=model_space,
            require_verified_gt_video=require_verified_gt_video,
            require_gt_span_keyframe=require_gt_span_keyframe,
            global_depth=global_depth,
            in_video_depth=in_video_depth,
            output_path=output_path,
        )
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def diagnose_video_aggregation_strategies(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = 100,
    offset: int = 0,
    model_spaces: str = "openclip,beit3",
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
    global_depth: int = 1000,
    output_path: str | None = None,
) -> str:
    _prepare_runtime()
    from datetime import datetime, timezone
    from pathlib import Path

    from src.eval.run_evaluation import (
        filter_samples_with_gt_span_keyframes,
        load_query_samples,
        load_verified_video_ids,
    )
    from src.models.load_beit3 import BEiT3Encoder
    from src.models.load_openclip import OpenCLIPH14Encoder
    from src.storage.zilliz_client import ZillizClient
    from src.utils.config import get_config_value, load_config

    selected_models = [item.strip() for item in str(model_spaces).split(",") if item.strip()]
    if not selected_models:
        raise ValueError("model_spaces must contain at least one model name.")

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    dataset_path = dataset_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples.json"
        if modal is not None
        else get_config_value(config, "paths.dataset_query_samples")
    )
    samples = load_query_samples(dataset_path)
    if require_verified_gt_video:
        verified_video_ids = load_verified_video_ids(config)
        samples = [sample for sample in samples if sample.video_id in verified_video_ids]
    if require_gt_span_keyframe:
        samples = filter_samples_with_gt_span_keyframes(config, samples)
    if offset:
        samples = samples[int(offset) :]
    if limit is not None:
        samples = samples[:limit]

    zilliz = ZillizClient(config)
    effective_global_depth = min(int(global_depth), int(get_config_value(config, "zilliz.max_search_top_k", 1024)))
    encoders: Dict[str, Any] = {}
    vector_fields: Dict[str, str] = {}
    for model_name in selected_models:
        if model_name == "openclip":
            encoders[model_name] = OpenCLIPH14Encoder(config).load()
            vector_fields[model_name] = zilliz.openclip_field
        elif model_name == "beit3":
            encoders[model_name] = BEiT3Encoder(config).load()
            vector_fields[model_name] = zilliz.beit3_field
        else:
            raise ValueError(f"Unsupported model_space={model_name!r}. Expected openclip or beit3.")

    per_query = []
    strategy_ranks: Dict[str, list[int]] = {}
    strategy_missing: Dict[str, int] = {}

    for sample in samples:
        results_by_model: Dict[str, list[Any]] = {}
        for model_name in selected_models:
            vector = encoders[model_name].encode_texts([sample.query_text])[0]
            results_by_model[model_name] = zilliz.search(vector_fields[model_name], vector, effective_global_depth)

        ranked_by_strategy = _video_score_strategies(results_by_model)
        query_ranks: Dict[str, int | None] = {}
        query_top_videos: Dict[str, list[Dict[str, Any]]] = {}
        for strategy_name, ranked_videos in ranked_by_strategy.items():
            rank = _rank_of_video(ranked_videos, sample.video_id)
            query_ranks[strategy_name] = rank
            if rank is None:
                strategy_missing[strategy_name] = strategy_missing.get(strategy_name, 0) + 1
            else:
                strategy_ranks.setdefault(strategy_name, []).append(rank)
            query_top_videos[strategy_name] = ranked_videos[:10]

        per_query.append(
            {
                "query_id": sample.query_id,
                "query_text": sample.query_text,
                "video_id": sample.video_id,
                "gt_span": sample.gt_span,
                "strategy_video_ranks": query_ranks,
                "top_videos": query_top_videos,
            }
        )

    n = len(per_query)
    strategy_summary = []
    strategy_names = sorted(set(strategy_ranks) | set(strategy_missing))
    for strategy_name in strategy_names:
        ranks = strategy_ranks.get(strategy_name, [])

        def recall_at(k: int) -> float:
            return (sum(1 for rank in ranks if rank <= k) / n) if n else 0.0

        strategy_summary.append(
            {
                "strategy": strategy_name,
                "video_recall_at_1": recall_at(1),
                "video_recall_at_5": recall_at(5),
                "video_recall_at_10": recall_at(10),
                "video_recall_at_20": recall_at(20),
                "video_recall_at_50": recall_at(50),
                "video_recall_at_100": recall_at(100),
                "found_within_depth": (len(ranks) / n) if n else 0.0,
                "avg_video_rank": (sum(ranks) / len(ranks)) if ranks else None,
                "missing": strategy_missing.get(strategy_name, 0),
            }
        )
    strategy_summary = sorted(
        strategy_summary,
        key=lambda item: (
            -float(item["video_recall_at_10"]),
            -float(item["video_recall_at_20"]),
            float(item["avg_video_rank"]) if item["avg_video_rank"] is not None else 10**9,
            str(item["strategy"]),
        ),
    )

    payload = {
        "num_queries": n,
        "offset": int(offset),
        "limit": limit,
        "model_spaces": selected_models,
        "vector_fields": vector_fields,
        "require_verified_gt_video": bool(require_verified_gt_video),
        "require_gt_span_keyframe": bool(require_gt_span_keyframe),
        "requested_global_depth": int(global_depth),
        "effective_global_depth": int(effective_global_depth),
        "strategy_summary": strategy_summary,
        "per_query": per_query,
    }

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_root = Path("/data/outputs/diagnostics" if modal is not None else "outputs/diagnostics")
        safe_models = "_".join(selected_models)
        output_path_obj = output_root / f"video_aggregation_{safe_models}_{timestamp}.json"
    else:
        output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    if modal is not None:
        volume.commit()

    summary = {
        "num_queries": n,
        "model_spaces": selected_models,
        "requested_global_depth": int(global_depth),
        "effective_global_depth": int(effective_global_depth),
        "top_strategies": strategy_summary[:10],
        "output_path": str(output_path_obj),
    }
    return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)


@app.local_entrypoint(name="diagnose_video_aggregation_strategies_cli")
def diagnose_video_aggregation_strategies_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = 100,
    offset: int = 0,
    model_spaces: str = "openclip,beit3",
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
    global_depth: int = 1000,
    output_path: str | None = None,
):
    print(
        diagnose_video_aggregation_strategies.remote(
            config_path=config_path,
            dataset_path=dataset_path,
            limit=limit,
            offset=offset,
            model_spaces=model_spaces,
            require_verified_gt_video=require_verified_gt_video,
            require_gt_span_keyframe=require_gt_span_keyframe,
            global_depth=global_depth,
            output_path=output_path,
        )
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def diagnose_gemini_visual_rerank(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = 10,
    offset: int = 0,
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
    global_depth: int = 1024,
    candidate_videos: int = 10,
    output_path: str | None = None,
) -> str:
    _prepare_runtime()
    from datetime import datetime, timezone
    from pathlib import Path

    from src.eval.metrics import compute_hits
    from src.eval.run_evaluation import (
        filter_samples_with_gt_span_keyframes,
        load_query_samples,
        load_verified_video_ids,
    )
    from src.models.load_beit3 import BEiT3Encoder
    from src.schemas import RetrievalCandidate
    from src.storage.zilliz_client import ZillizClient
    from src.utils.config import get_config_value, load_config
    from src.utils.video_io import read_frames_by_index

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    dataset_path = dataset_path or (
        "/data/TimeLens-Bench/charades-timelens-query-samples.json"
        if modal is not None
        else get_config_value(config, "paths.dataset_query_samples")
    )
    samples = load_query_samples(dataset_path)
    if require_verified_gt_video:
        verified_video_ids = load_verified_video_ids(config)
        samples = [sample for sample in samples if sample.video_id in verified_video_ids]
    if require_gt_span_keyframe:
        samples = filter_samples_with_gt_span_keyframes(config, samples)
    if offset:
        samples = samples[int(offset) :]
    if limit is not None:
        samples = samples[:limit]

    api_key_env = get_config_value(config, "models.gemini.api_key_env", "GEMINI_API_KEY")
    model_env = get_config_value(config, "models.gemini.model_env", "GEMINI_MODEL")
    api_key = os.environ.get(api_key_env, "")
    model_name = os.environ.get(model_env, "")
    if not api_key or not model_name:
        raise RuntimeError(f"{api_key_env} and {model_env} must be set.")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name)
    zilliz = ZillizClient(config)
    beit3 = BEiT3Encoder(config).load()
    effective_global_depth = min(int(global_depth), int(get_config_value(config, "zilliz.max_search_top_k", 1024)))
    candidate_videos = max(1, min(int(candidate_videos), 20))

    per_query = []
    baseline_rows = []
    rerank_rows = []
    errors = []

    for sample in samples:
        try:
            vector = beit3.encode_texts([sample.query_text])[0]
            global_results = zilliz.search(zilliz.beit3_field, vector, effective_global_depth)
            grouped = _group_candidates_by_video_for_modal(global_results)
            ranked_videos = sorted(
                grouped.items(),
                key=lambda item: (-_mean_top_k_score_for_modal(item[1], k=3), item[0]),
            )
            baseline_candidates = [candidates[0] for _, candidates in ranked_videos[:candidate_videos]]

            candidate_images = []
            usable_candidates = []
            for candidate in baseline_candidates:
                source_path, _ = _resolve_video_path("", candidate.video_id, config)
                frames = read_frames_by_index(source_path, [candidate.frame_index_raw]) if source_path is not None else []
                if frames:
                    usable_candidates.append(candidate)
                    candidate_images.append(frames[0])

            if not usable_candidates:
                raise RuntimeError(f"No candidate frames could be read for query_id={sample.query_id}.")

            contact_sheet = _make_candidate_contact_sheet(candidate_images)
            candidate_lines = [
                f"{index}. keyframe_id={candidate.keyframe_id}, video_id={candidate.video_id}, timestamp={candidate.timestamp_raw:.3f}"
                for index, candidate in enumerate(usable_candidates, start=1)
            ]
            prompt = (
                "You are reranking candidate video keyframes for text-to-keyframe retrieval.\n"
                "Choose the candidates whose visual content best matches the query action, objects, and scene.\n"
                "The attached contact sheet contains numbered candidate keyframes.\n"
                "Return valid JSON only in this format: {\"ranking\": [1, 2, 3], \"reason\": \"short reason\"}.\n"
                "Rank all visible candidate numbers from best to worst. Do not include numbers outside the list.\n\n"
                f"Query: {sample.query_text}\n\n"
                "Candidates:\n" + "\n".join(candidate_lines)
            )
            response = gemini_model.generate_content([prompt, contact_sheet])
            payload = _parse_json_from_text(response.text or "{}")
            ranking = []
            seen = set()
            for item in payload.get("ranking", []):
                try:
                    number = int(item)
                except Exception:
                    continue
                if 1 <= number <= len(usable_candidates) and number not in seen:
                    seen.add(number)
                    ranking.append(number)
            for number in range(1, len(usable_candidates) + 1):
                if number not in seen:
                    ranking.append(number)

            reranked_candidates = [usable_candidates[number - 1] for number in ranking]
            baseline_ranked = assign_ranks_for_modal(baseline_candidates)
            reranked = assign_ranks_for_modal(reranked_candidates)
            baseline_hits = compute_hits(baseline_ranked, sample.video_id, sample.gt_span)
            rerank_hits = compute_hits(reranked, sample.video_id, sample.gt_span)
            baseline_rows.append(baseline_hits)
            rerank_rows.append(rerank_hits)
            per_query.append(
                {
                    "query_id": sample.query_id,
                    "query_text": sample.query_text,
                    "gt_video_id": sample.video_id,
                    "gt_span": sample.gt_span,
                    "gemini_ranking": ranking,
                    "gemini_reason": payload.get("reason"),
                    "baseline_hits": baseline_hits,
                    "rerank_hits": rerank_hits,
                    "baseline_candidates": [
                        _candidate_to_eval_dict(candidate, rank)
                        for rank, candidate in enumerate(baseline_ranked, start=1)
                    ],
                    "reranked_candidates": [
                        _candidate_to_eval_dict(candidate, rank)
                        for rank, candidate in enumerate(reranked, start=1)
                    ],
                }
            )
        except Exception as exc:
            errors.append({"query_id": sample.query_id, "error": repr(exc)})

    def recall(rows: list[Dict[str, Any]], key: str) -> float:
        return (sum(1 for row in rows if row.get(key)) / len(samples)) if samples else 0.0

    summary = {
        "num_queries": len(samples),
        "evaluated_queries": len(per_query),
        "errors": len(errors),
        "global_depth": int(effective_global_depth),
        "candidate_videos": int(candidate_videos),
        "baseline": {
            "recall_at_1": recall(baseline_rows, "hit_at_1"),
            "recall_at_5": recall(baseline_rows, "hit_at_5"),
            "recall_at_10": recall(baseline_rows, "hit_at_10"),
            "video_recall_at_1": recall(baseline_rows, "video_hit_at_1"),
            "video_recall_at_5": recall(baseline_rows, "video_hit_at_5"),
            "video_recall_at_10": recall(baseline_rows, "video_hit_at_10"),
        },
        "gemini_visual_rerank": {
            "recall_at_1": recall(rerank_rows, "hit_at_1"),
            "recall_at_5": recall(rerank_rows, "hit_at_5"),
            "recall_at_10": recall(rerank_rows, "hit_at_10"),
            "video_recall_at_1": recall(rerank_rows, "video_hit_at_1"),
            "video_recall_at_5": recall(rerank_rows, "video_hit_at_5"),
            "video_recall_at_10": recall(rerank_rows, "video_hit_at_10"),
        },
    }

    payload = {**summary, "per_query": per_query, "error_samples": errors[:20]}
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_root = Path("/data/outputs/diagnostics" if modal is not None else "outputs/diagnostics")
        output_path_obj = output_root / f"gemini_visual_rerank_{timestamp}.json"
    else:
        output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    if modal is not None:
        volume.commit()
    summary["output_path"] = str(output_path_obj)
    return json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)


def _group_candidates_by_video_for_modal(candidates: list[Any]) -> Dict[str, list[Any]]:
    grouped: Dict[str, list[Any]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.video_id, []).append(candidate)
    for video_candidates in grouped.values():
        video_candidates.sort(key=lambda item: (-float(item.score), item.keyframe_id))
    return grouped


def _mean_top_k_score_for_modal(candidates: list[Any], k: int = 3) -> float:
    scores = sorted((float(candidate.score) for candidate in candidates), reverse=True)[:k]
    return sum(scores) / len(scores) if scores else float("-inf")


def assign_ranks_for_modal(candidates: list[Any]) -> list[Any]:
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates


@app.local_entrypoint(name="diagnose_gemini_visual_rerank_cli")
def diagnose_gemini_visual_rerank_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    limit: int | None = 10,
    offset: int = 0,
    require_verified_gt_video: bool = True,
    require_gt_span_keyframe: bool = True,
    global_depth: int = 1024,
    candidate_videos: int = 10,
    output_path: str | None = None,
):
    print(
        diagnose_gemini_visual_rerank.remote(
            config_path=config_path,
            dataset_path=dataset_path,
            limit=limit,
            offset=offset,
            require_verified_gt_video=require_verified_gt_video,
            require_gt_span_keyframe=require_gt_span_keyframe,
            global_depth=global_depth,
            candidate_videos=candidate_videos,
            output_path=output_path,
        )
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def diagnose_embedding_self_retrieval(
    config_path: str | None = None,
    model_space: str = "openclip",
    limit: int = 20,
    top_k: int = 5,
) -> str:
    _prepare_runtime()
    from pathlib import Path

    from src.models.load_beit3 import BEiT3Encoder
    from src.models.load_openclip import OpenCLIPH14Encoder
    from src.storage.cockroach_client import CockroachClient
    from src.storage.zilliz_client import ZillizClient
    from src.utils.config import get_config_value, load_config
    from src.utils.video_io import read_frames_by_index

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    zilliz = ZillizClient(config)
    cockroach = CockroachClient(config)
    try:
        if model_space == "openclip":
            encoder = OpenCLIPH14Encoder(config).load()
            vector_field = zilliz.openclip_field
        elif model_space == "beit3":
            encoder = BEiT3Encoder(config).load()
            vector_field = zilliz.beit3_field
        else:
            raise ValueError("model_space must be 'openclip' or 'beit3'.")

        with cockroach.conn.cursor() as cur:
            cur.execute(
                """
                SELECT k.keyframe_id, k.video_id, k.frame_index_raw, v.source_path
                FROM keyframes k
                JOIN videos v ON v.video_id = k.video_id
                WHERE k.processing_status = 'VERIFIED'
                  AND k.model_version = %s
                  AND k.config_version = %s
                ORDER BY k.video_id, k.frame_index_raw
                LIMIT %s
                """,
                (config["project"]["model_version"], config["project"]["config_version"], int(limit)),
            )
            rows = list(cur.fetchall())

        checked = 0
        top1_hits = 0
        topk_hits = 0
        failures = []
        for row in rows:
            video_id = str(row["video_id"])
            keyframe_id = str(row["keyframe_id"])
            frame_index = int(row["frame_index_raw"])
            source_path, checked_paths = _resolve_video_path(str(row["source_path"]), video_id, config)
            frames = read_frames_by_index(source_path, [frame_index]) if source_path is not None else []
            if not frames:
                if len(failures) < 20:
                    failures.append(
                        {
                            "keyframe_id": keyframe_id,
                            "video_id": video_id,
                            "frame_index_raw": frame_index,
                            "error": "cannot_read_frame",
                            "checked_paths": checked_paths,
                        }
                    )
                continue

            vector = encoder.encode_images(frames)[0]
            results = zilliz.search(vector_field, vector, int(top_k))
            checked += 1
            result_ids = [candidate.keyframe_id for candidate in results]
            if result_ids[:1] == [keyframe_id]:
                top1_hits += 1
            if keyframe_id in result_ids:
                topk_hits += 1
            elif len(failures) < 20:
                failures.append(
                    {
                        "keyframe_id": keyframe_id,
                        "video_id": video_id,
                        "frame_index_raw": frame_index,
                        "top_results": _candidate_preview(results, limit=int(top_k)),
                    }
                )

        payload = {
            "model_space": model_space,
            "vector_field": vector_field,
            "requested": int(limit),
            "checked": checked,
            "top_k": int(top_k),
            "self_retrieval_top1": (top1_hits / checked) if checked else 0.0,
            "self_retrieval_topk": (topk_hits / checked) if checked else 0.0,
            "failures": failures,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    finally:
        cockroach.close()


@app.local_entrypoint(name="diagnose_embedding_self_retrieval_cli")
def diagnose_embedding_self_retrieval_cli(
    config_path: str | None = None,
    model_space: str = "openclip",
    limit: int = 20,
    top_k: int = 5,
):
    print(
        diagnose_embedding_self_retrieval.remote(
            config_path=config_path,
            model_space=model_space,
            limit=limit,
            top_k=top_k,
        )
    )


@app.function(cpu=1, memory=1024, timeout=300, **function_kwargs)
def inspect_modal_volume_video_paths(video_ids: str = "00607,00T1E,02DPI") -> str:
    _prepare_runtime()
    from pathlib import Path

    ids = [item.strip() for item in str(video_ids).split(",") if item.strip()]
    roots = [
        Path("/data/TimeLens-Bench/videos/charades"),
        Path("/data/data/TimeLens-Bench/videos/charades"),
        Path("/data/TimeLens-Bench"),
        Path("/data/data/TimeLens-Bench"),
        Path("/data"),
    ]
    root_payload = []
    for root in roots:
        item: Dict[str, Any] = {
            "path": str(root),
            "exists": root.exists(),
            "is_dir": root.is_dir(),
        }
        if root.exists() and root.is_dir():
            mp4s = list(root.glob("*.mp4"))
            item["mp4_count_direct"] = len(mp4s)
            item["sample_mp4s_direct"] = [path.name for path in mp4s[:10]]
            item["sample_children"] = [path.name for path in list(root.iterdir())[:20]]
        root_payload.append(item)

    video_payload = []
    for video_id in ids:
        candidates = []
        for root in roots[:2]:
            path = root / f"{video_id}.mp4"
            candidates.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
                }
            )
        video_payload.append({"video_id": video_id, "candidates": candidates})

    return json.dumps(
        {
            "roots": root_payload,
            "videos": video_payload,
        },
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    )


@app.local_entrypoint(name="inspect_modal_volume_video_paths_cli")
def inspect_modal_volume_video_paths_cli(video_ids: str = "00607,00T1E,02DPI"):
    print(inspect_modal_volume_video_paths.remote(video_ids=video_ids))


@app.function(cpu=1, memory=1024, timeout=900, **function_kwargs)
def repair_sdxl_assets(config_path: str | None = None, repo_id: str = "stabilityai/stable-diffusion-xl-base-1.0") -> str:
    _prepare_runtime()
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from src.utils.config import load_config

    config_path = _default_config_path(config_path)
    config = load_config(config_path)
    model_dir = Path(config["models"]["stable_diffusion"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename="model_index.json",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    if modal is not None:
        volume.commit()
    payload = {
        "downloaded": downloaded,
        "model_dir": str(model_dir),
        "exists": (model_dir / "model_index.json").exists(),
    }
    return json.dumps(payload, indent=2, default=_json_default)


@app.local_entrypoint(name="repair_sdxl_assets_cli")
def repair_sdxl_assets_cli(config_path: str | None = None, repo_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    print(repair_sdxl_assets.remote(config_path=config_path, repo_id=repo_id))


@app.function(gpu="L4", cpu=8, memory=65536, timeout=21600, **function_kwargs)
def run_online_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    disable_object_filter: bool = False,
    online_mode: str | None = None,
    baseline_openclip_only: bool = False,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
    video_aggregation_global_depth: int | None = None,
    video_aggregation_keyframes_per_video: int | None = None,
    xpool_paperlike_num_frames: int | None = None,
    xpool_paperlike_keyframes_per_video: int | None = None,
    xpool_paperlike_batch_size: int | None = None,
    xpool_paperlike_num_heads: int | None = None,
    xpool_paperlike_dropout: float | None = None,
    xpool_paperlike_checkpoint: str | None = None,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        disable_object_filter=disable_object_filter,
        online_mode=online_mode,
        baseline_openclip_only=baseline_openclip_only,
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
        video_aggregation_global_depth=video_aggregation_global_depth,
        video_aggregation_keyframes_per_video=video_aggregation_keyframes_per_video,
        xpool_paperlike_num_frames=xpool_paperlike_num_frames,
        xpool_paperlike_keyframes_per_video=xpool_paperlike_keyframes_per_video,
        xpool_paperlike_batch_size=xpool_paperlike_batch_size,
        xpool_paperlike_num_heads=xpool_paperlike_num_heads,
        xpool_paperlike_dropout=xpool_paperlike_dropout,
        xpool_paperlike_checkpoint=xpool_paperlike_checkpoint,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_fusionista_temporal_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    output_depth: int = 10,
    llm_timeout_seconds: int = 90,
    checkpoint_every: int = 10,
    max_subqueries: int = 4,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.online.temporal_solutions import run_temporal_solution_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or "/data/TimeLens-Bench/charades-timelens-query-samples-temporal-scene-moment-first1000.json"
    output_dir = output_dir or "/data/outputs/eval"
    result = run_temporal_solution_evaluation(
        solution="fusionista",
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        output_depth=output_depth,
        llm_timeout_seconds=llm_timeout_seconds,
        checkpoint_every=checkpoint_every,
        max_subqueries=max_subqueries,
    )
    if modal is not None:
        volume.commit()
    return result


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_exquisitor_temporal_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    output_depth: int = 10,
    llm_timeout_seconds: int = 90,
    checkpoint_every: int = 10,
    max_subqueries: int = 4,
    initial_depth: int = 1000,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.online.temporal_solutions import run_temporal_solution_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or "/data/TimeLens-Bench/charades-timelens-query-samples-temporal-scene-moment-first1000.json"
    output_dir = output_dir or "/data/outputs/eval"
    result = run_temporal_solution_evaluation(
        solution="exquisitor",
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        output_depth=output_depth,
        llm_timeout_seconds=llm_timeout_seconds,
        checkpoint_every=checkpoint_every,
        max_subqueries=max_subqueries,
        exquisitor_initial_depth=initial_depth,
    )
    if modal is not None:
        volume.commit()
    return result


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_viewsinsight_temporal_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    output_depth: int = 10,
    llm_timeout_seconds: int = 90,
    checkpoint_every: int = 10,
    max_subqueries: int = 4,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.online.temporal_solutions import run_temporal_solution_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or "/data/TimeLens-Bench/charades-timelens-query-samples-temporal-scene-moment-first1000.json"
    output_dir = output_dir or "/data/outputs/eval"
    result = run_temporal_solution_evaluation(
        solution="viewsinsight",
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        output_depth=output_depth,
        llm_timeout_seconds=llm_timeout_seconds,
        checkpoint_every=checkpoint_every,
        max_subqueries=max_subqueries,
    )
    if modal is not None:
        volume.commit()
    return result


@app.local_entrypoint(name="run_fusionista_temporal_evaluation_cli")
def run_fusionista_temporal_evaluation_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    output_depth: int = 10,
    llm_timeout_seconds: int = 90,
    checkpoint_every: int = 10,
    max_subqueries: int = 4,
):
    result = run_fusionista_temporal_evaluation.remote(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        output_depth=output_depth,
        llm_timeout_seconds=llm_timeout_seconds,
        checkpoint_every=checkpoint_every,
        max_subqueries=max_subqueries,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


@app.local_entrypoint(name="run_exquisitor_temporal_evaluation_cli")
def run_exquisitor_temporal_evaluation_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    output_depth: int = 10,
    llm_timeout_seconds: int = 90,
    checkpoint_every: int = 10,
    max_subqueries: int = 4,
    initial_depth: int = 1000,
):
    result = run_exquisitor_temporal_evaluation.remote(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        output_depth=output_depth,
        llm_timeout_seconds=llm_timeout_seconds,
        checkpoint_every=checkpoint_every,
        max_subqueries=max_subqueries,
        initial_depth=initial_depth,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


@app.local_entrypoint(name="run_viewsinsight_temporal_evaluation_cli")
def run_viewsinsight_temporal_evaluation_cli(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    output_depth: int = 10,
    llm_timeout_seconds: int = 90,
    checkpoint_every: int = 10,
    max_subqueries: int = 4,
):
    result = run_viewsinsight_temporal_evaluation.remote(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        output_depth=output_depth,
        llm_timeout_seconds=llm_timeout_seconds,
        checkpoint_every=checkpoint_every,
        max_subqueries=max_subqueries,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_openclip_baseline_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        baseline_openclip_only=True,
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_openclip_scene_rewrite_baseline_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="openclip_llm_scene_rewrite_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_beit3_baseline_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="beit3_user_query_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_textual_baseline_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="textual_query_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_video_aggregation_baseline_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
    global_depth: int = 1000,
    keyframes_per_video: int = 1,
    score_strategy: str = "mean_topk",
    score_top_k: int = 3,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="beit3_video_mean_top3_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
        video_aggregation_global_depth=global_depth,
        video_aggregation_keyframes_per_video=keyframes_per_video,
        video_score_strategy=score_strategy,
        video_score_top_k=score_top_k,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_window_aggregation_baseline_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
    global_depth: int = 1000,
    window_size: float = 8.0,
    window_stride: float = 4.0,
    score_strategy: str = "mean_topk",
    score_top_k: int = 3,
    keyframes_per_window: int = 1,
    max_windows_per_video: int = 1,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="beit3_window_mean_top3_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
        video_aggregation_global_depth=global_depth,
        window_size_seconds=window_size,
        window_stride_seconds=window_stride,
        window_score_strategy=score_strategy,
        window_score_top_k=score_top_k,
        keyframes_per_window=keyframes_per_window,
        max_windows_per_video=max_windows_per_video,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_beit3_xpool_softmax_heuristic_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
    candidate_videos: int = 100,
    keyframes_per_video: int = 1,
    temperature: float = 0.07,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="beit3_video_xpool_softmax_heuristic_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
        xpool_candidate_videos=candidate_videos,
        xpool_keyframes_per_video=keyframes_per_video,
        xpool_temperature=temperature,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_beit3_xpool_heuristic_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
    candidate_videos: int = 100,
    keyframes_per_video: int = 1,
    temperature: float = 0.07,
    num_heads: int | None = None,
    dropout: float | None = None,
    checkpoint: str | None = None,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="beit3_video_xpool_heuristic_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
        xpool_candidate_videos=candidate_videos,
        xpool_keyframes_per_video=keyframes_per_video,
        xpool_temperature=temperature,
        xpool_num_heads=num_heads,
        xpool_dropout=dropout,
        xpool_checkpoint=checkpoint,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_beit3_xpool_paperlike_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
    num_frames: int = 12,
    keyframes_per_video: int = 1,
    batch_size: int = 128,
    num_heads: int | None = None,
    dropout: float | None = None,
    checkpoint: str | None = None,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="beit3_video_xpool_paperlike_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
        xpool_paperlike_num_frames=num_frames,
        xpool_paperlike_keyframes_per_video=keyframes_per_video,
        xpool_paperlike_batch_size=batch_size,
        xpool_paperlike_num_heads=num_heads,
        xpool_paperlike_dropout=dropout,
        xpool_paperlike_checkpoint=checkpoint,
    )


@app.function(gpu="L4", cpu=4, memory=32768, timeout=21600, **function_kwargs)
def run_llm_video_aggregation_baseline_evaluation(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    latency_mode: str | None = None,
    require_verified_gt_video: bool = False,
    require_gt_span_keyframe: bool = False,
    global_depth: int = 1000,
    keyframes_per_video: int = 1,
    score_strategy: str = "mean_topk",
    score_top_k: int = 3,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.run_evaluation import run_online_evaluation as _run_online_evaluation

    config_path = _default_config_path(config_path)
    dataset_path = dataset_path or ("/data/TimeLens-Bench/charades-timelens-query-samples.json" if modal is not None else None)
    output_dir = output_dir or ("/data/outputs/eval" if modal is not None else None)
    return _run_online_evaluation(
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=limit,
        offset=offset,
        latency_mode=latency_mode,
        online_mode="beit3_llm_video_mean_top3_baseline",
        require_verified_gt_video=require_verified_gt_video,
        require_gt_span_keyframe=require_gt_span_keyframe,
        video_aggregation_global_depth=global_depth,
        video_aggregation_keyframes_per_video=keyframes_per_video,
        video_score_strategy=score_strategy,
        video_score_top_k=score_top_k,
    )


@app.function(gpu="L4", cpu=8, memory=65536, timeout=3600, **function_kwargs)
def search_single_query(
    query: str,
    config_path: str | None = None,
    latency_mode: str | None = None,
    disable_object_filter: bool = False,
    online_mode: str | None = None,
    baseline_openclip_only: bool = False,
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.online.search_pipeline import search_single_query as _search_single_query

    config_path = _default_config_path(config_path)
    return _search_single_query(
        query=query,
        config_path=config_path,
        latency_mode=latency_mode,
        disable_object_filter=disable_object_filter,
        online_mode=online_mode,
        baseline_openclip_only=baseline_openclip_only,
    )


@app.function(gpu="L4", cpu=8, memory=65536, timeout=3600, **function_kwargs)
def search_single_query_metrics(
    query: str,
    gt_video_id: str,
    gt_start: float = 0.0,
    gt_end: float = 1e9,
    config_path: str | None = None,
    latency_mode: str | None = None,
    online_mode: str | None = "beit3_video_xpool_heuristic_baseline",
) -> Dict[str, Any]:
    _prepare_runtime()
    from src.eval.metrics import compute_hits
    from src.online.search_pipeline import SearchPipeline
    from src.utils.config import load_config

    config_path = _default_config_path(config_path)
    overrides: Dict[str, Any] = {"object_filter": {"enabled": False}}
    if online_mode:
        overrides["online"] = {"mode": online_mode}
    pipeline = SearchPipeline(load_config(config_path, overrides=overrides))
    response = pipeline.search(query, latency_mode=latency_mode)
    hits = compute_hits(response.results, gt_video_id, [gt_start, gt_end])
    return {
        "keyframe_recall_at_1": 1.0 if hits["hit_at_1"] else 0.0,
        "keyframe_recall_at_5": 1.0 if hits["hit_at_5"] else 0.0,
        "keyframe_recall_at_10": 1.0 if hits["hit_at_10"] else 0.0,
        "video_recall_at_1": 1.0 if hits["video_hit_at_1"] else 0.0,
        "video_recall_at_5": 1.0 if hits["video_hit_at_5"] else 0.0,
        "video_recall_at_10": 1.0 if hits["video_hit_at_10"] else 0.0,
    }


@app.local_entrypoint(name="search_single_query_cli")
def search_single_query_cli(
    query: str,
    config_path: str | None = None,
    latency_mode: str | None = None,
    disable_object_filter: bool = False,
    online_mode: str | None = None,
    baseline_openclip_only: bool = False,
):
    result = search_single_query.remote(
        query=query,
        config_path=config_path,
        latency_mode=latency_mode,
        disable_object_filter=disable_object_filter,
        online_mode=online_mode,
        baseline_openclip_only=baseline_openclip_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


@app.local_entrypoint(name="search_single_query_metrics_cli")
def search_single_query_metrics_cli(
    query: str,
    gt_video_id: str,
    gt_start: float = 0.0,
    gt_end: float = 1e9,
    config_path: str | None = None,
    latency_mode: str | None = None,
    online_mode: str | None = "beit3_video_xpool_heuristic_baseline",
):
    result = search_single_query_metrics.remote(
        query=query,
        gt_video_id=gt_video_id,
        gt_start=gt_start,
        gt_end=gt_end,
        config_path=config_path,
        latency_mode=latency_mode,
        online_mode=online_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
