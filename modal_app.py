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
            results_by_model[model_name] = zilliz.search(vector_fields[model_name], vector, int(global_depth))

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
        "global_depth": int(global_depth),
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
        "global_depth": int(global_depth),
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
    )


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
