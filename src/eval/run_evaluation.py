from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.eval.export_results import write_manifest, write_metrics, write_per_query_results
from src.eval.metrics import aggregate_metrics, compute_hits
from src.online.search_pipeline import SearchPipeline
from src.schemas import QuerySample
from src.utils.config import get_config_value, load_config
from src.utils.ids import make_eval_run_id
from src.utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


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
    video_score_strategy: str | None = None,
    video_score_top_k: int | None = None,
    window_size_seconds: float | None = None,
    window_stride_seconds: float | None = None,
    window_score_strategy: str | None = None,
    window_score_top_k: int | None = None,
    keyframes_per_window: int | None = None,
    max_windows_per_video: int | None = None,
    xpool_candidate_videos: int | None = None,
    xpool_keyframes_per_video: int | None = None,
    xpool_temperature: float | None = None,
    xpool_num_heads: int | None = None,
    xpool_dropout: float | None = None,
    xpool_checkpoint: str | None = None,
    xpool_paperlike_num_frames: int | None = None,
    xpool_paperlike_keyframes_per_video: int | None = None,
    xpool_paperlike_batch_size: int | None = None,
    xpool_paperlike_num_heads: int | None = None,
    xpool_paperlike_dropout: float | None = None,
    xpool_paperlike_checkpoint: str | None = None,
) -> Dict[str, Any]:
    setup_logging()
    overrides: Dict[str, Any] = {}
    if disable_object_filter:
        overrides["object_filter"] = {"enabled": False}
    if online_mode:
        overrides["online"] = {"mode": online_mode}
        if online_mode in {
            "openclip_user_query_baseline",
            "openclip_llm_scene_rewrite_baseline",
            "beit3_user_query_baseline",
            "textual_query_baseline",
            "beit3_video_mean_top3_baseline",
            "beit3_window_mean_top3_baseline",
            "beit3_video_xpool_softmax_heuristic_baseline",
            "beit3_video_xpool_heuristic_baseline",
            "beit3_video_xpool_paperlike_baseline",
            "beit3_llm_video_mean_top3_baseline",
        }:
            overrides["object_filter"] = {"enabled": False}
    video_aggregation_overrides: Dict[str, Any] = {}
    if video_aggregation_global_depth is not None:
        video_aggregation_overrides["global_depth"] = int(video_aggregation_global_depth)
    if video_aggregation_keyframes_per_video is not None:
        video_aggregation_overrides["keyframes_per_video"] = int(video_aggregation_keyframes_per_video)
    if video_score_strategy is not None:
        video_aggregation_overrides["score_strategy"] = str(video_score_strategy)
    if video_score_top_k is not None:
        video_aggregation_overrides["score_top_k"] = int(video_score_top_k)
    if video_aggregation_overrides:
        overrides["video_aggregation"] = video_aggregation_overrides
    window_aggregation_overrides: Dict[str, Any] = {}
    if window_size_seconds is not None:
        window_aggregation_overrides["window_size_seconds"] = float(window_size_seconds)
    if window_stride_seconds is not None:
        window_aggregation_overrides["window_stride_seconds"] = float(window_stride_seconds)
    if window_score_strategy is not None:
        window_aggregation_overrides["score_strategy"] = str(window_score_strategy)
    if window_score_top_k is not None:
        window_aggregation_overrides["window_score_top_k"] = int(window_score_top_k)
    if keyframes_per_window is not None:
        window_aggregation_overrides["keyframes_per_window"] = int(keyframes_per_window)
    if max_windows_per_video is not None:
        window_aggregation_overrides["max_windows_per_video"] = int(max_windows_per_video)
    if window_aggregation_overrides:
        overrides["window_aggregation"] = window_aggregation_overrides
    xpool_overrides: Dict[str, Any] = {}
    if xpool_candidate_videos is not None:
        xpool_overrides["candidate_videos"] = int(xpool_candidate_videos)
    if xpool_keyframes_per_video is not None:
        xpool_overrides["keyframes_per_video"] = int(xpool_keyframes_per_video)
    if xpool_temperature is not None:
        xpool_overrides["temperature"] = float(xpool_temperature)
    if xpool_num_heads is not None:
        xpool_overrides["num_heads"] = int(xpool_num_heads)
    if xpool_dropout is not None:
        xpool_overrides["dropout"] = float(xpool_dropout)
    if xpool_checkpoint is not None:
        xpool_overrides["checkpoint"] = str(xpool_checkpoint)
    if xpool_overrides:
        overrides["xpool_heuristic"] = xpool_overrides
    xpool_paperlike_overrides: Dict[str, Any] = {}
    if xpool_paperlike_num_frames is not None:
        xpool_paperlike_overrides["num_frames"] = int(xpool_paperlike_num_frames)
    if xpool_paperlike_keyframes_per_video is not None:
        xpool_paperlike_overrides["keyframes_per_video"] = int(xpool_paperlike_keyframes_per_video)
    if xpool_paperlike_batch_size is not None:
        xpool_paperlike_overrides["batch_size"] = int(xpool_paperlike_batch_size)
    if xpool_paperlike_num_heads is not None:
        xpool_paperlike_overrides["num_heads"] = int(xpool_paperlike_num_heads)
    if xpool_paperlike_dropout is not None:
        xpool_paperlike_overrides["dropout"] = float(xpool_paperlike_dropout)
    if xpool_paperlike_checkpoint is not None:
        xpool_paperlike_overrides["checkpoint"] = str(xpool_paperlike_checkpoint)
    if xpool_paperlike_overrides:
        overrides["xpool_paperlike"] = xpool_paperlike_overrides
    if baseline_openclip_only:
        overrides["online"] = {"mode": "openclip_user_query_baseline"}
        overrides["object_filter"] = {"enabled": False}
    if not overrides:
        overrides = None
    config = load_config(config_path, overrides=overrides)
    dataset_path = dataset_path or get_config_value(config, "paths.dataset_query_samples")
    output_dir = output_dir or get_config_value(config, "evaluation.output_dir", "outputs/eval")
    latency_mode = latency_mode or get_config_value(config, "evaluation.latency_mode", "cache_miss_full_online")
    samples = load_query_samples(dataset_path)
    if require_verified_gt_video:
        verified_video_ids = load_verified_video_ids(config)
        before_count = len(samples)
        samples = [sample for sample in samples if sample.video_id in verified_video_ids]
        logger.info(
            "eval_samples_filtered_verified_gt before=%d after=%d",
            before_count,
            len(samples),
        )
    if require_gt_span_keyframe:
        before_count = len(samples)
        samples = filter_samples_with_gt_span_keyframes(config, samples)
        logger.info(
            "eval_samples_filtered_gt_span_keyframe before=%d after=%d",
            before_count,
            len(samples),
        )
    if offset:
        samples = samples[int(offset):]
    if limit is not None:
        samples = samples[:limit]

    pipeline = SearchPipeline(config)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    eval_run_id = make_eval_run_id(
        config["project"]["dataset_name"],
        config["project"]["config_version"],
        config["project"]["model_version"],
        timestamp_utc,
    )
    initial_manifest = build_run_manifest(config, latency_mode)
    try:
        pipeline.cockroach.insert_eval_run(
            eval_run_id=eval_run_id,
            dataset_name=config["project"]["dataset_name"],
            dataset_path=str(dataset_path),
            config_version=config["project"]["config_version"],
            model_version=config["project"]["model_version"],
            latency_mode=latency_mode,
            branch_config=initial_manifest["config"],
            metrics={},
        )
    except Exception:
        logger.exception("eval_run_initial_db_insert_failed eval_run_id=%s", eval_run_id)

    per_query: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        logger.info("eval_query_start index=%d total=%d query_id=%s", index, len(samples), sample.query_id)
        start = time.perf_counter()
        try:
            response = pipeline.search(sample.query_text, latency_mode=latency_mode)
            hits = compute_hits(response.results, sample.video_id, sample.gt_span)
            row = {
                "query_id": sample.query_id,
                "query_text": sample.query_text,
                "gt_video_id": sample.video_id,
                "gt_span": sample.gt_span,
                "top_retrieved_keyframes": [candidate.to_eval_dict() for candidate in response.results],
                **hits,
                "latency_ms": response.latency_ms,
                "latency_breakdown_ms": response.latency_breakdown_ms,
                "error": None,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.exception("eval_query_failed query_id=%s", sample.query_id)
            row = {
                "query_id": sample.query_id,
                "query_text": sample.query_text,
                "gt_video_id": sample.video_id,
                "gt_span": sample.gt_span,
                "top_retrieved_keyframes": [],
                "hit_at_1": False,
                "hit_at_5": False,
                "hit_at_10": False,
                "latency_ms": elapsed_ms,
                "latency_breakdown_ms": {},
                "error": str(exc),
            }
        per_query.append(row)
        try:
            pipeline.cockroach.insert_eval_query_result(eval_run_id, row)
        except Exception:
            logger.exception("eval_query_db_insert_failed query_id=%s", sample.query_id)

    aggregate = aggregate_metrics(per_query)
    metrics = {
        "eval_run_id": eval_run_id,
        "dataset_name": config["project"]["dataset_name"],
        "num_queries": aggregate["num_queries"],
        "recall_at_1": aggregate["recall_at_1"],
        "recall_at_5": aggregate["recall_at_5"],
        "recall_at_10": aggregate["recall_at_10"],
        "video_recall_at_1": aggregate["video_recall_at_1"],
        "video_recall_at_5": aggregate["video_recall_at_5"],
        "video_recall_at_10": aggregate["video_recall_at_10"],
        "mean_time_latency_ms": aggregate["mean_time_latency_ms"],
        "latency_mode": latency_mode,
        "online_mode": str(get_config_value(config, "online.mode", "full")),
        "object_filter_enabled": bool(get_config_value(config, "object_filter.enabled", True)),
        "require_verified_gt_video": bool(require_verified_gt_video),
        "require_gt_span_keyframe": bool(require_gt_span_keyframe),
        "video_aggregation_global_depth": get_config_value(config, "video_aggregation.global_depth"),
        "video_aggregation_keyframes_per_video": get_config_value(config, "video_aggregation.keyframes_per_video"),
        "video_score_strategy": get_config_value(config, "video_aggregation.score_strategy"),
        "video_score_top_k": get_config_value(config, "video_aggregation.score_top_k"),
        "window_size_seconds": get_config_value(config, "window_aggregation.window_size_seconds"),
        "window_stride_seconds": get_config_value(config, "window_aggregation.window_stride_seconds"),
        "window_score_strategy": get_config_value(config, "window_aggregation.score_strategy"),
        "window_score_top_k": get_config_value(config, "window_aggregation.window_score_top_k"),
        "keyframes_per_window": get_config_value(config, "window_aggregation.keyframes_per_window"),
        "max_windows_per_video": get_config_value(config, "window_aggregation.max_windows_per_video"),
        "xpool_candidate_videos": get_config_value(config, "xpool_heuristic.candidate_videos"),
        "xpool_keyframes_per_video": get_config_value(config, "xpool_heuristic.keyframes_per_video"),
        "xpool_temperature": get_config_value(config, "xpool_heuristic.temperature"),
        "xpool_num_heads": get_config_value(config, "xpool_heuristic.num_heads"),
        "xpool_dropout": get_config_value(config, "xpool_heuristic.dropout"),
        "xpool_checkpoint": get_config_value(config, "xpool_heuristic.checkpoint"),
        "xpool_paperlike_num_frames": get_config_value(config, "xpool_paperlike.num_frames"),
        "xpool_paperlike_keyframes_per_video": get_config_value(config, "xpool_paperlike.keyframes_per_video"),
        "xpool_paperlike_batch_size": get_config_value(config, "xpool_paperlike.batch_size"),
        "xpool_paperlike_num_heads": get_config_value(config, "xpool_paperlike.num_heads"),
        "xpool_paperlike_dropout": get_config_value(config, "xpool_paperlike.dropout"),
        "xpool_paperlike_checkpoint": get_config_value(config, "xpool_paperlike.checkpoint"),
        "model_version": config["project"]["model_version"],
        "config_version": config["project"]["config_version"],
        "notes": _build_metrics_note(config),
    }
    manifest = initial_manifest
    output_root = Path(output_dir) / eval_run_id
    write_metrics(output_root, metrics)
    write_per_query_results(output_root, per_query)
    write_manifest(output_root, manifest)
    try:
        pipeline.cockroach.insert_eval_run(
            eval_run_id=eval_run_id,
            dataset_name=config["project"]["dataset_name"],
            dataset_path=str(dataset_path),
            config_version=config["project"]["config_version"],
            model_version=config["project"]["model_version"],
            latency_mode=latency_mode,
            branch_config=manifest["config"],
            metrics=metrics,
        )
    except Exception:
        logger.exception("eval_run_db_insert_failed eval_run_id=%s", eval_run_id)
    return {"eval_run_id": eval_run_id, "output_dir": str(output_root), "metrics": metrics}


def load_query_samples(path: str) -> List[QuerySample]:
    with open(path, "r", encoding="utf-8") as reader:
        payload = json.load(reader)
    if isinstance(payload, dict):
        rows = list(payload.values())
    else:
        rows = payload
    return [QuerySample.from_mapping(row) for row in rows]


def load_verified_video_ids(config: Dict[str, Any]) -> set[str]:
    from src.storage.cockroach_client import CockroachClient

    client = CockroachClient(config)
    try:
        with client.conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT video_id
                FROM videos
                WHERE processing_status = 'VERIFIED'
                  AND model_version = %s
                  AND config_version = %s
                """,
                (config["project"]["model_version"], config["project"]["config_version"]),
            )
            return {str(row["video_id"]) for row in cur.fetchall()}
    finally:
        client.close()


def filter_samples_with_gt_span_keyframes(config: Dict[str, Any], samples: List[QuerySample]) -> List[QuerySample]:
    from src.storage.cockroach_client import CockroachClient

    client = CockroachClient(config)
    try:
        filtered: List[QuerySample] = []
        with client.conn.cursor() as cur:
            for sample in samples:
                gt_start, gt_end = float(sample.gt_span[0]), float(sample.gt_span[1])
                cur.execute(
                    """
                    SELECT 1
                    FROM keyframes
                    WHERE video_id = %s
                      AND timestamp_raw >= %s
                      AND timestamp_raw <= %s
                      AND processing_status = 'VERIFIED'
                      AND model_version = %s
                      AND config_version = %s
                    LIMIT 1
                    """,
                    (
                        sample.video_id,
                        gt_start,
                        gt_end,
                        config["project"]["model_version"],
                        config["project"]["config_version"],
                    ),
                )
                if cur.fetchone():
                    filtered.append(sample)
        return filtered
    finally:
        client.close()


def build_run_manifest(config: Dict[str, Any], latency_mode: str) -> Dict[str, Any]:
    return {
        "models": {
            "beit3": get_config_value(config, "models.beit3.model_dir"),
            "openclip_h14": get_config_value(config, "models.openclip_h14.model_dir"),
            "transnetv2": get_config_value(config, "models.transnetv2.weights_dir"),
            "codetr": get_config_value(config, "models.codetr.checkpoint"),
            "stable_diffusion": get_config_value(config, "models.stable_diffusion.model_dir"),
            "gemini": "from_env",
        },
        "config": {
            "rrf_k": get_config_value(config, "rrf.rrf_k", 60),
            "branch_weights": get_config_value(config, "weighted_rrf.branch_weights", {}),
            "retrieval_depth": get_config_value(config, "retrieval.depth_per_model", 200),
            "post_fusion_depth": get_config_value(config, "weighted_rrf.post_fusion_depth", 500),
            "object_filter_enabled": bool(get_config_value(config, "object_filter.enabled", True)),
            "online_mode": str(get_config_value(config, "online.mode", "full")),
        },
        "hardware": {
            "offline_gpu": "L4",
            "online_gpu": "L4",
            "note": "Modal runtime configured for L4 GPUs; monitor VRAM for full SD/Co-DETR runs.",
        },
        "latency_mode": latency_mode,
    }


def _build_metrics_note(config: Dict[str, Any]) -> str:
    online_mode = str(get_config_value(config, "online.mode", "full"))
    if online_mode == "openclip_user_query_baseline":
        return (
            "Baseline latency includes only OpenCLIP H/14 text encoding and Zilliz vector search. "
            "It excludes Gemini, Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "openclip_llm_scene_rewrite_baseline":
        return (
            "Baseline latency includes Gemini query-only scene/object/event rewrite, OpenCLIP H/14 text encoding of "
            "the rewritten query, and Zilliz vector search. Gemini only sees the original text query, not the source "
            "video or ground-truth frames. It excludes Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "beit3_user_query_baseline":
        return (
            "Baseline latency includes only BEiT-3 text encoding and Zilliz vector search. "
            "It excludes OpenCLIP, Gemini, Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "textual_query_baseline":
        return (
            "Baseline latency includes BEiT-3 and OpenCLIP H/14 text encoding, Zilliz vector search, and RRF fusion. "
            "It excludes Gemini, Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "beit3_video_mean_top3_baseline":
        return (
            "Baseline latency includes BEiT-3 text encoding, Zilliz top-depth retrieval, video-level mean-top3 score "
            "aggregation, and keyframe selection from ranked videos. It excludes OpenCLIP, Gemini, Stable Diffusion, "
            "Object Filter, UI and video rendering."
        )
    if online_mode == "beit3_window_mean_top3_baseline":
        return (
            "Baseline latency includes BEiT-3 text encoding, Zilliz top-depth retrieval, sliding-window mean-top3 "
            "score aggregation, and keyframe selection from ranked windows/videos. It excludes OpenCLIP, Gemini, "
            "Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "beit3_video_xpool_softmax_heuristic_baseline":
        return (
            "Prototype latency includes BEiT-3 text encoding, online Zilliz keyframe-vector loading, video-level "
            "mean-pool candidate generation, no-train query-conditioned softmax pooling for video reranking, and "
            "BEiT-3 query-keyframe similarity search inside each ranked video. The final ranking is video-first. "
            "It excludes OpenCLIP, Gemini, Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "beit3_video_xpool_heuristic_baseline":
        return (
            "Prototype latency includes BEiT-3 text encoding, online Zilliz keyframe-vector loading, video-level "
            "mean-pool candidate generation, X-Pool-style transformer query-conditioned pooling for video reranking, "
            "and BEiT-3 query-keyframe similarity search inside each ranked video. The final ranking is video-first. "
            "It excludes OpenCLIP, Gemini, Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "beit3_video_xpool_paperlike_baseline":
        return (
            "Prototype latency includes BEiT-3 text encoding, online Zilliz keyframe-vector loading, uniform sampling "
            "of existing keyframes into fixed-frame video slots, batched X-Pool-style transformer "
            "query-conditioned pooling over all videos, and BEiT-3 query-keyframe similarity search inside each "
            "ranked video. The final ranking is video-first and does not use mean-pool candidate pruning. It excludes "
            "OpenCLIP, Gemini, Stable Diffusion, Object Filter, UI and video rendering."
        )
    if online_mode == "beit3_llm_video_mean_top3_baseline":
        return (
            "Baseline latency includes Gemini paraphrase expansion, BEiT-3 text encoding over original query plus "
            "paraphrases, Zilliz top-depth retrieval, video-level mean-top3 score aggregation, and keyframe selection "
            "from ranked videos. It excludes OpenCLIP, Stable Diffusion, Object Filter, UI and video rendering."
        )
    object_filter_enabled = bool(get_config_value(config, "object_filter.enabled", True))
    object_filter_note = "with Object Filter" if object_filter_enabled else "without Object Filter"
    return (
        f"Latency includes Gemini, SD, Zilliz retrieval and fusion, {object_filter_note}. "
        "It excludes UI/video rendering."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latency-mode", default=None, choices=["cache_miss_full_online", "cache_hit_retrieval_only"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--disable-object-filter", action="store_true")
    parser.add_argument(
        "--online-mode",
        default=None,
        choices=[
            "full",
            "openclip_user_query_baseline",
            "openclip_llm_scene_rewrite_baseline",
            "beit3_user_query_baseline",
            "textual_query_baseline",
            "beit3_video_mean_top3_baseline",
            "beit3_window_mean_top3_baseline",
            "beit3_video_xpool_softmax_heuristic_baseline",
            "beit3_video_xpool_heuristic_baseline",
            "beit3_video_xpool_paperlike_baseline",
            "beit3_llm_video_mean_top3_baseline",
        ],
    )
    parser.add_argument("--baseline-openclip-only", action="store_true")
    parser.add_argument("--require-verified-gt-video", action="store_true")
    parser.add_argument("--require-gt-span-keyframe", action="store_true")
    parser.add_argument("--video-aggregation-global-depth", type=int, default=None)
    parser.add_argument("--video-aggregation-keyframes-per-video", type=int, default=None)
    parser.add_argument("--video-score-strategy", default=None)
    parser.add_argument("--video-score-top-k", type=int, default=None)
    parser.add_argument("--window-size-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=None)
    parser.add_argument("--window-score-strategy", default=None)
    parser.add_argument("--window-score-top-k", type=int, default=None)
    parser.add_argument("--keyframes-per-window", type=int, default=None)
    parser.add_argument("--max-windows-per-video", type=int, default=None)
    parser.add_argument("--xpool-candidate-videos", type=int, default=None)
    parser.add_argument("--xpool-keyframes-per-video", type=int, default=None)
    parser.add_argument("--xpool-temperature", type=float, default=None)
    parser.add_argument("--xpool-num-heads", type=int, default=None)
    parser.add_argument("--xpool-dropout", type=float, default=None)
    parser.add_argument("--xpool-checkpoint", default=None)
    parser.add_argument("--xpool-paperlike-num-frames", type=int, default=None)
    parser.add_argument("--xpool-paperlike-keyframes-per-video", type=int, default=None)
    parser.add_argument("--xpool-paperlike-batch-size", type=int, default=None)
    parser.add_argument("--xpool-paperlike-num-heads", type=int, default=None)
    parser.add_argument("--xpool-paperlike-dropout", type=float, default=None)
    parser.add_argument("--xpool-paperlike-checkpoint", default=None)
    args = parser.parse_args()
    result = run_online_evaluation(
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        limit=args.limit,
        offset=args.offset,
        latency_mode=args.latency_mode,
        disable_object_filter=args.disable_object_filter,
        online_mode=args.online_mode,
        baseline_openclip_only=args.baseline_openclip_only,
        require_verified_gt_video=args.require_verified_gt_video,
        require_gt_span_keyframe=args.require_gt_span_keyframe,
        video_aggregation_global_depth=args.video_aggregation_global_depth,
        video_aggregation_keyframes_per_video=args.video_aggregation_keyframes_per_video,
        video_score_strategy=args.video_score_strategy,
        video_score_top_k=args.video_score_top_k,
        window_size_seconds=args.window_size_seconds,
        window_stride_seconds=args.window_stride_seconds,
        window_score_strategy=args.window_score_strategy,
        window_score_top_k=args.window_score_top_k,
        keyframes_per_window=args.keyframes_per_window,
        max_windows_per_video=args.max_windows_per_video,
        xpool_candidate_videos=args.xpool_candidate_videos,
        xpool_keyframes_per_video=args.xpool_keyframes_per_video,
        xpool_temperature=args.xpool_temperature,
        xpool_num_heads=args.xpool_num_heads,
        xpool_dropout=args.xpool_dropout,
        xpool_checkpoint=args.xpool_checkpoint,
        xpool_paperlike_num_frames=args.xpool_paperlike_num_frames,
        xpool_paperlike_keyframes_per_video=args.xpool_paperlike_keyframes_per_video,
        xpool_paperlike_batch_size=args.xpool_paperlike_batch_size,
        xpool_paperlike_num_heads=args.xpool_paperlike_num_heads,
        xpool_paperlike_dropout=args.xpool_paperlike_dropout,
        xpool_paperlike_checkpoint=args.xpool_paperlike_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
