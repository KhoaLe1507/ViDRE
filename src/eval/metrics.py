from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from src.schemas import RetrievalCandidate


def is_correct_keyframe(candidate: RetrievalCandidate | dict, gt_video_id: str, gt_start: float, gt_end: float) -> bool:
    if isinstance(candidate, RetrievalCandidate):
        video_id = candidate.video_id
        timestamp_raw = candidate.timestamp_raw
    else:
        video_id = str(candidate.get("video_id"))
        timestamp_raw = float(candidate.get("timestamp_raw", -1.0))
    return video_id == gt_video_id and gt_start <= timestamp_raw <= gt_end


def is_correct_video(candidate: RetrievalCandidate | dict, gt_video_id: str) -> bool:
    if isinstance(candidate, RetrievalCandidate):
        video_id = candidate.video_id
    else:
        video_id = str(candidate.get("video_id"))
    return video_id == gt_video_id


def compute_hits(candidates: Sequence[RetrievalCandidate], gt_video_id: str, gt_span: Sequence[float]) -> Dict[str, bool]:
    gt_start, gt_end = float(gt_span[0]), float(gt_span[1])
    return {
        "hit_at_1": any(is_correct_keyframe(candidate, gt_video_id, gt_start, gt_end) for candidate in candidates[:1]),
        "hit_at_5": any(is_correct_keyframe(candidate, gt_video_id, gt_start, gt_end) for candidate in candidates[:5]),
        "hit_at_10": any(is_correct_keyframe(candidate, gt_video_id, gt_start, gt_end) for candidate in candidates[:10]),
        "video_hit_at_1": any(is_correct_video(candidate, gt_video_id) for candidate in candidates[:1]),
        "video_hit_at_5": any(is_correct_video(candidate, gt_video_id) for candidate in candidates[:5]),
        "video_hit_at_10": any(is_correct_video(candidate, gt_video_id) for candidate in candidates[:10]),
    }


def aggregate_metrics(per_query_results: Iterable[dict]) -> Dict[str, float]:
    rows = list(per_query_results)
    n = len(rows)
    if n == 0:
        return {
            "num_queries": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "video_recall_at_1": 0.0,
            "video_recall_at_5": 0.0,
            "video_recall_at_10": 0.0,
            "mean_time_latency_ms": 0.0,
        }
    return {
        "num_queries": n,
        "recall_at_1": sum(1 for row in rows if row.get("hit_at_1")) / n,
        "recall_at_5": sum(1 for row in rows if row.get("hit_at_5")) / n,
        "recall_at_10": sum(1 for row in rows if row.get("hit_at_10")) / n,
        "video_recall_at_1": sum(1 for row in rows if row.get("video_hit_at_1")) / n,
        "video_recall_at_5": sum(1 for row in rows if row.get("video_hit_at_5")) / n,
        "video_recall_at_10": sum(1 for row in rows if row.get("video_hit_at_10")) / n,
        "mean_time_latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in rows) / n,
    }
