from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

from src.schemas import RetrievalCandidate


def rrf_fuse(
    ranked_lists: Sequence[Sequence[RetrievalCandidate]],
    rrf_k: int = 60,
    top_n: int = 200,
) -> List[RetrievalCandidate]:
    scores: Dict[str, float] = {}
    best_candidate: Dict[str, RetrievalCandidate] = {}
    best_rank: Dict[str, int] = {}

    for ranked_list in ranked_lists:
        seen_in_list = set()
        for rank, candidate in enumerate(ranked_list, start=1):
            if candidate.keyframe_id in seen_in_list:
                continue
            seen_in_list.add(candidate.keyframe_id)
            scores[candidate.keyframe_id] = scores.get(candidate.keyframe_id, 0.0) + 1.0 / (rrf_k + rank)
            current_best = best_rank.get(candidate.keyframe_id, math.inf)
            if rank < current_best:
                best_candidate[candidate.keyframe_id] = candidate
                best_rank[candidate.keyframe_id] = rank

    return _materialize(scores, best_candidate, top_n=top_n)


def weighted_rrf_fuse(
    branch_lists: Mapping[str, Sequence[RetrievalCandidate]],
    branch_weights: Mapping[str, float],
    rrf_k: int = 60,
    top_n: int = 500,
    textual_branch_name: str = "textual_query",
) -> List[RetrievalCandidate]:
    scores: Dict[str, float] = {}
    best_candidate: Dict[str, RetrievalCandidate] = {}
    textual_ranks: Dict[str, int] = {}

    for branch_name, ranked_list in branch_lists.items():
        weight = float(branch_weights.get(branch_name, 0.0))
        if weight <= 0:
            continue
        seen_in_branch = set()
        for rank, candidate in enumerate(ranked_list, start=1):
            if candidate.keyframe_id in seen_in_branch:
                continue
            seen_in_branch.add(candidate.keyframe_id)
            scores[candidate.keyframe_id] = scores.get(candidate.keyframe_id, 0.0) + weight / (rrf_k + rank)
            best_candidate.setdefault(candidate.keyframe_id, candidate)
            if branch_name == textual_branch_name:
                textual_ranks[candidate.keyframe_id] = rank

    fused = _materialize(scores, best_candidate, top_n=10**9, textual_ranks=textual_ranks)
    return fused[:top_n]


def assign_ranks(candidates: Sequence[RetrievalCandidate]) -> List[RetrievalCandidate]:
    ranked: List[RetrievalCandidate] = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
        ranked.append(candidate)
    return ranked


def _materialize(
    scores: Mapping[str, float],
    best_candidate: Mapping[str, RetrievalCandidate],
    top_n: int,
    textual_ranks: Mapping[str, int] | None = None,
) -> List[RetrievalCandidate]:
    textual_ranks = textual_ranks or {}
    ordered_ids = sorted(
        scores.keys(),
        key=lambda key: (-scores[key], textual_ranks.get(key, math.inf), key),
    )
    fused: List[RetrievalCandidate] = []
    for key in ordered_ids[:top_n]:
        source = best_candidate[key]
        fused.append(
            RetrievalCandidate(
                keyframe_id=source.keyframe_id,
                video_id=source.video_id,
                shot_id=source.shot_id,
                timestamp_raw=source.timestamp_raw,
                frame_index_raw=source.frame_index_raw,
                score=float(scores[key]),
                rank=len(fused) + 1,
                object_counts=dict(source.object_counts),
                metadata=dict(source.metadata),
            )
        )
    return fused

