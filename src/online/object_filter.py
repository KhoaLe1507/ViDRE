from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from src.models.gemini_client import normalize_coco_constraints
from src.schemas import RetrievalCandidate


def apply_object_filter(
    candidates: Sequence[RetrievalCandidate],
    constraints: Sequence[dict],
    object_counts_by_keyframe: Dict[str, Dict[str, int]] | None = None,
    backfill: bool = False,
) -> List[RetrievalCandidate]:
    valid_constraints = normalize_coco_constraints(list(constraints))
    if not valid_constraints:
        return list(candidates)

    counts_by_keyframe = object_counts_by_keyframe or {}
    kept: List[RetrievalCandidate] = []
    for candidate in candidates:
        counts = dict(candidate.object_counts or counts_by_keyframe.get(candidate.keyframe_id, {}) or {})
        candidate.object_counts = counts
        if _satisfies(counts, valid_constraints):
            kept.append(candidate)
    # Default spec behavior: do not backfill filtered items.
    _ = backfill
    for rank, candidate in enumerate(kept, start=1):
        candidate.rank = rank
    return kept


def _satisfies(counts: Dict[str, int], constraints: Sequence[dict]) -> bool:
    for constraint in constraints:
        name = str(constraint["name"])
        target = int(constraint["count"])
        if int(counts.get(name, 0)) < target:
            return False
    return True

