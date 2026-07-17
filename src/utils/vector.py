from __future__ import annotations

from typing import Iterable, List, Sequence


def l2_normalize_vector(vector: Sequence[float], eps: float = 1e-12) -> List[float]:
    norm = sum(float(x) * float(x) for x in vector) ** 0.5
    if norm < eps:
        return [0.0 for _ in vector]
    return [float(x) / norm for x in vector]


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a_norm = l2_normalize_vector(a)
    b_norm = l2_normalize_vector(b)
    similarity = sum(x * y for x, y in zip(a_norm, b_norm))
    return 1.0 - similarity


def l2_normalize_matrix(vectors: Iterable[Sequence[float]]) -> List[List[float]]:
    return [l2_normalize_vector(vector) for vector in vectors]

