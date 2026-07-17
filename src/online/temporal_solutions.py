from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from src.eval.export_results import write_manifest, write_metrics, write_per_query_results
from src.eval.metrics import aggregate_metrics, compute_hits
from src.eval.run_evaluation import load_query_samples
from src.models.load_beit3 import BEiT3Encoder
from src.models.load_openclip import OpenCLIPH14Encoder
from src.online.fusion import assign_ranks
from src.schemas import QuerySample, RetrievalCandidate
from src.storage.zilliz_client import ZillizClient
from src.utils.config import get_config_value, load_config
from src.utils.logging import get_logger, setup_logging


logger = get_logger(__name__)

TEMPORAL_DATASET_PATH = "/data/TimeLens-Bench/charades-timelens-query-samples-temporal-scene-moment-first1000.json"
SUPPORTED_SOLUTIONS = {"fusionista", "exquisitor", "viewsinsight"}


def run_temporal_solution_evaluation(
    solution: str,
    config_path: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    output_depth: int = 10,
    llm_timeout_seconds: int = 90,
    checkpoint_every: int = 10,
    max_subqueries: int = 4,
    exquisitor_initial_depth: int = 1000,
) -> Dict[str, Any]:
    setup_logging()
    solution = _normalize_solution(solution)
    config = load_config(config_path)
    dataset_path = dataset_path or TEMPORAL_DATASET_PATH
    output_root = Path(output_dir or get_config_value(config, "evaluation.output_dir", "outputs/eval"))
    run_id = _make_temporal_run_id(solution)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = load_query_samples(dataset_path)
    if offset:
        samples = samples[int(offset) :]
    if limit is not None:
        samples = samples[: int(limit)]

    cache_path = run_dir / f"{solution}_llm_cache.json"
    llm = TemporalQueryDecomposer(config, solution=solution, cache_path=cache_path, timeout_seconds=llm_timeout_seconds)
    engine = TemporalSolutionSearchEngine(
        config,
        solution=solution,
        output_depth=output_depth,
        max_subqueries=max_subqueries,
        exquisitor_initial_depth=exquisitor_initial_depth,
    )

    per_query: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        logger.info("temporal_eval_query_start solution=%s index=%d total=%d query_id=%s", solution, index, len(samples), sample.query_id)
        start = time.perf_counter()
        try:
            plan = llm.decompose(sample.query_text)
            results = engine.search(sample.query_text, plan)
            latency_ms = (time.perf_counter() - start) * 1000.0
            hits = compute_hits(results, sample.video_id, sample.gt_span)
            row = {
                "query_id": sample.query_id,
                "query_text": sample.query_text,
                "gt_video_id": sample.video_id,
                "gt_span": sample.gt_span,
                "solution": solution,
                "decomposition": plan,
                "top_retrieved_keyframes": [candidate.to_eval_dict() for candidate in results],
                **hits,
                "latency_ms": latency_ms,
                "error": None,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.exception("temporal_eval_query_failed solution=%s query_id=%s", solution, sample.query_id)
            row = _error_row(solution, sample, latency_ms, exc)
        per_query.append(row)
        if index % max(1, int(checkpoint_every)) == 0:
            _write_temporal_outputs(run_dir, solution, config, dataset_path, per_query, partial=True)

    return _write_temporal_outputs(run_dir, solution, config, dataset_path, per_query, partial=False)


class TemporalQueryDecomposer:
    def __init__(self, config: Dict[str, Any], solution: str, cache_path: Path, timeout_seconds: int = 90):
        self.config = config
        self.solution = solution
        self.cache_path = cache_path
        self.timeout_seconds = max(10, int(timeout_seconds))
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path.exists():
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.cache = {str(key): dict(value) for key, value in payload.items()}
        self.model_name = os.environ.get(get_config_value(config, "models.gemini.model_env", "GEMINI_MODEL"), "")
        self.api_key = os.environ.get(get_config_value(config, "models.gemini.api_key_env", "GEMINI_API_KEY"), "")
        if not self.model_name or not self.api_key:
            raise RuntimeError("GEMINI_API_KEY and GEMINI_MODEL must be set.")
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def decompose(self, query: str) -> Dict[str, Any]:
        cache_key = _cache_key(self.solution, query)
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        prompt = _decomposition_prompt(self.solution, query)
        response = self.model.generate_content(prompt, request_options={"timeout": self.timeout_seconds})
        plan = _parse_json_object(response.text or "{}")
        plan = _normalize_decomposition(self.solution, query, plan)
        self.cache[cache_key] = plan
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return plan


class TemporalSolutionSearchEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        solution: str,
        output_depth: int = 10,
        max_subqueries: int = 4,
        exquisitor_initial_depth: int = 1000,
    ):
        self.config = config
        self.solution = _normalize_solution(solution)
        self.output_depth = max(1, int(output_depth))
        self.max_subqueries = max(2, int(max_subqueries))
        self.exquisitor_initial_depth = max(1, int(exquisitor_initial_depth))
        self.zilliz = ZillizClient(config)
        self.beit3: BEiT3Encoder | None = None
        self.openclip: OpenCLIPH14Encoder | None = None
        if self.solution == "exquisitor":
            self.beit3 = BEiT3Encoder(config).load()
            self.index = self._load_index(self.zilliz.beit3_field)
        else:
            self.openclip = OpenCLIPH14Encoder(config).load()
            self.index = self._load_index(self.zilliz.openclip_field)

    def search(self, original_query: str, plan: Dict[str, Any]) -> List[RetrievalCandidate]:
        if self.solution == "fusionista":
            return self._search_fusionista(plan)
        if self.solution == "exquisitor":
            return self._search_exquisitor(plan)
        if self.solution == "viewsinsight":
            return self._search_viewsinsight(plan)
        raise ValueError(f"Unsupported temporal solution: {self.solution}")

    def _load_index(self, vector_field: str) -> Dict[str, Any]:
        model_version = str(self.config["project"]["model_version"])
        config_version = str(self.config["project"]["config_version"])
        filter_expr = f'model_version == "{model_version}" and config_version == "{config_version}"'
        pairs = self.zilliz.fetch_keyframe_vectors(
            vector_field,
            filter_expr=filter_expr,
            batch_size=int(get_config_value(self.config, "zilliz.query_batch_size", 4096)),
        )
        candidates = [candidate for candidate, _ in pairs]
        vectors = _l2_normalize_matrix([vector for _, vector in pairs])
        videos: Dict[str, List[int]] = {}
        for index, candidate in enumerate(candidates):
            videos.setdefault(candidate.video_id, []).append(index)
        for indices in videos.values():
            indices.sort(key=lambda idx: (float(candidates[idx].timestamp_raw), int(candidates[idx].frame_index_raw), candidates[idx].keyframe_id))
        return {
            "field": vector_field,
            "candidates": candidates,
            "vectors": vectors,
            "videos": videos,
        }

    def _encode(self, queries: Sequence[str]) -> np.ndarray:
        clean = [str(query).strip() for query in queries if str(query).strip()]
        if not clean:
            raise ValueError("At least one sub-query is required.")
        clean = clean[: self.max_subqueries]
        if self.solution == "exquisitor":
            assert self.beit3 is not None
            vectors = self.beit3.encode_texts(clean)
        else:
            assert self.openclip is not None
            vectors = self.openclip.encode_texts(clean)
        return _l2_normalize_matrix(vectors)

    def _search_fusionista(self, plan: Dict[str, Any]) -> List[RetrievalCandidate]:
        queries = _ordered_queries_from_plan(plan, self.max_subqueries)
        query_vectors = self._encode(queries)
        frame_vectors = self.index["vectors"]
        candidates = self.index["candidates"]
        sims = frame_vectors @ query_vectors.T
        chains = []
        for video_id, indices in self.index["videos"].items():
            if len(indices) < len(queries):
                continue
            video_scores = sims[indices, :]
            chain = _best_ordered_chain_dp(video_scores)
            if not chain:
                continue
            global_indices = [indices[position] for position in chain["positions"]]
            score = float(chain["score"] / max(1, len(queries)))
            chains.append({"video_id": video_id, "indices": global_indices, "score": score})
        chains.sort(key=lambda item: (-float(item["score"]), str(item["video_id"])))
        return assign_ranks(_chain_representatives(chains, candidates, "fusionista", self.output_depth))

    def _search_exquisitor(self, plan: Dict[str, Any]) -> List[RetrievalCandidate]:
        queries = _ordered_queries_from_plan(plan, self.max_subqueries)
        query_vectors = self._encode(queries)
        frame_vectors = self.index["vectors"]
        candidates = self.index["candidates"]
        sims = frame_vectors @ query_vectors.T
        first_order = np.argsort(-sims[:, 0], kind="mergesort")[: self.exquisitor_initial_depth]
        pool = set()
        for frame_index in first_order.tolist():
            candidate = candidates[int(frame_index)]
            for idx in self.index["videos"].get(candidate.video_id, []):
                if candidates[idx].timestamp_raw > candidate.timestamp_raw:
                    pool.add(idx)
        pool_indices = sorted(pool)
        query_ranks: Dict[int, Dict[int, int]] = {}
        for query_idx in range(1, len(queries)):
            ranked_pool = sorted(pool_indices, key=lambda idx: (-float(sims[idx, query_idx]), candidates[idx].keyframe_id))
            query_ranks[query_idx] = {idx: rank for rank, idx in enumerate(ranked_pool, start=1)}

        best_by_video: Dict[str, Dict[str, Any]] = {}
        for first_rank, start_idx in enumerate(first_order.tolist(), start=1):
            start_idx = int(start_idx)
            start = candidates[start_idx]
            chain = [start_idx]
            ranks = [first_rank]
            previous_time = float(start.timestamp_raw)
            video_indices = self.index["videos"].get(start.video_id, [])
            for query_idx in range(1, len(queries)):
                valid = [
                    idx
                    for idx in video_indices
                    if idx in query_ranks[query_idx] and float(candidates[idx].timestamp_raw) > previous_time
                ]
                if not valid:
                    break
                chosen = min(valid, key=lambda idx: (query_ranks[query_idx][idx], candidates[idx].timestamp_raw, candidates[idx].keyframe_id))
                chain.append(chosen)
                ranks.append(query_ranks[query_idx][chosen])
                previous_time = float(candidates[chosen].timestamp_raw)
            score = _rrf_from_ranks(ranks)
            item = {"video_id": start.video_id, "indices": chain, "score": score, "chain_length": len(chain), "ranks": ranks}
            previous = best_by_video.get(start.video_id)
            if previous is None or (len(chain), score) > (int(previous["chain_length"]), float(previous["score"])):
                best_by_video[start.video_id] = item
        chains = list(best_by_video.values())
        chains.sort(key=lambda item: (-int(item["chain_length"]), -float(item["score"]), str(item["video_id"])))
        return assign_ranks(_chain_representatives(chains, candidates, "exquisitor", self.output_depth))

    def _search_viewsinsight(self, plan: Dict[str, Any]) -> List[RetrievalCandidate]:
        candidates = self.index["candidates"]
        frame_vectors = self.index["vectors"]
        now_queries = _ensure_nonempty_list(plan.get("now_queries") or plan.get("local_context") or plan.get("main_query"))
        variant_queries = _ensure_nonempty_list(plan.get("variants"))[:3]
        now_query_vectors = self._encode((now_queries + variant_queries)[: self.max_subqueries])
        now_scores = frame_vectors @ now_query_vectors.T
        now_score = now_scores.mean(axis=1)

        before_queries = _ensure_nonempty_list(plan.get("before_queries"))
        after_queries = _ensure_nonempty_list(plan.get("after_queries"))
        before_score = None
        after_score = None
        if before_queries:
            before_score = frame_vectors @ self._encode(before_queries[: self.max_subqueries]).T
            before_score = before_score.max(axis=1)
        if after_queries:
            after_score = frame_vectors @ self._encode(after_queries[: self.max_subqueries]).T
            after_score = after_score.max(axis=1)

        ranked = []
        for idx, candidate in enumerate(candidates):
            score_parts = [float(now_score[idx])]
            if before_score is not None:
                before_indices = [
                    other_idx
                    for other_idx in self.index["videos"].get(candidate.video_id, [])
                    if candidates[other_idx].timestamp_raw < candidate.timestamp_raw
                ]
                if before_indices:
                    score_parts.append(float(max(before_score[before_indices])))
                else:
                    score_parts.append(-1.0)
            if after_score is not None:
                after_indices = [
                    other_idx
                    for other_idx in self.index["videos"].get(candidate.video_id, [])
                    if candidates[other_idx].timestamp_raw > candidate.timestamp_raw
                ]
                if after_indices:
                    score_parts.append(float(max(after_score[after_indices])))
                else:
                    score_parts.append(-1.0)
            updated_score = sum(score_parts) / len(score_parts)
            ranked.append((idx, updated_score))
        ranked.sort(key=lambda item: (-float(item[1]), candidates[item[0]].keyframe_id))
        results = []
        for idx, score in ranked[: self.output_depth]:
            source = candidates[int(idx)]
            results.append(_copy_candidate(source, float(score), {"solution": "viewsinsight", "decomposition": plan}))
        return assign_ranks(results)


def _decomposition_prompt(solution: str, query: str) -> str:
    if solution == "fusionista":
        method = (
            "Fusionista handles temporal search as an ordered sequence of short frame-level queries. "
            "Split the input into 2-4 concise sub-queries, each describing one consecutive visual frame/event. "
            "Avoid long clauses; each sub-query should be suitable for CLIP image-text search."
        )
        schema = '{"ordered_queries": ["first visual event", "next visual event"], "reason": "..."}'
    elif solution == "exquisitor":
        method = (
            "Exquisitor handles temporal search as sequence chains. The first sub-query anchors the chain; later "
            "sub-queries must occur strictly after the previous one. Split the input into ordered anchors q1..qN."
        )
        schema = '{"ordered_queries": ["anchor event", "later event"], "reason": "..."}'
    elif solution == "viewsinsight":
        method = (
            "ViewsInsight2.0 uses an automatic query generator to produce before, now, and after contexts, plus "
            "variants of the main event. Extract the central event into now_queries and put temporal context in "
            "before_queries and after_queries only when explicitly supported by the input."
        )
        schema = (
            '{"before_queries": ["event before"], "now_queries": ["central event"], '
            '"after_queries": ["event after"], "variants": ["paraphrase"], "metadata_terms": ["object"], "reason": "..."}'
        )
    else:
        raise ValueError(solution)
    return (
        "You decompose English temporal video retrieval queries for a specific retrieval pipeline.\n"
        f"Pipeline method: {method}\n"
        "Rules:\n"
        "- Preserve the original meaning exactly.\n"
        "- Do not invent new objects, people, locations, colors, or events.\n"
        "- Keep sub-queries visually grounded and concise.\n"
        "- Return valid JSON only.\n\n"
        f"JSON schema example:\n{schema}\n\n"
        f"Original query:\n{query}\n"
    )


def _normalize_decomposition(solution: str, original_query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    if solution in {"fusionista", "exquisitor"}:
        ordered = _ensure_nonempty_list(plan.get("ordered_queries"))
        if len(ordered) < 2:
            ordered = _fallback_ordered_queries(original_query)
        return {
            "solution": solution,
            "ordered_queries": ordered[:4],
            "reason": str(plan.get("reason") or ""),
        }
    before = _ensure_nonempty_list(plan.get("before_queries"))
    now = _ensure_nonempty_list(plan.get("now_queries"))
    after = _ensure_nonempty_list(plan.get("after_queries"))
    variants = _ensure_nonempty_list(plan.get("variants"))
    metadata = _ensure_nonempty_list(plan.get("metadata_terms"))
    if not now:
        now = [original_query]
    return {
        "solution": solution,
        "before_queries": before[:4],
        "now_queries": now[:4],
        "after_queries": after[:4],
        "variants": variants[:4],
        "metadata_terms": metadata[:8],
        "reason": str(plan.get("reason") or ""),
    }


def _best_ordered_chain_dp(scores: np.ndarray) -> Dict[str, Any] | None:
    frames, queries = scores.shape
    if frames < queries:
        return None
    dp = np.full((queries, frames), -np.inf, dtype=np.float32)
    prev = np.full((queries, frames), -1, dtype=np.int64)
    dp[0, :] = scores[:, 0]
    for q in range(1, queries):
        best_score = -np.inf
        best_pos = -1
        for f in range(frames):
            if dp[q - 1, f] > best_score:
                best_score = float(dp[q - 1, f])
                best_pos = f
            if best_pos >= 0:
                dp[q, f] = best_score + scores[f, q]
                prev[q, f] = best_pos
    end_pos = int(np.argmax(dp[queries - 1, :]))
    if not np.isfinite(dp[queries - 1, end_pos]):
        return None
    positions = [end_pos]
    for q in range(queries - 1, 0, -1):
        end_pos = int(prev[q, end_pos])
        if end_pos < 0:
            return None
        positions.append(end_pos)
    positions.reverse()
    return {"positions": positions, "score": float(dp[queries - 1, positions[-1]])}


def _chain_representatives(
    chains: Sequence[Dict[str, Any]],
    candidates: Sequence[RetrievalCandidate],
    solution: str,
    output_depth: int,
) -> List[RetrievalCandidate]:
    results: List[RetrievalCandidate] = []
    for chain_rank, chain in enumerate(chains, start=1):
        indices = list(chain["indices"])
        if not indices:
            continue
        representative = indices[len(indices) // 2]
        source = candidates[int(representative)]
        results.append(
            _copy_candidate(
                source,
                float(chain["score"]),
                {
                    "solution": solution,
                    "chain_rank": chain_rank,
                    "chain_keyframe_ids": [candidates[int(idx)].keyframe_id for idx in indices],
                    "chain_timestamps": [float(candidates[int(idx)].timestamp_raw) for idx in indices],
                    "chain_length": len(indices),
                },
            )
        )
        if len(results) >= output_depth:
            break
    return results


def _write_temporal_outputs(
    run_dir: Path,
    solution: str,
    config: Dict[str, Any],
    dataset_path: str,
    per_query: List[Dict[str, Any]],
    partial: bool,
) -> Dict[str, Any]:
    aggregate = aggregate_metrics(per_query)
    metrics = {
        "solution": solution,
        "num_queries": int(aggregate["num_queries"]),
        "KeyframeRecall@1": float(aggregate["recall_at_1"]),
        "KeyframeRecall@5": float(aggregate["recall_at_5"]),
        "KeyframeRecall@10": float(aggregate["recall_at_10"]),
        "mean_latency_ms": float(aggregate["mean_time_latency_ms"]),
        "partial": bool(partial),
        "model_version": config["project"]["model_version"],
        "config_version": config["project"]["config_version"],
    }
    manifest = {
        "solution": solution,
        "dataset_path": str(dataset_path),
        "notes": _solution_note(solution),
        "metrics": ["KeyframeRecall@1", "KeyframeRecall@5", "KeyframeRecall@10", "mean_latency_ms"],
    }
    write_metrics(run_dir, metrics)
    write_per_query_results(run_dir, per_query)
    write_manifest(run_dir, manifest)
    return {"eval_run_id": run_dir.name, "output_dir": str(run_dir), "metrics": metrics}


def _solution_note(solution: str) -> str:
    if solution == "fusionista":
        return "Fusionista-inspired ordered frame sequence scoring with dynamic programming over per-video keyframes."
    if solution == "exquisitor":
        return "Exquisitor-inspired q1-anchored sequence-chain search with chain length priority and RRF tie-break."
    return "ViewsInsight2.0-inspired before/now/after LLM decomposition with temporal context score updates for now keyframes."


def _error_row(solution: str, sample: QuerySample, latency_ms: float, exc: Exception) -> Dict[str, Any]:
    return {
        "query_id": sample.query_id,
        "query_text": sample.query_text,
        "gt_video_id": sample.video_id,
        "gt_span": sample.gt_span,
        "solution": solution,
        "decomposition": {},
        "top_retrieved_keyframes": [],
        "hit_at_1": False,
        "hit_at_5": False,
        "hit_at_10": False,
        "video_hit_at_1": False,
        "video_hit_at_5": False,
        "video_hit_at_10": False,
        "latency_ms": latency_ms,
        "error": repr(exc),
    }


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def _ensure_nonempty_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    results = []
    seen = set()
    for item in value:
        text = " ".join(str(item).split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            results.append(text)
    return results


def _fallback_ordered_queries(query: str) -> List[str]:
    separators = [" immediately after ", " after ", " before ", " then ", " while ", " during "]
    lowered = query
    for sep in separators:
        if sep in lowered:
            parts = [part.strip(" .,;") for part in lowered.split(sep) if part.strip(" .,;")]
            if len(parts) >= 2:
                return parts[:4]
    return [query, query]


def _ordered_queries_from_plan(plan: Dict[str, Any], max_subqueries: int) -> List[str]:
    ordered = _ensure_nonempty_list(plan.get("ordered_queries"))
    if len(ordered) < 2:
        raise ValueError("Temporal ordered solution requires at least two ordered sub-queries.")
    return ordered[:max_subqueries]


def _copy_candidate(source: RetrievalCandidate, score: float, metadata: Dict[str, Any]) -> RetrievalCandidate:
    merged_metadata = dict(source.metadata or {})
    merged_metadata.update(metadata)
    return RetrievalCandidate(
        keyframe_id=source.keyframe_id,
        video_id=source.video_id,
        shot_id=source.shot_id,
        timestamp_raw=source.timestamp_raw,
        frame_index_raw=source.frame_index_raw,
        score=float(score),
        rank=0,
        object_counts=dict(source.object_counts or {}),
        metadata=merged_metadata,
    )


def _l2_normalize_matrix(vectors: Sequence[Sequence[float]], eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def _rrf_from_ranks(ranks: Iterable[int], k: int = 60) -> float:
    return sum(1.0 / (float(k) + float(rank)) for rank in ranks)


def _cache_key(solution: str, query: str) -> str:
    return hashlib.sha256(f"{solution}\n{query.strip()}".encode("utf-8")).hexdigest()


def _make_temporal_run_id(solution: str) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"temporal_{solution}_{timestamp}"


def _normalize_solution(solution: str) -> str:
    normalized = solution.strip().lower().replace("-", "").replace("_", "").replace(".", "")
    if normalized == "viewsinsight20":
        normalized = "viewsinsight"
    mapping = {
        "fusionista": "fusionista",
        "exquisitor": "exquisitor",
        "viewsinsight": "viewsinsight",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported solution={solution!r}. Expected one of {sorted(SUPPORTED_SOLUTIONS)}.")
    return mapping[normalized]
