from __future__ import annotations

from typing import Any, Dict, List

from src.models.gemini_client import GeminiClient
from src.models.load_beit3 import BEiT3Encoder
from src.models.load_openclip import OpenCLIPH14Encoder
from src.models.load_stable_diffusion import StableDiffusionGenerator
from src.online.branches.llm_expansion_branch import LLMExpansionBranch
from src.online.branches.stable_diffusion_branch import StableDiffusionBranch
from src.online.branches.textual_branch import TextualQueryBranch
from src.online.fusion import assign_ranks, weighted_rrf_fuse
from src.online.latency import LatencyTracker
from src.online.object_filter import apply_object_filter
from src.schemas import RetrievalCandidate, SearchResponse
from src.storage.cockroach_client import CockroachClient
from src.storage.zilliz_client import ZillizClient
from src.utils.config import get_config_value, load_config
from src.utils.hashing import query_cache_key
from src.utils.logging import setup_logging


class SearchPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = str(get_config_value(config, "online.mode", "full"))
        self.zilliz = ZillizClient(config)
        self.cockroach = CockroachClient(config)
        self.beit3 = None
        self.openclip = None
        self.gemini = None
        self.sd_generator = None

        supported_modes = {
            "full",
            "openclip_user_query_baseline",
            "beit3_user_query_baseline",
            "textual_query_baseline",
            "beit3_video_mean_top3_baseline",
        }
        if self.mode not in supported_modes:
            raise ValueError(f"Unsupported online.mode={self.mode!r}. Expected one of {sorted(supported_modes)}.")

        if self.mode in {"full", "openclip_user_query_baseline", "textual_query_baseline"}:
            self.openclip = OpenCLIPH14Encoder(config).load()
        if self.mode in {"full", "beit3_user_query_baseline", "textual_query_baseline", "beit3_video_mean_top3_baseline"}:
            self.beit3 = BEiT3Encoder(config).load()
        if self.mode == "full":
            self.gemini = GeminiClient(config)
            self.sd_generator = StableDiffusionGenerator(config)

    def search(self, query: str, latency_mode: str | None = None, output_depth: int | None = None) -> SearchResponse:
        latency_mode = latency_mode or get_config_value(self.config, "evaluation.latency_mode", "cache_miss_full_online")
        if self.mode == "openclip_user_query_baseline":
            assert self.openclip is not None
            return self._search_openclip_user_query(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "beit3_user_query_baseline":
            assert self.beit3 is not None
            return self._search_beit3_user_query(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "textual_query_baseline":
            assert self.beit3 is not None and self.openclip is not None
            return self._search_textual_query(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "beit3_video_mean_top3_baseline":
            assert self.beit3 is not None
            return self._search_beit3_video_mean_top3(query, latency_mode=latency_mode, output_depth=output_depth)

        assert self.beit3 is not None and self.openclip is not None and self.gemini is not None and self.sd_generator is not None
        tracker = LatencyTracker()
        branch_debug: Dict[str, Any] = {}
        branch_lists: Dict[str, Any] = {}

        with tracker.measure("textual_query_branch"):
            textual = TextualQueryBranch(self.config, self.beit3, self.openclip, self.zilliz).run(query)
            branch_lists["textual_query"] = textual

        with tracker.measure("llm_query_expansion_branch"):
            llm_results, llm_debug = LLMExpansionBranch(
                self.config, self.beit3, self.openclip, self.zilliz, self.cockroach, self.gemini
            ).run(query, latency_mode=latency_mode)
            branch_lists["llm_query_expansion"] = llm_results
            branch_debug["llm_query_expansion"] = llm_debug

        with tracker.measure("stable_diffusion_branch"):
            sd_results, sd_debug = StableDiffusionBranch(
                self.config, self.beit3, self.openclip, self.zilliz, self.cockroach, self.sd_generator
            ).run(query, latency_mode=latency_mode)
            branch_lists["stable_diffusion"] = sd_results
            branch_debug["stable_diffusion"] = sd_debug

        with tracker.measure("weighted_rrf"):
            fused = weighted_rrf_fuse(
                branch_lists,
                branch_weights=get_config_value(self.config, "weighted_rrf.branch_weights", {}),
                rrf_k=int(get_config_value(self.config, "weighted_rrf.rrf_k", 60)),
                top_n=int(get_config_value(self.config, "weighted_rrf.post_fusion_depth", 500)),
            )

        with tracker.measure("object_filter"):
            object_filter_enabled = bool(get_config_value(self.config, "object_filter.enabled", True))
            branch_debug["object_filter_enabled"] = object_filter_enabled
            if object_filter_enabled:
                constraints = self._get_object_constraints(query, latency_mode)
                branch_debug["object_constraints"] = constraints
            else:
                constraints = []
                branch_debug["object_constraints"] = []
            if object_filter_enabled and constraints:
                counts = self.cockroach.get_object_counts([candidate.keyframe_id for candidate in fused])
            else:
                counts = {}
            if object_filter_enabled:
                filtered = apply_object_filter(
                    fused,
                    constraints,
                    object_counts_by_keyframe=counts,
                    backfill=bool(get_config_value(self.config, "object_filter.backfill", False)),
                )
            else:
                filtered = fused

        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        final_results = assign_ranks(filtered[:final_depth])
        return SearchResponse(
            query=query,
            results=final_results,
            latency_ms=tracker.total_ms,
            latency_breakdown_ms=tracker.breakdown_ms,
            branch_debug=branch_debug,
        )

    def _search_openclip_user_query(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        with tracker.measure("openclip_user_query_branch"):
            openclip_vector = self.openclip.encode_texts([query])[0]
            results = self.zilliz.search(self.zilliz.openclip_field, openclip_vector, final_depth)
        final_results = assign_ranks(results[:final_depth])
        return SearchResponse(
            query=query,
            results=final_results,
            latency_ms=tracker.total_ms,
            latency_breakdown_ms=tracker.breakdown_ms,
            branch_debug={
                "online_mode": self.mode,
                "latency_mode": latency_mode,
                "vector_field": self.zilliz.openclip_field,
                "object_filter_enabled": False,
                "branches": ["openclip_user_query"],
            },
        )

    def _search_beit3_user_query(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        with tracker.measure("beit3_user_query_branch"):
            beit3_vector = self.beit3.encode_texts([query])[0]
            results = self.zilliz.search(self.zilliz.beit3_field, beit3_vector, final_depth)
        final_results = assign_ranks(results[:final_depth])
        return SearchResponse(
            query=query,
            results=final_results,
            latency_ms=tracker.total_ms,
            latency_breakdown_ms=tracker.breakdown_ms,
            branch_debug={
                "online_mode": self.mode,
                "latency_mode": latency_mode,
                "vector_field": self.zilliz.beit3_field,
                "object_filter_enabled": False,
                "branches": ["beit3_user_query"],
            },
        )

    def _search_textual_query(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None and self.openclip is not None
        tracker = LatencyTracker()
        with tracker.measure("textual_query_branch"):
            results = TextualQueryBranch(self.config, self.beit3, self.openclip, self.zilliz).run(query)
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        final_results = assign_ranks(results[:final_depth])
        return SearchResponse(
            query=query,
            results=final_results,
            latency_ms=tracker.total_ms,
            latency_breakdown_ms=tracker.breakdown_ms,
            branch_debug={
                "online_mode": self.mode,
                "latency_mode": latency_mode,
                "object_filter_enabled": False,
                "branches": ["textual_query"],
            },
        )

    def _search_beit3_video_mean_top3(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        global_depth = int(get_config_value(self.config, "video_aggregation.global_depth", 1000))
        keyframes_per_video = int(get_config_value(self.config, "video_aggregation.keyframes_per_video", 1))
        keyframes_per_video = max(1, keyframes_per_video)

        with tracker.measure("beit3_video_mean_top3_branch"):
            beit3_vector = self.beit3.encode_texts([query])[0]
            global_results = self.zilliz.search(self.zilliz.beit3_field, beit3_vector, global_depth)
            grouped = _group_candidates_by_video(global_results)
            ranked_videos = sorted(
                grouped.items(),
                key=lambda item: (-_mean_top_k_score(item[1], k=3), item[0]),
            )
            final_results: List[RetrievalCandidate] = []
            for _, candidates in ranked_videos:
                final_results.extend(candidates[:keyframes_per_video])
                if len(final_results) >= final_depth:
                    break

        return SearchResponse(
            query=query,
            results=assign_ranks(final_results[:final_depth]),
            latency_ms=tracker.total_ms,
            latency_breakdown_ms=tracker.breakdown_ms,
            branch_debug={
                "online_mode": self.mode,
                "latency_mode": latency_mode,
                "vector_field": self.zilliz.beit3_field,
                "object_filter_enabled": False,
                "global_depth": global_depth,
                "keyframes_per_video": keyframes_per_video,
                "video_score_strategy": "mean_top3_score",
                "branches": ["beit3_video_mean_top3"],
            },
        )

    def _get_object_constraints(self, query: str, latency_mode: str) -> List[Dict[str, int]]:
        cache_version = self.config["project"]["config_version"]
        cache_key = query_cache_key(query, cache_version)
        cache = self.cockroach.get_query_cache(cache_key, cache_version)
        constraints = list((cache or {}).get("gemini_object_constraints") or [])
        if constraints:
            return constraints
        if latency_mode == "cache_hit_retrieval_only":
            return []
        constraints = self.gemini.extract_object_constraints(query)
        existing = cache or {}
        self.cockroach.upsert_query_cache(
            query_hash=cache_key,
            query_text=query,
            cache_version=cache_version,
            gemini_paraphrases=existing.get("gemini_paraphrases") or [],
            gemini_object_constraints=constraints,
            sd_prompt=existing.get("sd_prompt"),
            sd_image_paths_or_hashes=existing.get("sd_image_paths_or_hashes") or [],
            sd_seeds=existing.get("sd_seeds") or [],
        )
        return constraints


def _group_candidates_by_video(candidates: List[RetrievalCandidate]) -> Dict[str, List[RetrievalCandidate]]:
    grouped: Dict[str, List[RetrievalCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.video_id, []).append(candidate)
    return grouped


def _mean_top_k_score(candidates: List[RetrievalCandidate], k: int = 3) -> float:
    scores = sorted((float(candidate.score) for candidate in candidates), reverse=True)[:k]
    return sum(scores) / len(scores) if scores else float("-inf")


def search_single_query(
    query: str,
    config_path: str | None = None,
    latency_mode: str | None = None,
    disable_object_filter: bool = False,
    online_mode: str | None = None,
    baseline_openclip_only: bool = False,
) -> Dict[str, Any]:
    setup_logging()
    overrides: Dict[str, Any] = {}
    if disable_object_filter:
        overrides["object_filter"] = {"enabled": False}
    if online_mode:
        overrides["online"] = {"mode": online_mode}
        if online_mode in {
            "openclip_user_query_baseline",
            "beit3_user_query_baseline",
            "textual_query_baseline",
            "beit3_video_mean_top3_baseline",
        }:
            overrides["object_filter"] = {"enabled": False}
    if baseline_openclip_only:
        overrides["online"] = {"mode": "openclip_user_query_baseline"}
        overrides["object_filter"] = {"enabled": False}
    if not overrides:
        overrides = None
    pipeline = SearchPipeline(load_config(config_path, overrides=overrides))
    return pipeline.search(query, latency_mode=latency_mode).to_dict()
