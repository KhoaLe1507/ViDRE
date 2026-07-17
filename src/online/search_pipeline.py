from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

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
from src.online.xpool import XPoolTransformer, load_xpool_transformer_checkpoint
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
        self._video_xpool_index_by_field: Dict[str, Dict[str, Any]] = {}
        self._xpool_transformer = None
        self._xpool_transformer_by_key: Dict[str, XPoolTransformer] = {}
        self._xpool_checkpoint_info: Dict[str, Any] = {}
        self._xpool_checkpoint_info_by_key: Dict[str, Dict[str, Any]] = {}

        supported_modes = {
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
        }
        if self.mode not in supported_modes:
            raise ValueError(f"Unsupported online.mode={self.mode!r}. Expected one of {sorted(supported_modes)}.")

        if self.mode in {"full", "openclip_user_query_baseline", "openclip_llm_scene_rewrite_baseline", "textual_query_baseline"}:
            self.openclip = OpenCLIPH14Encoder(config).load()
        if self.mode in {
            "full",
            "beit3_user_query_baseline",
            "textual_query_baseline",
            "beit3_video_mean_top3_baseline",
            "beit3_window_mean_top3_baseline",
            "beit3_video_xpool_softmax_heuristic_baseline",
            "beit3_video_xpool_heuristic_baseline",
            "beit3_video_xpool_paperlike_baseline",
            "beit3_llm_video_mean_top3_baseline",
        }:
            self.beit3 = BEiT3Encoder(config).load()
        if self.mode in {"full", "openclip_llm_scene_rewrite_baseline", "beit3_llm_video_mean_top3_baseline"}:
            self.gemini = GeminiClient(config)
        if self.mode == "full":
            self.sd_generator = StableDiffusionGenerator(config)

    def search(self, query: str, latency_mode: str | None = None, output_depth: int | None = None) -> SearchResponse:
        latency_mode = latency_mode or get_config_value(self.config, "evaluation.latency_mode", "cache_miss_full_online")
        if self.mode == "openclip_user_query_baseline":
            assert self.openclip is not None
            return self._search_openclip_user_query(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "openclip_llm_scene_rewrite_baseline":
            assert self.openclip is not None and self.gemini is not None
            return self._search_openclip_llm_scene_rewrite(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "beit3_user_query_baseline":
            assert self.beit3 is not None
            return self._search_beit3_user_query(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "textual_query_baseline":
            assert self.beit3 is not None and self.openclip is not None
            return self._search_textual_query(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "beit3_video_mean_top3_baseline":
            assert self.beit3 is not None
            return self._search_beit3_video_mean_top3(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "beit3_window_mean_top3_baseline":
            assert self.beit3 is not None
            return self._search_beit3_window_mean_top3(query, latency_mode=latency_mode, output_depth=output_depth)
        if self.mode == "beit3_video_xpool_softmax_heuristic_baseline":
            assert self.beit3 is not None
            return self._search_beit3_video_xpool_softmax_heuristic(
                query,
                latency_mode=latency_mode,
                output_depth=output_depth,
            )
        if self.mode == "beit3_video_xpool_heuristic_baseline":
            assert self.beit3 is not None
            return self._search_beit3_video_xpool_heuristic(
                query,
                latency_mode=latency_mode,
                output_depth=output_depth,
            )
        if self.mode == "beit3_video_xpool_paperlike_baseline":
            assert self.beit3 is not None
            return self._search_beit3_video_xpool_paperlike(
                query,
                latency_mode=latency_mode,
                output_depth=output_depth,
            )
        if self.mode == "beit3_llm_video_mean_top3_baseline":
            assert self.beit3 is not None and self.gemini is not None
            return self._search_beit3_llm_video_mean_top3(query, latency_mode=latency_mode, output_depth=output_depth)

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

    def _search_openclip_llm_scene_rewrite(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.openclip is not None and self.gemini is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        with tracker.measure("gemini_scene_rewrite"):
            rewritten_query, cache_hit = self._get_gemini_scene_rewrite(query, latency_mode)
        with tracker.measure("openclip_scene_rewrite_branch"):
            openclip_vector = self.openclip.encode_texts([rewritten_query])[0]
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
                "original_query": query,
                "rewritten_query": rewritten_query,
                "gemini_cache_hit": cache_hit,
                "rewrite_policy": "query_only_scene_object_event_rewrite_no_video_access",
                "branches": ["openclip_llm_scene_rewrite"],
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
        score_strategy = str(get_config_value(self.config, "video_aggregation.score_strategy", "mean_topk"))
        score_top_k = int(get_config_value(self.config, "video_aggregation.score_top_k", 3))

        with tracker.measure("beit3_video_mean_top3_branch"):
            beit3_vector = self.beit3.encode_texts([query])[0]
            global_results = self.zilliz.search(self.zilliz.beit3_field, beit3_vector, global_depth)
            grouped = _group_candidates_by_video(global_results)
            ranked_videos = sorted(
                grouped.items(),
                key=lambda item: (-_aggregate_candidate_score(item[1], strategy=score_strategy, k=score_top_k), item[0]),
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
                "video_score_strategy": score_strategy,
                "video_score_top_k": score_top_k,
                "branches": ["beit3_video_mean_top3"],
            },
        )

    def _search_beit3_window_mean_top3(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        global_depth = int(get_config_value(self.config, "video_aggregation.global_depth", 1000))
        window_size_seconds = float(get_config_value(self.config, "window_aggregation.window_size_seconds", 8.0))
        window_stride_seconds = float(get_config_value(self.config, "window_aggregation.window_stride_seconds", 4.0))
        window_score_strategy = str(get_config_value(self.config, "window_aggregation.score_strategy", "mean_topk"))
        window_score_top_k = int(get_config_value(self.config, "window_aggregation.window_score_top_k", 3))
        keyframes_per_window = max(1, int(get_config_value(self.config, "window_aggregation.keyframes_per_window", 1)))
        max_windows_per_video = max(1, int(get_config_value(self.config, "window_aggregation.max_windows_per_video", 1)))

        with tracker.measure("beit3_window_mean_top3_branch"):
            beit3_vector = self.beit3.encode_texts([query])[0]
            global_results = self.zilliz.search(self.zilliz.beit3_field, beit3_vector, global_depth)
            grouped = _group_candidates_by_window(
                global_results,
                window_size_seconds=window_size_seconds,
                window_stride_seconds=window_stride_seconds,
            )
            ranked_windows = sorted(
                grouped.items(),
                key=lambda item: (
                    -_aggregate_candidate_score(item[1], strategy=window_score_strategy, k=window_score_top_k),
                    item[0],
                ),
            )

            final_results: List[RetrievalCandidate] = []
            seen_keyframes: set[str] = set()
            windows_by_video: Dict[str, int] = {}
            for _, candidates in ranked_windows:
                if not candidates:
                    continue
                video_id = candidates[0].video_id
                if windows_by_video.get(video_id, 0) >= max_windows_per_video:
                    continue
                windows_by_video[video_id] = windows_by_video.get(video_id, 0) + 1
                for candidate in candidates[:keyframes_per_window]:
                    if candidate.keyframe_id in seen_keyframes:
                        continue
                    seen_keyframes.add(candidate.keyframe_id)
                    final_results.append(candidate)
                    if len(final_results) >= final_depth:
                        break
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
                "window_size_seconds": window_size_seconds,
                "window_stride_seconds": window_stride_seconds,
                "window_score_strategy": window_score_strategy,
                "window_score_top_k": window_score_top_k,
                "keyframes_per_window": keyframes_per_window,
                "max_windows_per_video": max_windows_per_video,
                "branches": ["beit3_window_mean_top3"],
            },
        )

    def _search_beit3_video_xpool_softmax_heuristic(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        candidate_videos = max(1, int(get_config_value(self.config, "xpool_heuristic.candidate_videos", 100)))
        keyframes_per_video = max(1, int(get_config_value(self.config, "xpool_heuristic.keyframes_per_video", 1)))
        temperature = max(1e-6, float(get_config_value(self.config, "xpool_heuristic.temperature", 0.07)))

        with tracker.measure("beit3_text_encoding"):
            query_vector = _l2_normalize_np(self.beit3.encode_texts([query])[0])

        with tracker.measure("load_video_mean_pool_index"):
            video_index = self._get_video_xpool_index(self.zilliz.beit3_field)

        with tracker.measure("video_mean_pool_candidate_generation"):
            ranked_video_ids = _rank_video_means(query_vector, video_index, candidate_videos)

        final_results: List[RetrievalCandidate] = []
        with tracker.measure("xpool_softmax_heuristic_video_rerank"):
            reranked_videos = []
            for candidate_rank, (video_id, mean_pool_score) in enumerate(ranked_video_ids, start=1):
                video_item = video_index["videos"][video_id]
                frame_vectors = video_item["vectors"]
                frame_similarities = frame_vectors @ query_vector
                attention = _softmax_np(frame_similarities / temperature)
                pooled_vector = _l2_normalize_np(np.sum(frame_vectors * attention[:, None], axis=0))
                xpool_score = float(np.dot(query_vector, pooled_vector))
                reranked_videos.append(
                    {
                        "video_id": video_id,
                        "candidate_rank": candidate_rank,
                        "mean_pool_score": float(mean_pool_score),
                        "xpool_score": xpool_score,
                        "attention": attention,
                        "frame_similarities": frame_similarities,
                    }
                )

            reranked_videos.sort(key=lambda item: (-float(item["xpool_score"]), str(item["video_id"])))
            for rerank, video_item in enumerate(reranked_videos, start=1):
                indexed_video = video_index["videos"][video_item["video_id"]]
                attention = video_item["attention"]
                frame_similarities = video_item["frame_similarities"]
                keyframe_order = np.argsort(-frame_similarities, kind="mergesort")
                for keyframe_order_rank, frame_index in enumerate(keyframe_order[:keyframes_per_video], start=1):
                    source = indexed_video["candidates"][int(frame_index)]
                    final_results.append(
                        _copy_candidate_with_score(
                            source,
                            score=float(video_item["xpool_score"]),
                            metadata={
                                "video_candidate_rank": int(video_item["candidate_rank"]),
                                "video_rerank_rank": int(rerank),
                                "video_mean_pool_score": float(video_item["mean_pool_score"]),
                                "xpool_softmax_score": float(video_item["xpool_score"]),
                                "xpool_softmax_attention": float(attention[int(frame_index)]),
                                "xpool_temperature": float(temperature),
                                "keyframe_query_similarity": float(frame_similarities[int(frame_index)]),
                                "keyframe_rank_in_video": int(keyframe_order_rank),
                                "ranking_policy": "video_rank_first_then_keyframe_similarity_in_video",
                            },
                        )
                    )
                    if len(final_results) >= final_depth:
                        break
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
                "candidate_videos": candidate_videos,
                "keyframes_per_video": keyframes_per_video,
                "xpool_temperature": temperature,
                "video_index_videos": int(video_index["num_videos"]),
                "video_index_keyframes": int(video_index["num_keyframes"]),
                "video_candidate_stage": "mean_pool_l2_normalized_keyframe_embeddings",
                "video_rerank_stage": "query_conditioned_softmax_pooling_no_train",
                "keyframe_selection_stage": "rank_keyframes_by_query_similarity_inside_each_ranked_video",
                "branches": ["beit3_video_xpool_softmax_heuristic"],
            },
        )

    def _search_beit3_video_xpool_heuristic(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        candidate_videos = max(1, int(get_config_value(self.config, "xpool_heuristic.candidate_videos", 100)))
        keyframes_per_video = max(1, int(get_config_value(self.config, "xpool_heuristic.keyframes_per_video", 1)))
        temperature = max(1e-6, float(get_config_value(self.config, "xpool_heuristic.temperature", 0.07)))

        with tracker.measure("beit3_text_encoding"):
            query_vector = _l2_normalize_np(self.beit3.encode_texts([query])[0])

        with tracker.measure("load_video_mean_pool_index"):
            video_index = self._get_video_xpool_index(self.zilliz.beit3_field)

        with tracker.measure("video_mean_pool_candidate_generation"):
            ranked_video_ids = _rank_video_means(query_vector, video_index, candidate_videos)

        final_results: List[RetrievalCandidate] = []
        with tracker.measure("xpool_heuristic_video_rerank"):
            xpool_transformer = self._get_xpool_transformer(embed_dim=int(query_vector.shape[0]))
            query_tensor = _torch_tensor_from_np(query_vector, next(xpool_transformer.parameters()).device).unsqueeze(0)
            reranked_videos = []
            for candidate_rank, (video_id, mean_pool_score) in enumerate(ranked_video_ids, start=1):
                video_item = video_index["videos"][video_id]
                frame_vectors = video_item["vectors"]
                frame_similarities = frame_vectors @ query_vector
                frame_tensor = _torch_tensor_from_np(frame_vectors, query_tensor.device).unsqueeze(0)
                with torch.no_grad():
                    pooled_tensor, attention_tensor = xpool_transformer(
                        query_tensor,
                        frame_tensor,
                        return_attention=True,
                    )
                    pooled_tensor = torch.nn.functional.normalize(pooled_tensor.squeeze(0), dim=-1)
                    normalized_query = torch.nn.functional.normalize(query_tensor, dim=-1)
                    xpool_score = float((normalized_query * pooled_tensor.squeeze(0)).sum().detach().cpu().item())
                    attention = attention_tensor.mean(dim=1).squeeze(0).squeeze(-1).detach().cpu().numpy()
                reranked_videos.append(
                    {
                        "video_id": video_id,
                        "candidate_rank": candidate_rank,
                        "mean_pool_score": float(mean_pool_score),
                        "xpool_score": xpool_score,
                        "attention": attention,
                        "frame_similarities": frame_similarities,
                    }
                )

            reranked_videos.sort(key=lambda item: (-float(item["xpool_score"]), str(item["video_id"])))
            for rerank, video_item in enumerate(reranked_videos, start=1):
                indexed_video = video_index["videos"][video_item["video_id"]]
                attention = video_item["attention"]
                frame_similarities = video_item["frame_similarities"]
                keyframe_order = np.argsort(-frame_similarities, kind="mergesort")
                for keyframe_order_rank, frame_index in enumerate(keyframe_order[:keyframes_per_video], start=1):
                    source = indexed_video["candidates"][int(frame_index)]
                    final_results.append(
                        _copy_candidate_with_score(
                            source,
                            score=float(video_item["xpool_score"]),
                            metadata={
                                "video_candidate_rank": int(video_item["candidate_rank"]),
                                "video_rerank_rank": int(rerank),
                                "video_mean_pool_score": float(video_item["mean_pool_score"]),
                                "xpool_transformer_score": float(video_item["xpool_score"]),
                                "xpool_transformer_attention": float(attention[int(frame_index)]),
                                "keyframe_query_similarity": float(frame_similarities[int(frame_index)]),
                                "keyframe_rank_in_video": int(keyframe_order_rank),
                                "ranking_policy": "video_rank_first_then_keyframe_similarity_in_video",
                            },
                        )
                    )
                    if len(final_results) >= final_depth:
                        break
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
                "candidate_videos": candidate_videos,
                "keyframes_per_video": keyframes_per_video,
                "xpool_temperature": temperature,
                "xpool_num_heads": int(get_config_value(self.config, "xpool_heuristic.num_heads", 1)),
                "xpool_dropout": float(get_config_value(self.config, "xpool_heuristic.dropout", 0.3)),
                "xpool_checkpoint": self._xpool_checkpoint_info,
                "video_index_videos": int(video_index["num_videos"]),
                "video_index_keyframes": int(video_index["num_keyframes"]),
                "video_candidate_stage": "mean_pool_l2_normalized_keyframe_embeddings",
                "video_rerank_stage": "xpool_transformer_query_conditioned_pooling",
                "keyframe_selection_stage": "rank_keyframes_by_query_similarity_inside_each_ranked_video",
                "branches": ["beit3_video_xpool_heuristic"],
            },
        )

    def _search_beit3_video_xpool_paperlike(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        num_frames = max(1, int(get_config_value(self.config, "xpool_paperlike.num_frames", 12)))
        keyframes_per_video = max(1, int(get_config_value(self.config, "xpool_paperlike.keyframes_per_video", 1)))
        batch_size = max(1, int(get_config_value(self.config, "xpool_paperlike.batch_size", 128)))

        with tracker.measure("beit3_text_encoding"):
            query_vector = _l2_normalize_np(self.beit3.encode_texts([query])[0])

        with tracker.measure("load_video_keyframe_index"):
            video_index = self._get_video_xpool_index(self.zilliz.beit3_field)

        final_results: List[RetrievalCandidate] = []
        sampled_indices_by_video: Dict[str, List[int]] = {}
        with tracker.measure("xpool_paperlike_all_video_rerank"):
            xpool_transformer = self._get_xpool_transformer(
                embed_dim=int(query_vector.shape[0]),
                config_root="xpool_paperlike",
            )
            device = next(xpool_transformer.parameters()).device
            query_tensor = _torch_tensor_from_np(query_vector, device).unsqueeze(0)
            normalized_query = torch.nn.functional.normalize(query_tensor, dim=-1)
            video_ids = sorted(video_index["videos"].keys())
            reranked_videos: List[Dict[str, Any]] = []

            for start in range(0, len(video_ids), batch_size):
                batch_video_ids = video_ids[start : start + batch_size]
                sampled_frame_vectors = []
                sampled_frame_indices = []
                for video_id in batch_video_ids:
                    frame_vectors, frame_indices = _sample_video_frame_vectors(
                        video_index["videos"][video_id],
                        num_frames=num_frames,
                    )
                    sampled_frame_vectors.append(frame_vectors)
                    sampled_frame_indices.append(frame_indices)
                    sampled_indices_by_video[video_id] = frame_indices

                frame_tensor = _torch_tensor_from_np(np.stack(sampled_frame_vectors, axis=0), device)
                with torch.no_grad():
                    pooled_tensor, attention_tensor = xpool_transformer(
                        query_tensor,
                        frame_tensor,
                        return_attention=True,
                    )
                    pooled_tensor = torch.nn.functional.normalize(pooled_tensor.squeeze(1), dim=-1)
                    scores = (pooled_tensor * normalized_query).sum(dim=-1).detach().cpu().numpy()
                    attention = attention_tensor.mean(dim=1).squeeze(-1).detach().cpu().numpy()

                for local_index, video_id in enumerate(batch_video_ids):
                    reranked_videos.append(
                        {
                            "video_id": video_id,
                            "xpool_score": float(scores[local_index]),
                            "attention": attention[local_index],
                            "sampled_indices": sampled_frame_indices[local_index],
                        }
                    )

            reranked_videos.sort(key=lambda item: (-float(item["xpool_score"]), str(item["video_id"])))
            for video_rank, video_item in enumerate(reranked_videos, start=1):
                indexed_video = video_index["videos"][video_item["video_id"]]
                frame_vectors = indexed_video["vectors"]
                frame_similarities = frame_vectors @ query_vector
                keyframe_order = np.argsort(-frame_similarities, kind="mergesort")
                for keyframe_rank, frame_index in enumerate(keyframe_order[:keyframes_per_video], start=1):
                    frame_index_int = int(frame_index)
                    source = indexed_video["candidates"][frame_index_int]
                    sampled_attention = _sampled_attention_for_frame_index(
                        frame_index_int,
                        video_item["sampled_indices"],
                        video_item["attention"],
                    )
                    final_results.append(
                        _copy_candidate_with_score(
                            source,
                            score=float(video_item["xpool_score"]),
                            metadata={
                                "video_rerank_rank": int(video_rank),
                                "xpool_transformer_score": float(video_item["xpool_score"]),
                                "keyframe_query_similarity": float(frame_similarities[frame_index_int]),
                                "keyframe_rank_in_video": int(keyframe_rank),
                                "xpool_num_sampled_frames": int(num_frames),
                                "xpool_sampled_frame_indices": list(video_item["sampled_indices"]),
                                "selected_keyframe_was_sampled_for_xpool": frame_index_int
                                in set(video_item["sampled_indices"]),
                                "selected_keyframe_xpool_attention": sampled_attention,
                                "ranking_policy": "rank_all_videos_by_xpool_then_keyframe_similarity_in_video",
                            },
                        )
                    )
                    if len(final_results) >= final_depth:
                        break
                if len(final_results) >= final_depth:
                    break

        checkpoint_key = f"xpool_paperlike:{int(query_vector.shape[0])}"
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
                "xpool_num_frames": num_frames,
                "xpool_batch_size": batch_size,
                "keyframes_per_video": keyframes_per_video,
                "xpool_num_heads": int(get_config_value(self.config, "xpool_paperlike.num_heads", 1)),
                "xpool_dropout": float(get_config_value(self.config, "xpool_paperlike.dropout", 0.3)),
                "xpool_checkpoint": self._xpool_checkpoint_info_by_key.get(checkpoint_key, {}),
                "video_index_videos": int(video_index["num_videos"]),
                "video_index_keyframes": int(video_index["num_keyframes"]),
                "video_candidate_stage": "all_videos_no_mean_pool_pruning",
                "frame_sampling_stage": "uniform_sample_existing_keyframes_to_fixed_slots",
                "video_rerank_stage": "batched_xpool_transformer_query_conditioned_pooling_all_videos",
                "keyframe_selection_stage": "rank_all_keyframes_by_query_similarity_inside_each_ranked_video",
                "branches": ["beit3_video_xpool_paperlike"],
            },
        )

    def _get_xpool_transformer(self, embed_dim: int, config_root: str = "xpool_heuristic") -> XPoolTransformer:
        cache_key = f"{config_root}:{int(embed_dim)}"
        cached = self._xpool_transformer_by_key.get(cache_key)
        if cached is not None:
            return cached
        num_heads = int(get_config_value(self.config, f"{config_root}.num_heads", 1) or 1)
        dropout = float(get_config_value(self.config, f"{config_root}.dropout", 0.3) or 0.3)
        checkpoint_path = get_config_value(self.config, f"{config_root}.checkpoint", None)
        device_name = str(get_config_value(self.config, "models.beit3.device", "cuda"))
        device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
        module = XPoolTransformer(embed_dim=int(embed_dim), num_heads=num_heads, dropout=dropout)
        checkpoint_info = load_xpool_transformer_checkpoint(module, checkpoint_path)
        self._xpool_checkpoint_info = checkpoint_info
        self._xpool_checkpoint_info_by_key[cache_key] = checkpoint_info
        module.to(device)
        module.eval()
        self._xpool_transformer = module
        self._xpool_transformer_by_key[cache_key] = module
        return module

    def _get_video_xpool_index(self, vector_field: str) -> Dict[str, Any]:
        cached = self._video_xpool_index_by_field.get(vector_field)
        if cached is not None:
            return cached

        model_version = str(self.config["project"]["model_version"])
        config_version = str(self.config["project"]["config_version"])
        filter_expr = (
            f"model_version == {_zilliz_string(model_version)} "
            f"and config_version == {_zilliz_string(config_version)}"
        )
        batch_size = int(get_config_value(self.config, "zilliz.query_batch_size", 16000))
        pairs = self.zilliz.fetch_keyframe_vectors(vector_field, filter_expr=filter_expr, batch_size=batch_size)
        grouped: Dict[str, List[tuple[RetrievalCandidate, Sequence[float]]]] = {}
        for candidate, vector in pairs:
            grouped.setdefault(candidate.video_id, []).append((candidate, vector))

        videos: Dict[str, Dict[str, Any]] = {}
        num_keyframes = 0
        for video_id, items in grouped.items():
            items.sort(key=lambda item: (float(item[0].timestamp_raw), int(item[0].frame_index_raw), item[0].keyframe_id))
            candidates = [item[0] for item in items]
            vectors = _l2_normalize_matrix_np([item[1] for item in items])
            if vectors.size == 0:
                continue
            video_vector = _l2_normalize_np(vectors.mean(axis=0))
            videos[video_id] = {
                "candidates": candidates,
                "vectors": vectors,
                "video_vector": video_vector,
            }
            num_keyframes += len(candidates)

        index = {
            "videos": videos,
            "num_videos": len(videos),
            "num_keyframes": num_keyframes,
        }
        self._video_xpool_index_by_field[vector_field] = index
        return index

    def _search_beit3_llm_video_mean_top3(
        self,
        query: str,
        latency_mode: str,
        output_depth: int | None = None,
    ) -> SearchResponse:
        assert self.beit3 is not None and self.gemini is not None
        tracker = LatencyTracker()
        final_depth = output_depth or int(get_config_value(self.config, "retrieval.output_depth", 500))
        global_depth = int(get_config_value(self.config, "video_aggregation.global_depth", 1000))
        keyframes_per_video = int(get_config_value(self.config, "video_aggregation.keyframes_per_video", 1))
        keyframes_per_video = max(1, keyframes_per_video)
        score_strategy = str(get_config_value(self.config, "video_aggregation.score_strategy", "mean_topk"))
        score_top_k = int(get_config_value(self.config, "video_aggregation.score_top_k", 3))

        with tracker.measure("gemini_paraphrases"):
            paraphrases, cache_hit = self._get_gemini_paraphrases(query, latency_mode)
        query_variants = [query] + paraphrases

        with tracker.measure("beit3_llm_video_mean_top3_branch"):
            text_vectors = self.beit3.encode_texts(query_variants)
            merged_by_keyframe: Dict[str, RetrievalCandidate] = {}
            for vector in text_vectors:
                for candidate in self.zilliz.search(self.zilliz.beit3_field, vector, global_depth):
                    previous = merged_by_keyframe.get(candidate.keyframe_id)
                    if previous is None or candidate.score > previous.score:
                        merged_by_keyframe[candidate.keyframe_id] = candidate

            global_results = list(merged_by_keyframe.values())
            grouped = _group_candidates_by_video(global_results)
            for candidates in grouped.values():
                candidates.sort(key=lambda item: (-float(item.score), item.keyframe_id))
            ranked_videos = sorted(
                grouped.items(),
                key=lambda item: (-_aggregate_candidate_score(item[1], strategy=score_strategy, k=score_top_k), item[0]),
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
                "video_score_strategy": score_strategy,
                "video_score_top_k": score_top_k,
                "gemini_cache_hit": cache_hit,
                "query_variants": query_variants,
                "branches": ["beit3_llm_video_mean_top3"],
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

    def _get_gemini_paraphrases(self, query: str, latency_mode: str) -> tuple[List[str], bool]:
        assert self.gemini is not None
        cache_version = self.config["project"]["config_version"]
        cache_key = query_cache_key(query, cache_version)
        cache = self.cockroach.get_query_cache(cache_key, cache_version)
        paraphrases = list((cache or {}).get("gemini_paraphrases") or [])
        cache_hit = len(paraphrases) == 5
        if cache_hit:
            return paraphrases, True
        if latency_mode == "cache_hit_retrieval_only":
            raise RuntimeError(f"Missing Gemini paraphrase cache for query_hash={cache_key}")
        paraphrases = self.gemini.generate_paraphrases(query)
        existing = cache or {}
        self.cockroach.upsert_query_cache(
            query_hash=cache_key,
            query_text=query,
            cache_version=cache_version,
            gemini_paraphrases=paraphrases,
            gemini_object_constraints=existing.get("gemini_object_constraints") or [],
            sd_prompt=existing.get("sd_prompt"),
            sd_image_paths_or_hashes=existing.get("sd_image_paths_or_hashes") or [],
            sd_seeds=existing.get("sd_seeds") or [],
        )
        return paraphrases, False

    def _get_gemini_scene_rewrite(self, query: str, latency_mode: str) -> tuple[str, bool]:
        assert self.gemini is not None
        cache_version = self.config["project"]["config_version"]
        cache_query = f"scene_rewrite_for_keyframe_search::{query}"
        cache_key = query_cache_key(cache_query, cache_version)
        cache = self.cockroach.get_query_cache(cache_key, cache_version)
        cached_rewrites = list((cache or {}).get("gemini_paraphrases") or [])
        if cached_rewrites and isinstance(cached_rewrites[0], str) and cached_rewrites[0].strip():
            return str(cached_rewrites[0]), True
        if latency_mode == "cache_hit_retrieval_only":
            raise RuntimeError(f"Missing Gemini scene rewrite cache for query_hash={cache_key}")
        rewritten_query = self.gemini.rewrite_for_keyframe_search(query)
        existing = cache or {}
        self.cockroach.upsert_query_cache(
            query_hash=cache_key,
            query_text=cache_query,
            cache_version=cache_version,
            gemini_paraphrases=[rewritten_query],
            gemini_object_constraints=existing.get("gemini_object_constraints") or [],
            sd_prompt=existing.get("sd_prompt"),
            sd_image_paths_or_hashes=existing.get("sd_image_paths_or_hashes") or [],
            sd_seeds=existing.get("sd_seeds") or [],
        )
        return rewritten_query, False


def _group_candidates_by_video(candidates: List[RetrievalCandidate]) -> Dict[str, List[RetrievalCandidate]]:
    grouped: Dict[str, List[RetrievalCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.video_id, []).append(candidate)
    return grouped


def _group_candidates_by_window(
    candidates: List[RetrievalCandidate],
    window_size_seconds: float,
    window_stride_seconds: float,
) -> Dict[str, List[RetrievalCandidate]]:
    window_size_seconds = max(0.001, float(window_size_seconds))
    window_stride_seconds = max(0.001, float(window_stride_seconds))
    grouped: Dict[str, List[RetrievalCandidate]] = {}
    for candidate in candidates:
        for window_key in _candidate_window_keys(candidate, window_size_seconds, window_stride_seconds):
            grouped.setdefault(window_key, []).append(candidate)
    for window_candidates in grouped.values():
        window_candidates.sort(key=lambda item: (-float(item.score), item.keyframe_id))
    return grouped


def _candidate_window_keys(
    candidate: RetrievalCandidate,
    window_size_seconds: float,
    window_stride_seconds: float,
) -> List[str]:
    timestamp = max(0.0, float(candidate.timestamp_raw))
    latest_index = int(timestamp // window_stride_seconds)
    earliest_start = max(0.0, timestamp - window_size_seconds)
    earliest_index = int(earliest_start // window_stride_seconds)
    keys: List[str] = []
    for index in range(earliest_index, latest_index + 1):
        start = index * window_stride_seconds
        end = start + window_size_seconds
        if start <= timestamp <= end:
            keys.append(f"{candidate.video_id}|{start:.3f}|{end:.3f}")
    if not keys:
        start = latest_index * window_stride_seconds
        keys.append(f"{candidate.video_id}|{start:.3f}|{start + window_size_seconds:.3f}")
    return keys


def _zilliz_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _l2_normalize_np(vector: Sequence[float] | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm < eps:
        return np.zeros_like(array, dtype=np.float32)
    return array / norm


def _l2_normalize_matrix_np(vectors: Sequence[Sequence[float]], eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def _softmax_np(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0.0:
        return np.full_like(values, 1.0 / max(1, values.shape[0]), dtype=np.float32)
    return exp_values / total


def _torch_tensor_from_np(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _sample_video_frame_vectors(video_item: Dict[str, Any], num_frames: int) -> tuple[np.ndarray, List[int]]:
    vectors = video_item["vectors"]
    total_frames = int(vectors.shape[0])
    if total_frames <= 0:
        raise ValueError("Cannot sample X-Pool frames from an empty video.")
    num_frames = max(1, int(num_frames))
    indices = np.rint(np.linspace(0, total_frames - 1, num_frames)).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    return vectors[indices], [int(index) for index in indices.tolist()]


def _sampled_attention_for_frame_index(
    frame_index: int,
    sampled_indices: Sequence[int],
    attention: Sequence[float] | np.ndarray,
) -> float | None:
    matches = [
        float(attention[position])
        for position, sampled_index in enumerate(sampled_indices)
        if int(sampled_index) == int(frame_index)
    ]
    if not matches:
        return None
    return float(sum(matches) / len(matches))


def _rank_video_means(query_vector: np.ndarray, video_index: Dict[str, Any], candidate_videos: int) -> List[tuple[str, float]]:
    scores = []
    for video_id, item in video_index["videos"].items():
        scores.append((video_id, float(np.dot(query_vector, item["video_vector"]))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:candidate_videos]


def _copy_candidate_with_score(
    candidate: RetrievalCandidate,
    score: float,
    metadata: Dict[str, Any] | None = None,
) -> RetrievalCandidate:
    merged_metadata = dict(candidate.metadata or {})
    if metadata:
        merged_metadata.update(metadata)
    return RetrievalCandidate(
        keyframe_id=candidate.keyframe_id,
        video_id=candidate.video_id,
        shot_id=candidate.shot_id,
        timestamp_raw=candidate.timestamp_raw,
        frame_index_raw=candidate.frame_index_raw,
        score=float(score),
        rank=0,
        object_counts=dict(candidate.object_counts or {}),
        metadata=merged_metadata,
    )


def _mean_top_k_score(candidates: List[RetrievalCandidate], k: int = 3) -> float:
    scores = sorted((float(candidate.score) for candidate in candidates), reverse=True)[:k]
    return sum(scores) / len(scores) if scores else float("-inf")


def _aggregate_candidate_score(candidates: List[RetrievalCandidate], strategy: str, k: int = 3) -> float:
    if not candidates:
        return float("-inf")
    k = max(1, int(k))
    strategy = strategy.lower().strip()
    ranked = sorted(candidates, key=lambda item: (-float(item.score), int(item.rank or 10**9), item.keyframe_id))
    top = ranked[:k]
    scores = [float(candidate.score) for candidate in top]
    if strategy in {"mean_topk", "mean_top_k", "mean"}:
        return sum(scores) / len(scores)
    if strategy in {"max", "max_score", "top1"}:
        return max(scores)
    if strategy in {"sum_topk", "sum_top_k", "sum"}:
        return sum(scores)
    if strategy in {"rrf_topk", "rrf_top_k", "rrf"}:
        return sum(1.0 / (60.0 + float(candidate.rank or 10**9)) for candidate in top)
    raise ValueError(
        f"Unsupported aggregation score_strategy={strategy!r}. "
        "Expected one of: mean_topk, max, sum_topk, rrf_topk."
    )


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
    if baseline_openclip_only:
        overrides["online"] = {"mode": "openclip_user_query_baseline"}
        overrides["object_filter"] = {"enabled": False}
    if not overrides:
        overrides = None
    pipeline = SearchPipeline(load_config(config_path, overrides=overrides))
    return pipeline.search(query, latency_mode=latency_mode).to_dict()
