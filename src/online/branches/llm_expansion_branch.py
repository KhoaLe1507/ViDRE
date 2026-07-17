from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models.gemini_client import GeminiClient
from src.models.load_beit3 import BEiT3Encoder
from src.models.load_openclip import OpenCLIPH14Encoder
from src.online.fusion import rrf_fuse
from src.schemas import RetrievalCandidate
from src.storage.cockroach_client import CockroachClient
from src.storage.zilliz_client import ZillizClient
from src.utils.config import get_config_value
from src.utils.hashing import query_cache_key


class LLMExpansionBranch:
    branch_name = "llm_query_expansion"

    def __init__(
        self,
        config: Dict[str, Any],
        beit3: BEiT3Encoder,
        openclip: OpenCLIPH14Encoder,
        zilliz: ZillizClient,
        cockroach: CockroachClient,
        gemini: GeminiClient,
    ):
        self.config = config
        self.beit3 = beit3
        self.openclip = openclip
        self.zilliz = zilliz
        self.cockroach = cockroach
        self.gemini = gemini

    def run(self, query: str, latency_mode: str) -> tuple[List[RetrievalCandidate], Dict[str, Any]]:
        cache_version = self.config["project"]["config_version"]
        cache_key = query_cache_key(query, cache_version)
        cache = self.cockroach.get_query_cache(cache_key, cache_version)
        paraphrases = list((cache or {}).get("gemini_paraphrases") or [])
        cache_hit = len(paraphrases) == 5
        if not cache_hit:
            if latency_mode == "cache_hit_retrieval_only":
                raise RuntimeError(f"Missing Gemini paraphrase cache for query_hash={cache_key}")
            paraphrases = self.gemini.generate_paraphrases(query)
            self._save_paraphrase_cache(cache_key, query, cache_version, cache, paraphrases)

        top_k = int(get_config_value(self.config, "retrieval.depth_per_model", 200))
        branch_depth = int(get_config_value(self.config, "retrieval.branch_depth", 200))
        beit3_vectors = self.beit3.encode_texts(paraphrases)
        openclip_vectors = self.openclip.encode_texts(paraphrases)

        per_paraphrase_lists: List[List[RetrievalCandidate]] = []
        for beit3_vector, openclip_vector in zip(beit3_vectors, openclip_vectors):
            beit3_results = self.zilliz.search(self.zilliz.beit3_field, beit3_vector, top_k)
            openclip_results = self.zilliz.search(self.zilliz.openclip_field, openclip_vector, top_k)
            per_paraphrase_lists.append(
                rrf_fuse(
                    [beit3_results, openclip_results],
                    rrf_k=int(get_config_value(self.config, "rrf.rrf_k", 60)),
                    top_n=branch_depth,
                )
            )
        fused = rrf_fuse(per_paraphrase_lists, rrf_k=int(get_config_value(self.config, "rrf.rrf_k", 60)), top_n=branch_depth)
        return fused, {"paraphrases": paraphrases, "cache_hit": cache_hit}

    def _save_paraphrase_cache(
        self,
        cache_key: str,
        query: str,
        cache_version: str,
        existing_cache: Optional[Dict[str, Any]],
        paraphrases: List[str],
    ) -> None:
        existing_cache = existing_cache or {}
        self.cockroach.upsert_query_cache(
            query_hash=cache_key,
            query_text=query,
            cache_version=cache_version,
            gemini_paraphrases=paraphrases,
            gemini_object_constraints=existing_cache.get("gemini_object_constraints") or [],
            sd_prompt=existing_cache.get("sd_prompt"),
            sd_image_paths_or_hashes=existing_cache.get("sd_image_paths_or_hashes") or [],
            sd_seeds=existing_cache.get("sd_seeds") or [],
        )

