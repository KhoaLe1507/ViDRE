from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.load_beit3 import BEiT3Encoder
from src.models.load_openclip import OpenCLIPH14Encoder
from src.models.load_stable_diffusion import StableDiffusionGenerator
from src.online.fusion import rrf_fuse
from src.schemas import RetrievalCandidate
from src.storage.cockroach_client import CockroachClient
from src.storage.zilliz_client import ZillizClient
from src.utils.config import get_config_value
from src.utils.hashing import query_cache_key


class StableDiffusionBranch:
    branch_name = "stable_diffusion"

    def __init__(
        self,
        config: Dict[str, Any],
        beit3: BEiT3Encoder,
        openclip: OpenCLIPH14Encoder,
        zilliz: ZillizClient,
        cockroach: CockroachClient,
        sd_generator: StableDiffusionGenerator,
    ):
        self.config = config
        self.beit3 = beit3
        self.openclip = openclip
        self.zilliz = zilliz
        self.cockroach = cockroach
        self.sd_generator = sd_generator

    def run(self, query: str, latency_mode: str) -> tuple[List[RetrievalCandidate], Dict[str, Any]]:
        cache_version = self.config["project"]["config_version"]
        cache_key = query_cache_key(query, cache_version)
        cache = self.cockroach.get_query_cache(cache_key, cache_version)
        image_records = list((cache or {}).get("sd_image_paths_or_hashes") or [])
        cache_hit = _all_cached_images_exist(image_records)
        if not cache_hit:
            if latency_mode == "cache_hit_retrieval_only":
                raise RuntimeError(f"Missing Stable Diffusion cache for query_hash={cache_key}")
            output_dir = Path(get_config_value(self.config, "paths.sd_cache_dir", "outputs/cache/stable_diffusion")) / cache_key
            image_records = self.sd_generator.generate(query, output_dir)
            self._save_sd_cache(cache_key, query, cache_version, cache, image_records)

        image_paths = [record["path"] for record in image_records]
        top_k = int(get_config_value(self.config, "retrieval.depth_per_model", 200))
        branch_depth = int(get_config_value(self.config, "retrieval.branch_depth", 200))
        beit3_vectors = self.beit3.encode_images(image_paths)
        openclip_vectors = self.openclip.encode_images(image_paths)

        per_image_lists: List[List[RetrievalCandidate]] = []
        for beit3_vector, openclip_vector in zip(beit3_vectors, openclip_vectors):
            beit3_results = self.zilliz.search(self.zilliz.beit3_field, beit3_vector, top_k)
            openclip_results = self.zilliz.search(self.zilliz.openclip_field, openclip_vector, top_k)
            per_image_lists.append(
                rrf_fuse(
                    [beit3_results, openclip_results],
                    rrf_k=int(get_config_value(self.config, "rrf.rrf_k", 60)),
                    top_n=branch_depth,
                )
            )
        fused = rrf_fuse(per_image_lists, rrf_k=int(get_config_value(self.config, "rrf.rrf_k", 60)), top_n=branch_depth)
        return fused, {"image_records": image_records, "cache_hit": cache_hit}

    def _save_sd_cache(
        self,
        cache_key: str,
        query: str,
        cache_version: str,
        existing_cache: Optional[Dict[str, Any]],
        image_records: List[Dict[str, Any]],
    ) -> None:
        existing_cache = existing_cache or {}
        self.cockroach.upsert_query_cache(
            query_hash=cache_key,
            query_text=query,
            cache_version=cache_version,
            gemini_paraphrases=existing_cache.get("gemini_paraphrases") or [],
            gemini_object_constraints=existing_cache.get("gemini_object_constraints") or [],
            sd_prompt=self.sd_generator.build_prompt(query),
            sd_image_paths_or_hashes=image_records,
            sd_seeds=[int(record["seed"]) for record in image_records],
        )


def _all_cached_images_exist(image_records: List[Dict[str, Any]]) -> bool:
    if len(image_records) != 5:
        return False
    return all(record.get("path") and Path(record["path"]).exists() for record in image_records)

