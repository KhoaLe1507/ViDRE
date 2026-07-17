from __future__ import annotations

from typing import Any, Dict, List

from src.models.load_beit3 import BEiT3Encoder
from src.models.load_openclip import OpenCLIPH14Encoder
from src.online.fusion import rrf_fuse
from src.schemas import RetrievalCandidate
from src.storage.zilliz_client import ZillizClient
from src.utils.config import get_config_value


class TextualQueryBranch:
    branch_name = "textual_query"

    def __init__(
        self,
        config: Dict[str, Any],
        beit3: BEiT3Encoder,
        openclip: OpenCLIPH14Encoder,
        zilliz: ZillizClient,
    ):
        self.config = config
        self.beit3 = beit3
        self.openclip = openclip
        self.zilliz = zilliz

    def run(self, query: str) -> List[RetrievalCandidate]:
        top_k = int(get_config_value(self.config, "retrieval.depth_per_model", 200))
        branch_depth = int(get_config_value(self.config, "retrieval.branch_depth", 200))
        beit3_vector = self.beit3.encode_texts([query])[0]
        openclip_vector = self.openclip.encode_texts([query])[0]
        beit3_results = self.zilliz.search(self.zilliz.beit3_field, beit3_vector, top_k)
        openclip_results = self.zilliz.search(self.zilliz.openclip_field, openclip_vector, top_k)
        return rrf_fuse(
            [beit3_results, openclip_results],
            rrf_k=int(get_config_value(self.config, "rrf.rrf_k", 60)),
            top_n=branch_depth,
        )

