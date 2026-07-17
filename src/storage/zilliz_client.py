from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.schemas import KeyframeRecord, RetrievalCandidate
from src.utils.config import get_config_value


def _require_pymilvus():
    try:
        from pymilvus import DataType, MilvusClient
    except ImportError as exc:
        raise RuntimeError("pymilvus is required for Zilliz. Install dependencies from requirements.txt.") from exc
    return DataType, MilvusClient


class ZillizClient:
    def __init__(self, config: Dict[str, Any]):
        _, milvus_client_cls = _require_pymilvus()
        zilliz_cfg = config.get("zilliz", {})
        uri = os.environ.get(zilliz_cfg.get("uri_env", "ZILLIZ_URI"), "")
        token = os.environ.get(zilliz_cfg.get("token_env", "ZILLIZ_TOKEN"), "")
        if not uri or not token:
            raise RuntimeError("ZILLIZ_URI and ZILLIZ_TOKEN must be set.")
        self.collection_name = os.environ.get(
            zilliz_cfg.get("collection_env", "ZILLIZ_COLLECTION"),
            zilliz_cfg.get("default_collection", "vidre_keyframes"),
        )
        self.client = milvus_client_cls(uri=uri, token=token)
        self.config = config

    @property
    def beit3_field(self) -> str:
        return get_config_value(self.config, "zilliz.beit3_field", "beit3_vector")

    @property
    def openclip_field(self) -> str:
        return get_config_value(self.config, "zilliz.openclip_field", "openclip_h14_vector")

    def create_collection(self, drop_existing: bool = False) -> None:
        DataType, _ = _require_pymilvus()
        if self.client.has_collection(self.collection_name):
            if not drop_existing:
                return
            self.client.drop_collection(self.collection_name)

        beit3_dim = int(get_config_value(self.config, "models.beit3.embedding_dim", 1024))
        openclip_dim = int(get_config_value(self.config, "models.openclip_h14.embedding_dim", 1024))
        metric_type = get_config_value(self.config, "zilliz.metric_type", "IP")
        index_type = get_config_value(self.config, "zilliz.index_type", "AUTOINDEX")

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("keyframe_id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("video_id", DataType.VARCHAR, max_length=128)
        schema.add_field("shot_id", DataType.VARCHAR, max_length=512)
        schema.add_field("timestamp_raw", DataType.DOUBLE)
        schema.add_field("frame_index_raw", DataType.INT64)
        schema.add_field("model_version", DataType.VARCHAR, max_length=128)
        schema.add_field("config_version", DataType.VARCHAR, max_length=128)
        schema.add_field(self.beit3_field, DataType.FLOAT_VECTOR, dim=beit3_dim)
        schema.add_field(self.openclip_field, DataType.FLOAT_VECTOR, dim=openclip_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(self.beit3_field, index_type=index_type, metric_type=metric_type)
        index_params.add_index(self.openclip_field, index_type=index_type, metric_type=metric_type)

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_keyframes(
        self,
        keyframes: Sequence[KeyframeRecord],
        beit3_vectors: Sequence[Sequence[float]],
        openclip_vectors: Sequence[Sequence[float]],
    ) -> None:
        if not (len(keyframes) == len(beit3_vectors) == len(openclip_vectors)):
            raise ValueError("keyframes, beit3_vectors, and openclip_vectors must have the same length.")
        rows: List[Dict[str, Any]] = []
        for keyframe, beit3_vector, openclip_vector in zip(keyframes, beit3_vectors, openclip_vectors):
            rows.append(
                {
                    "keyframe_id": keyframe.keyframe_id,
                    "video_id": keyframe.video_id,
                    "shot_id": keyframe.shot_id,
                    "timestamp_raw": float(keyframe.timestamp_raw),
                    "frame_index_raw": int(keyframe.frame_index_raw),
                    "model_version": keyframe.model_version,
                    "config_version": keyframe.config_version,
                    self.beit3_field: [float(x) for x in beit3_vector],
                    self.openclip_field: [float(x) for x in openclip_vector],
                }
            )
        if rows:
            self.client.upsert(collection_name=self.collection_name, data=rows)
            try:
                self.client.flush(collection_name=self.collection_name)
            except TypeError:
                self.client.flush(self.collection_name)

    def search(
        self,
        vector_field: str,
        vector: Sequence[float],
        top_k: int,
        filter_expr: str | None = None,
    ) -> List[RetrievalCandidate]:
        top_k = max(1, min(int(top_k), int(get_config_value(self.config, "zilliz.max_search_top_k", 1024))))
        output_fields = [
            "keyframe_id",
            "video_id",
            "shot_id",
            "timestamp_raw",
            "frame_index_raw",
            "model_version",
            "config_version",
        ]
        search_kwargs: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [[float(x) for x in vector]],
            "anns_field": vector_field,
            "limit": top_k,
            "output_fields": output_fields,
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr
        results = self.client.search(
            **search_kwargs,
        )
        candidates: List[RetrievalCandidate] = []
        for rank, hit in enumerate(results[0] if results else [], start=1):
            entity = dict(hit.get("entity") or {})
            score = float(hit.get("distance", hit.get("score", 0.0)))
            candidates.append(RetrievalCandidate.from_mapping(entity, score=score, rank=rank))
        return candidates

    def fetch_keyframe_vectors(
        self,
        vector_field: str,
        filter_expr: str | None = None,
        batch_size: int | None = None,
    ) -> List[Tuple[RetrievalCandidate, List[float]]]:
        batch_size = int(batch_size or get_config_value(self.config, "zilliz.query_batch_size", 256))
        batch_size = max(1, batch_size)
        output_fields = [
            "keyframe_id",
            "video_id",
            "shot_id",
            "timestamp_raw",
            "frame_index_raw",
            "model_version",
            "config_version",
            vector_field,
        ]
        pairs: List[Tuple[RetrievalCandidate, List[float]]] = []
        offset = 0
        while True:
            query_kwargs: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "output_fields": output_fields,
                "limit": batch_size,
                "offset": offset,
            }
            if filter_expr is not None:
                query_kwargs["filter"] = filter_expr

            rows = self.client.query(**query_kwargs)
            if not rows:
                break
            for row in rows:
                item = dict(row)
                vector = item.pop(vector_field, None)
                if vector is None:
                    continue
                pairs.append(
                    (
                        RetrievalCandidate.from_mapping(item),
                        [float(x) for x in vector],
                    )
                )
            if len(rows) < batch_size:
                break
            offset += batch_size
        return pairs


def create_zilliz_collection(config: Dict[str, Any], drop_existing: bool = False) -> None:
    client = ZillizClient(config)
    client.create_collection(drop_existing=drop_existing)
