from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


VERIFIED = "VERIFIED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"
PENDING = "PENDING"
PROCESSING = "PROCESSING"


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    source_path: str
    duration_raw: float
    fps_raw: float
    width_raw: int
    height_raw: int
    checksum_raw: Optional[str]
    model_version: str
    config_version: str
    r2_raw_path: Optional[str] = None
    processing_status: str = PENDING

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShotRecord:
    video_id: str
    shot_id: str
    shot_index: int
    shot_start_frame: int
    shot_end_frame: int
    shot_start_time_raw: float
    shot_end_time_raw: float
    fps_raw: float
    duration_raw: float
    model_version: str
    config_version: str
    processing_status: str = PENDING
    skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KeyframeRecord:
    keyframe_id: str
    video_id: str
    shot_id: str
    shot_index: int
    frame_index_raw: int
    timestamp_raw: float
    timestamp_in_shot: float
    selection_reason: str
    beit3_selection_distance_prev: Optional[float]
    beit3_selection_distance_last_keyframe: Optional[float]
    model_version: str
    config_version: str
    object_counts: Dict[str, int] = field(default_factory=dict)
    zilliz_inserted: bool = False
    processing_status: str = PENDING

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCandidate:
    keyframe_id: str
    video_id: str
    shot_id: str
    timestamp_raw: float
    frame_index_raw: int
    score: float
    rank: int = 0
    object_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: Dict[str, Any], score: float = 0.0, rank: int = 0) -> "RetrievalCandidate":
        metadata = dict(item.get("metadata") or {})
        source = {**metadata, **{k: v for k, v in item.items() if k != "metadata"}}
        return cls(
            keyframe_id=str(source["keyframe_id"]),
            video_id=str(source["video_id"]),
            shot_id=str(source.get("shot_id", "")),
            timestamp_raw=float(source.get("timestamp_raw", 0.0)),
            frame_index_raw=int(source.get("frame_index_raw", -1)),
            score=float(score if score is not None else source.get("score", 0.0)),
            rank=int(rank),
            object_counts=dict(source.get("object_counts") or {}),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = float(self.score)
        return data

    def to_eval_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "timestamp_raw": self.timestamp_raw,
            "frame_index_raw": self.frame_index_raw,
            "score": self.score,
        }


@dataclass
class SearchResponse:
    query: str
    results: List[RetrievalCandidate]
    latency_ms: float
    latency_breakdown_ms: Dict[str, float] = field(default_factory=dict)
    branch_debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [candidate.to_dict() for candidate in self.results],
            "latency_ms": self.latency_ms,
            "latency_breakdown_ms": self.latency_breakdown_ms,
            "branch_debug": self.branch_debug,
        }


@dataclass(frozen=True)
class QuerySample:
    query_id: str
    video_id: str
    query_text: str
    gt_span: List[float]
    duration: float

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "QuerySample":
        query_text = data.get("query_text", data.get("query"))
        gt_span = data.get("gt_span", data.get("span"))
        if query_text is None:
            raise KeyError("query_text")
        if gt_span is None:
            raise KeyError("gt_span")
        return cls(
            query_id=str(data["query_id"]),
            video_id=str(data["video_id"]),
            query_text=str(query_text),
            gt_span=[float(gt_span[0]), float(gt_span[1])],
            duration=float(data.get("duration", 0.0)),
        )
