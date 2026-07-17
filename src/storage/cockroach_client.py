from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.schemas import KeyframeRecord, ShotRecord, VideoRecord
from src.utils.config import get_config_value

T = TypeVar("T")
RETRYABLE_SQLSTATES = {"40001", "40P01"}


def _require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("psycopg[binary] is required for CockroachDB. Install dependencies from requirements.txt.") from exc
    return psycopg, dict_row


class CockroachClient:
    def __init__(self, config: Dict[str, Any]):
        psycopg, dict_row = _require_psycopg()
        url_env = get_config_value(config, "cockroach.database_url_env", "COCKROACH_DATABASE_URL")
        database_url = os.environ.get(url_env, "")
        if not database_url:
            raise RuntimeError(f"{url_env} must be set.")
        database_url = _normalize_cockroach_database_url(database_url)
        self.conn = psycopg.connect(database_url, row_factory=dict_row)
        self._psycopg = psycopg
        self.config = config

    def close(self) -> None:
        self.conn.close()

    def rollback(self) -> None:
        self.conn.rollback()

    def _run_db_op(self, operation: Callable[[], T], max_retries: int = 5) -> T:
        for attempt in range(max_retries):
            try:
                result = operation()
                self.conn.commit()
                return result
            except Exception as exc:
                self.conn.rollback()
                if _is_retryable_db_error(exc) and attempt < max_retries - 1:
                    time.sleep(min(2.0, 0.2 * (2**attempt)))
                    continue
                raise
        raise RuntimeError("Unreachable CockroachDB retry state.")

    def execute_sql_file(self, path: str | os.PathLike[str]) -> None:
        sql = Path(path).read_text(encoding="utf-8")

        def operation() -> None:
            with self.conn.cursor() as cur:
                cur.execute(sql)

        self._run_db_op(operation)

    def is_video_verified(self, video_id: str, model_version: str, config_version: str) -> bool:
        def operation() -> bool:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT processing_status
                    FROM videos
                    WHERE video_id = %s AND model_version = %s AND config_version = %s
                    """,
                    (video_id, model_version, config_version),
                )
                row = cur.fetchone()
            return bool(row and row["processing_status"] == "VERIFIED")

        return self._run_db_op(operation)

    def is_shot_verified(self, shot_id: str, model_version: str, config_version: str) -> bool:
        def operation() -> bool:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT processing_status
                    FROM shots
                    WHERE shot_id = %s AND model_version = %s AND config_version = %s
                    """,
                    (shot_id, model_version, config_version),
                )
                row = cur.fetchone()
            return bool(row and row["processing_status"] == "VERIFIED")

        return self._run_db_op(operation)

    def upsert_video(self, video: VideoRecord) -> None:
        def operation() -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPSERT INTO videos (
                        video_id, source_path, r2_raw_path, duration_raw, fps_raw,
                        width_raw, height_raw, checksum_raw, processing_status,
                        model_version, config_version, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    """,
                    (
                        video.video_id,
                        video.source_path,
                        video.r2_raw_path,
                        video.duration_raw,
                        video.fps_raw,
                        video.width_raw,
                        video.height_raw,
                        video.checksum_raw,
                        video.processing_status,
                        video.model_version,
                        video.config_version,
                    ),
                )

        self._run_db_op(operation)

    def upsert_shots(self, shots: Sequence[ShotRecord]) -> None:
        def operation() -> None:
            with self.conn.cursor() as cur:
                for shot in shots:
                    cur.execute(
                        """
                        UPSERT INTO shots (
                            shot_id, video_id, shot_index, shot_start_frame, shot_end_frame,
                            shot_start_time_raw, shot_end_time_raw, duration_raw,
                            processing_status, model_version, config_version, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                        """,
                        (
                            shot.shot_id,
                            shot.video_id,
                            shot.shot_index,
                            shot.shot_start_frame,
                            shot.shot_end_frame,
                            shot.shot_start_time_raw,
                            shot.shot_end_time_raw,
                            shot.duration_raw,
                            shot.processing_status,
                            shot.model_version,
                            shot.config_version,
                        ),
                    )

        self._run_db_op(operation)

    def upsert_keyframes(self, keyframes: Sequence[KeyframeRecord]) -> None:
        def operation() -> None:
            with self.conn.cursor() as cur:
                for keyframe in keyframes:
                    cur.execute(
                        """
                        UPSERT INTO keyframes (
                            keyframe_id, video_id, shot_id, frame_index_raw, timestamp_raw,
                            timestamp_in_shot, selection_reason, beit3_distance_prev,
                            beit3_distance_last_keyframe, zilliz_inserted, object_counts,
                            processing_status, model_version, config_version, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, now())
                        """,
                        (
                            keyframe.keyframe_id,
                            keyframe.video_id,
                            keyframe.shot_id,
                            keyframe.frame_index_raw,
                            keyframe.timestamp_raw,
                            keyframe.timestamp_in_shot,
                            keyframe.selection_reason,
                            keyframe.beit3_selection_distance_prev,
                            keyframe.beit3_selection_distance_last_keyframe,
                            keyframe.zilliz_inserted,
                            json.dumps(keyframe.object_counts or {}),
                            keyframe.processing_status,
                            keyframe.model_version,
                            keyframe.config_version,
                        ),
                    )

        self._run_db_op(operation)

    def mark_video_status(self, video_id: str, status: str, r2_raw_deleted_at: bool = False) -> None:
        deleted_expr = "now()" if r2_raw_deleted_at else "r2_raw_deleted_at"

        def operation() -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE videos
                    SET processing_status = %s,
                        r2_raw_deleted_at = {deleted_expr},
                        updated_at = now()
                    WHERE video_id = %s
                    """,
                    (status, video_id),
                )

        self._run_db_op(operation)

    def update_shot_proxy(self, shot_id: str, r2_proxy_path: str, proxy_checksum: str, status: str = "VERIFIED") -> None:
        def operation() -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE shots
                    SET r2_proxy_path = %s,
                        proxy_checksum = %s,
                        proxy_status = %s,
                        processing_status = %s,
                        updated_at = now()
                    WHERE shot_id = %s
                    """,
                    (r2_proxy_path, proxy_checksum, status, status, shot_id),
                )

        self._run_db_op(operation)

    def get_object_counts(self, keyframe_ids: Iterable[str]) -> Dict[str, Dict[str, int]]:
        ids = list(dict.fromkeys(keyframe_ids))
        if not ids:
            return {}
        placeholders = ", ".join(["%s"] * len(ids))

        def operation() -> Dict[str, Dict[str, int]]:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT keyframe_id, object_counts FROM keyframes WHERE keyframe_id IN ({placeholders})",
                    ids,
                )
                rows = cur.fetchall()
            return {row["keyframe_id"]: dict(row["object_counts"] or {}) for row in rows}

        return self._run_db_op(operation)

    def get_query_cache(self, query_hash: str, cache_version: str) -> Optional[Dict[str, Any]]:
        def operation() -> Optional[Dict[str, Any]]:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM query_cache
                    WHERE query_hash = %s AND cache_version = %s
                    """,
                    (query_hash, cache_version),
                )
                row = cur.fetchone()
            return dict(row) if row else None

        return self._run_db_op(operation)

    def upsert_query_cache(
        self,
        query_hash: str,
        query_text: str,
        cache_version: str,
        gemini_paraphrases: Optional[List[str]] = None,
        gemini_object_constraints: Optional[List[Dict[str, Any]]] = None,
        sd_prompt: Optional[str] = None,
        sd_image_paths_or_hashes: Optional[List[Dict[str, Any]]] = None,
        sd_seeds: Optional[List[int]] = None,
    ) -> None:
        def operation() -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPSERT INTO query_cache (
                        query_hash, query_text, gemini_paraphrases, gemini_object_constraints,
                        sd_prompt, sd_image_paths_or_hashes, sd_seeds, cache_version, updated_at
                    )
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s, now())
                    """,
                    (
                        query_hash,
                        query_text,
                        json.dumps(gemini_paraphrases or []),
                        json.dumps(gemini_object_constraints or []),
                        sd_prompt,
                        json.dumps(sd_image_paths_or_hashes or []),
                        json.dumps(sd_seeds or []),
                        cache_version,
                    ),
                )

        self._run_db_op(operation)

    def insert_eval_run(
        self,
        eval_run_id: str,
        dataset_name: str,
        dataset_path: str,
        config_version: str,
        model_version: str,
        latency_mode: str,
        branch_config: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        def operation() -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPSERT INTO eval_runs (
                        eval_run_id, dataset_name, dataset_path, config_version,
                        model_version, latency_mode, branch_config, metrics, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
                    """,
                    (
                        eval_run_id,
                        dataset_name,
                        dataset_path,
                        config_version,
                        model_version,
                        latency_mode,
                        json.dumps(branch_config),
                        json.dumps(metrics),
                    ),
                )

        self._run_db_op(operation)

    def insert_eval_query_result(self, eval_run_id: str, result: Dict[str, Any]) -> None:
        gt_span = result["gt_span"]

        def operation() -> None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPSERT INTO eval_query_results (
                        eval_run_id, query_id, query_text, gt_video_id, gt_start,
                        gt_end, top_retrieved_keyframes, hit_at_1, hit_at_5,
                        hit_at_10, latency_ms, error_message, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, now())
                    """,
                    (
                        eval_run_id,
                        result["query_id"],
                        result["query_text"],
                        result["gt_video_id"],
                        gt_span[0],
                        gt_span[1],
                        json.dumps(result["top_retrieved_keyframes"]),
                        result["hit_at_1"],
                        result["hit_at_5"],
                        result["hit_at_10"],
                        result["latency_ms"],
                        result.get("error"),
                    ),
                )

        self._run_db_op(operation)


class NullCockroachClient:
    def close(self) -> None:
        return None

    def __getattr__(self, name: str):
        def _missing(*args: Any, **kwargs: Any):
            raise RuntimeError("CockroachDB client is not configured.")

        return _missing


def _is_retryable_db_error(exc: Exception) -> bool:
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate in RETRYABLE_SQLSTATES:
        return True
    return exc.__class__.__name__ in {"SerializationFailure", "DeadlockDetected"}


def _normalize_cockroach_database_url(database_url: str) -> str:
    parts = urlsplit(database_url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    param_names = {key.lower() for key, _ in params}
    sslmode = next((value.lower() for key, value in params if key.lower() == "sslmode"), "")
    if sslmode in {"verify-ca", "verify-full"} and "sslrootcert" not in param_names:
        params.append(("sslrootcert", "system"))
        return urlunsplit(parts._replace(query=urlencode(params)))
    return database_url
