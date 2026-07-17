from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import get_config_value
from src.utils.hashing import file_sha256


def _require_boto3():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for Cloudflare R2. Install dependencies from requirements.txt.") from exc
    return boto3


class R2Client:
    def __init__(self, config: Dict[str, Any]):
        boto3 = _require_boto3()
        r2_cfg = config.get("r2", {})
        endpoint_url = os.environ.get(r2_cfg.get("endpoint_url_env", "R2_ENDPOINT_URL"), "")
        access_key_id = os.environ.get(r2_cfg.get("access_key_id_env", "R2_ACCESS_KEY_ID"), "")
        secret_access_key = os.environ.get(r2_cfg.get("secret_access_key_env", "R2_SECRET_ACCESS_KEY"), "")
        bucket = os.environ.get(r2_cfg.get("bucket_env", "R2_BUCKET"), "")
        if not endpoint_url or not access_key_id or not secret_access_key or not bucket:
            raise RuntimeError("R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_BUCKET must be set.")
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
        self.config = config

    def upload_file(self, local_path: str | os.PathLike[str], object_key: str, content_type: Optional[str] = None) -> str:
        checksum = file_sha256(local_path)
        extra_args: Dict[str, Any] = {"Metadata": {"sha256": checksum}}
        if content_type:
            extra_args["ContentType"] = content_type
        self.client.upload_file(str(local_path), self.bucket, object_key, ExtraArgs=extra_args)
        return object_key

    def download_file(self, object_key: str, local_path: str | os.PathLike[str]) -> Path:
        output = Path(local_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, object_key, str(output))
        return output

    def object_exists(self, object_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except Exception:
            return False

    def verify_checksum(self, object_key: str, expected_sha256: str) -> bool:
        head = self.client.head_object(Bucket=self.bucket, Key=object_key)
        metadata = head.get("Metadata") or {}
        return metadata.get("sha256") == expected_sha256

    def make_shot_proxy_key(self, video_id: str, shot_id: str, container: str = "mp4") -> str:
        prefix = get_config_value(self.config, "r2.proxy_prefix", "shot-proxies").strip("/")
        return f"{prefix}/{video_id}/{shot_id}.{container}"

