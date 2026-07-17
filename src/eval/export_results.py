from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def write_metrics(output_dir: str | Path, metrics: Dict[str, Any]) -> Path:
    path = Path(output_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(metrics, writer, ensure_ascii=False, indent=2)
    return path


def write_per_query_results(output_dir: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(output_dir) / "per_query_results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            writer.write("\n")
    return path


def write_manifest(output_dir: str | Path, manifest: Dict[str, Any]) -> Path:
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(manifest, writer, ensure_ascii=False, indent=2)
    return path

