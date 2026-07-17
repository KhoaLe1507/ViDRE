from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_query_samples(video_level_path: str, output_path: str) -> List[Dict[str, Any]]:
    with open(video_level_path, "r", encoding="utf-8") as reader:
        video_level = json.load(reader)
    samples: List[Dict[str, Any]] = []
    for video_id, payload in sorted(video_level.items()):
        queries = payload.get("queries", [])
        spans = payload.get("spans", [])
        if len(queries) != len(spans):
            raise ValueError(f"queries/spans length mismatch for video_id={video_id}")
        for idx, (query, span) in enumerate(zip(queries, spans)):
            samples.append(
                {
                    "query_id": f"{video_id}_q{idx:03d}",
                    "video_id": video_id,
                    "query_text": query,
                    "gt_span": [float(span[0]), float(span[1])],
                    "duration": float(payload.get("duration", 0.0)),
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as writer:
        json.dump(samples, writer, ensure_ascii=False, indent=2)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    samples = build_query_samples(args.input, args.output)
    print(f"wrote {len(samples)} query-level samples to {args.output}")


if __name__ == "__main__":
    main()

