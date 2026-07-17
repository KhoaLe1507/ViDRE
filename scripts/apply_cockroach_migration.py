from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.cockroach_client import CockroachClient
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--migration", default="migrations/001_init_cockroach.sql")
    args = parser.parse_args()
    client = CockroachClient(load_config(args.config))
    client.execute_sql_file(args.migration)
    client.close()
    print(f"Applied migration: {args.migration}")


if __name__ == "__main__":
    main()
