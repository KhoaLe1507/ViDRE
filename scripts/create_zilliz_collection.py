from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.zilliz_client import create_zilliz_collection
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    create_zilliz_collection(config, drop_existing=args.drop_existing)
    print("Zilliz collection is ready.")


if __name__ == "__main__":
    main()
