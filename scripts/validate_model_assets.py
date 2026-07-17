from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, validate_model_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    missing = validate_model_assets(load_config(args.config))
    if missing:
        print(json.dumps({"ok": False, "missing": missing}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True, "missing": {}}, indent=2))


if __name__ == "__main__":
    main()
