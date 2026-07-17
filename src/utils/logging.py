from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "vidre")

