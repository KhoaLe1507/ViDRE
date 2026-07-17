from __future__ import annotations

import contextlib
import time
from typing import Dict, Iterator


class LatencyTracker:
    def __init__(self) -> None:
        self.breakdown_ms: Dict[str, float] = {}
        self._total_start = time.perf_counter()

    @contextlib.contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.breakdown_ms[name] = self.breakdown_ms.get(name, 0.0) + elapsed_ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._total_start) * 1000.0

