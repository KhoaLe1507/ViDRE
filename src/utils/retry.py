from __future__ import annotations

import functools
import time
from typing import Any, Callable, Iterable, Tuple, Type


def retry(
    attempts: int = 3,
    base_delay_sec: float = 1.0,
    max_delay_sec: float = 30.0,
    retry_on: Iterable[Type[BaseException]] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    retry_types: Tuple[Type[BaseException], ...] = tuple(retry_on)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_types as exc:
                    last_exc = exc
                    if attempt >= attempts:
                        break
                    delay = min(max_delay_sec, base_delay_sec * (2 ** (attempt - 1)))
                    time.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator

