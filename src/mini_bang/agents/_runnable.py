from __future__ import annotations

from collections.abc import Callable
from typing import Any


try:
    from langchain.schema.runnable import RunnableLambda as RunnableLambda
except ImportError:

    class RunnableLambda:
        """Small fallback for baseline agents when langchain is not installed."""

        def __init__(self, func: Callable[[Any], Any]) -> None:
            self._func = func

        def invoke(self, payload: Any) -> Any:
            return self._func(payload)

        def __or__(self, other: Any) -> "_RunnablePipeline":
            return _RunnablePipeline([self, other])


    class _RunnablePipeline:
        def __init__(self, steps: list[Any]) -> None:
            self._steps = steps

        def invoke(self, payload: Any) -> Any:
            value = payload
            for step in self._steps:
                value = step.invoke(value)
            return value

        def __or__(self, other: Any) -> "_RunnablePipeline":
            return _RunnablePipeline([*self._steps, other])
