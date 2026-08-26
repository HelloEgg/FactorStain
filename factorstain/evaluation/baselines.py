from __future__ import annotations

from collections.abc import Callable


class BaselineRegistry:
    """Small extension point for preprocessing/feature-correction baselines."""
    def __init__(self) -> None:
        self._constructors: dict[str, Callable] = {}

    def register(self, name: str, constructor: Callable) -> None:
        if name in self._constructors:
            raise KeyError(f"Baseline {name!r} is already registered")
        self._constructors[name] = constructor

    def create(self, name: str, **kwargs):
        if name not in self._constructors:
            raise KeyError(f"Unknown baseline {name!r}; available: {sorted(self._constructors)}")
        return self._constructors[name](**kwargs)

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._constructors))


BASELINES = BaselineRegistry()
