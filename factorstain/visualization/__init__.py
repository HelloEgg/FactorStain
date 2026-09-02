from __future__ import annotations

__all__ = ["appearance_statistics", "plot_plism_summary", "plot_real_grid"]


def __getattr__(name: str):
    """Keep optional image-analysis dependencies lazy for feature-only utilities."""
    if name in __all__:
        from . import plism

        return getattr(plism, name)
    raise AttributeError(name)
