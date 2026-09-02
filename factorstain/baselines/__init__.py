"""Acquisition baseline APIs and the M1 SOTA registry."""

from .base import AcquisitionMethod, FeatureHarmonizer, FitContext, MethodUnavailable
from .registry import BASELINES, BaselineSpec, get_baseline, resolve_methods

__all__ = [
    "BASELINES",
    "AcquisitionMethod",
    "BaselineSpec",
    "FeatureHarmonizer",
    "FitContext",
    "MethodUnavailable",
    "get_baseline",
    "resolve_methods",
]
