"""Backward-compat shim for modules that import from YOPO.config."""

from ..schema import DatasetManifest, TrainOutput, TRTOutput, YOPOConfig  # noqa: I001
from ..schema import config as cfg  # noqa: I001

__all__ = ['cfg', 'DatasetManifest', 'TrainOutput', 'TRTOutput', 'YOPOConfig']
