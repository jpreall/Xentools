"""Core object-model modules for xentools.

Phase 0 note:
Implementation still primarily lives in `xentools.py`. These modules exist as
stable destinations for the staged refactor.
"""

from .boundaries import LazyBoundaryGeoDataFrame
from .rois import ROI, ROIClass, ROICollection
from .transcripts import LazyTranscripts

__all__ = [
    "LazyBoundaryGeoDataFrame",
    "LazyTranscripts",
    "ROI",
    "ROIClass",
    "ROICollection",
]
