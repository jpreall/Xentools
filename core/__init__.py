"""Core object-model modules for xentools.

Phase 0 note:
Implementation still primarily lives in `xentools.py`. These modules exist as
stable destinations for the staged refactor.
"""

from .boundaries import LazyBoundaryGeoDataFrame
from .coordinates import CoordinateSystem, TransformRegistry
from .images import AlignedImage
from .rois import ROI, ROIClass, ROICollection, ROIGroup
from .transcripts import LazyTranscripts
from .xendata import XenData

__all__ = [
    "LazyBoundaryGeoDataFrame",
    "AlignedImage",
    "CoordinateSystem",
    "TransformRegistry",
    "LazyTranscripts",
    "ROI",
    "ROIClass",
    "ROIGroup",
    "ROICollection",
    "XenData",
]
