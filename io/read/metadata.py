"""Xenium experiment metadata and image discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
import sys


def _load_local_module(module_name, relative_path):
    """Load a sibling module by file path when imported outside a package."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(__file__), relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from .images import _detect_linked_protein_images
except ImportError:
    _detect_linked_protein_images = _load_local_module(
        "_xentools_io_read_images_for_metadata",
        "images.py",
    )._detect_linked_protein_images


__all__ = [
    "XeniumMetadata",
    "read_experiment_xenium",
    "discover_xenium_images",
    "load_xenium_metadata",
]


@dataclass
class XeniumMetadata:
    """Common metadata needed to initialize a XenData object."""

    metadata: dict
    pixel_size: float
    name: str
    images: dict
    protein_images: dict | None


def read_experiment_xenium(xenium_folder):
    """Read ``experiment.xenium`` from a Xenium output folder."""
    xenium_file = os.path.join(xenium_folder, "experiment.xenium")
    with open(xenium_file) as f:
        return json.load(f)


def _experiment_name(metadata):
    keys = ["run_name", "slide_id", "region_name"]
    return "_".join(str(metadata[k]) for k in keys if k in metadata)


def discover_xenium_images(xenium_folder):
    """
    Discover standard DAPI morphology and linked morphology_focus images.

    Returns
    -------
    images, protein_images
        ``images`` is the compact layer registry used by ``XenData``.
        ``protein_images`` is the richer linked-image metadata dictionary, or
        None when no linked morphology_focus images were found.
    """
    images = {}
    morphology_path = os.path.join(xenium_folder, "morphology.ome.tif")
    if os.path.exists(morphology_path):
        images["DAPI"] = morphology_path

    protein_images = _detect_linked_protein_images(xenium_folder)
    if protein_images is not None:
        images["Protein"] = protein_images["folder"]
        if "DAPI" not in images:
            channel_names = protein_images.get("channel_names", [])
            dapi_matches = [
                i for i, name in enumerate(channel_names)
                if str(name).upper() == "DAPI"
            ]
            if dapi_matches:
                images["DAPI"] = protein_images["files"][dapi_matches[0]]

    return images, protein_images


def load_xenium_metadata(xenium_folder):
    """
    Load experiment metadata and discover image layers for a Xenium folder.
    """
    metadata = read_experiment_xenium(xenium_folder)
    pixel_size = float(metadata["pixel_size"])
    images, protein_images = discover_xenium_images(xenium_folder)
    return XeniumMetadata(
        metadata=metadata,
        pixel_size=pixel_size,
        name=_experiment_name(metadata),
        images=images,
        protein_images=protein_images,
    )
