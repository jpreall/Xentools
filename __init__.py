import sys

from . import io
from . import pl
from . import tools as tl

from .io.write.xenium import (
    write_xenium_gene_groups,
    _invert_xen_gene_list_dict,
)
from .pl import (
    bin_expression,
    create_binned_image,
    create_bins,
    plot_binned_greyscale,
    plot_binned_rgb,
)
from .xentools import (
    XenData,
    LazyTranscripts,
    ROI,
    ROIClass,
    ROICollection,
    read_xen_panel,
    read_json,
    create_polygon,
    read_ROI_from_csv,
    read_ROI_from_geojson,
    build_niches,
    build_spatial_graph,
    evaluate_niche_k_values,
    generate_palette,
    plot_palette,
    normalize_tp10k,
)

__all__ = [
    'io', 
    'pl',
    'tl',
    'write_xenium_gene_groups',
    '_invert_xen_gene_list_dict',
    'XenData', 
    'LazyTranscripts',
    'ROI',
    'ROIClass',
    'ROICollection',
    'read_xen_panel', 
    'read_json',  
    'create_bins', 
    'bin_expression', 
    'create_binned_image', 
    'create_polygon', 
    'read_ROI_from_csv',
    'read_ROI_from_geojson',
    'build_niches',
    'build_spatial_graph',
    'evaluate_niche_k_values',
    'generate_palette', 
    'plot_palette',
    'normalize_tp10k',
    'plot_binned_rgb',
    'plot_binned_greyscale',
    ]
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "io", "pl"]})
