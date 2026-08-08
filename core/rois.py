"""ROI data model and ROI import helpers for xentools."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

__all__ = [
    "ROI",
    "ROIClass",
    "ROIGroup",
    "ROICollection",
    "read_roi_from_csv",
    "read_roi_from_geojson",
    "_coerce_roi_geometry",
    "_coerce_roi",
    "_geometry_to_roi_points",
    "_normalize_roi_property",
    "_geojson_class_name",
    "_geojson_selection_name",
    "_make_roi_name",
    "_register_roi_record",
    "_select_geojson_features",
    "_match_geojson_row",
    "_import_roi_records",
    "_resolve_roi_selection_selector",
    "_roi_bounds_um",
    "_shapely_bounds",
]


@dataclass
class ROI:
    """
    Lightweight ROI wrapper with a canonical plotting-bounds convention.

    Public ``bounds`` always use ``(xmin, xmax, ymin, ymax)`` so plotting,
    rasterization, and image cropping code do not need to remember shapely's
    native ``(minx, miny, maxx, maxy)`` ordering.
    """

    geometry: Any
    name: str | None = None
    selection_name: str | None = None
    class_name: str | None = None
    poly_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(closed=True, fill=False, edgecolor="yellow", linewidth=1)
    )
    source: str | None = None
    units: str = "micron"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shapely_bounds(self):
        return tuple(map(float, self.geometry.bounds))

    @property
    def bounds(self):
        minx, miny, maxx, maxy = self.shapely_bounds
        return (minx, maxx, miny, maxy)

    @property
    def centroid(self):
        c = self.geometry.centroid
        return (float(c.x), float(c.y))

    @property
    def area(self):
        return float(self.geometry.area)

    @property
    def area_um2(self):
        return self.area

    @property
    def points(self):
        return _geometry_to_roi_points(self.geometry)

    def contains_points(self, x, y):
        from shapely import contains_xy

        return contains_xy(self.geometry, x, y)

    def crop_dataframe(self, df, x_col="x_location", y_col="y_location"):
        mask = self.contains_points(df[x_col].to_numpy(), df[y_col].to_numpy())
        return df.loc[mask].copy()

    def query_lazy_transcripts(self, lazy_transcripts, genes=None, quality="all"):
        xmin, xmax, ymin, ymax = self.bounds
        candidates = lazy_transcripts.query(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            genes=genes,
            quality=quality,
        )
        return self.crop_dataframe(candidates)

    def plot(self, ax, **kwargs):
        """
        Draw the ROI outline on an existing axes.

        ROI vertices are stored and drawn directly in the canonical ``global``
        coordinate system. Xentools spatial axes use a downward Y direction,
        so no crop-dependent coordinate reflection is required.
        """
        poly_kwargs = dict(self.poly_kwargs)
        poly_kwargs.update(kwargs)
        for geometry in _geometry_polygons(self.geometry):
            coordinates = np.asarray(geometry.exterior.coords, dtype=float).copy()
            polygon = pl.Polygon(coordinates, **poly_kwargs)
            ax.add_patch(polygon)
        return ax

    def to_pixels(self, pixel_size, name=None):
        from shapely.affinity import scale

        factor = 1.0 / float(pixel_size)
        return ROI.from_geometry(
            scale(self.geometry, xfact=factor, yfact=factor, origin=(0, 0)),
            name=self.name if name is None else name,
            selection_name=self.selection_name,
            class_name=self.class_name,
            poly_kwargs=dict(self.poly_kwargs),
            source=self.source,
            units="pixel",
            metadata=dict(self.metadata),
        )

    def to_microns(self, pixel_size, name=None):
        from shapely.affinity import scale

        factor = float(pixel_size)
        return ROI.from_geometry(
            scale(self.geometry, xfact=factor, yfact=factor, origin=(0, 0)),
            name=self.name if name is None else name,
            selection_name=self.selection_name,
            class_name=self.class_name,
            poly_kwargs=dict(self.poly_kwargs),
            source=self.source,
            units="micron",
            metadata=dict(self.metadata),
        )

    def to_geojson_feature(self, include_name=True):
        from shapely.geometry import mapping

        props = dict(self.metadata)
        if include_name and self.name is not None:
            props.setdefault("name", self.name)
        if self.selection_name is not None:
            props.setdefault("selection_name", self.selection_name)
        if self.class_name is not None:
            props.setdefault("class_name", self.class_name)
        if self.source is not None:
            props.setdefault("source", self.source)
        if self.units:
            props.setdefault("units", self.units)
        return {
            "type": "Feature",
            "properties": props,
            "geometry": mapping(self.geometry),
        }

    def bbox(self, name=None, poly_kwargs=None):
        from shapely.geometry import box as shapely_box

        xmin, xmax, ymin, ymax = self.bounds
        return ROI(
            geometry=shapely_box(xmin, ymin, xmax, ymax),
            name=self.name if name is None else name,
            selection_name=self.selection_name,
            class_name=self.class_name,
            poly_kwargs=self.poly_kwargs if poly_kwargs is None else poly_kwargs,
            source=self.source,
            units=self.units,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_geometry(
        cls,
        geometry,
        name=None,
        selection_name=None,
        class_name=None,
        poly_kwargs=None,
        source=None,
        units="micron",
        metadata=None,
    ):
        return cls(
            geometry=geometry,
            name=name,
            selection_name=selection_name,
            class_name=class_name,
            poly_kwargs=dict(closed=True, fill=False, edgecolor="yellow", linewidth=1)
            if poly_kwargs is None
            else poly_kwargs,
            source=source,
            units=units,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_bounds(
        cls,
        xmin,
        xmax,
        ymin,
        ymax,
        name=None,
        selection_name=None,
        class_name=None,
        poly_kwargs=None,
        source=None,
        units="micron",
        metadata=None,
    ):
        from shapely.geometry import box as shapely_box

        return cls.from_geometry(
            shapely_box(float(xmin), float(ymin), float(xmax), float(ymax)),
            name=name,
            selection_name=selection_name,
            class_name=class_name,
            poly_kwargs=poly_kwargs,
            source=source,
            units=units,
            metadata=metadata,
        )

    @classmethod
    def from_geojson(
        cls,
        geojson_path,
        feature=None,
        scale_factor=1.0,
        name=None,
        selection_name=None,
        class_name=None,
        poly_kwargs=None,
        source=None,
        units="micron",
        metadata=None,
    ):
        geometry, area_um2 = read_roi_from_geojson(
            geojson_path,
            feature=feature,
            scale_factor=scale_factor,
        )
        roi_name = name
        if roi_name is None and isinstance(feature, str):
            roi_name = feature
        roi = cls.from_geometry(
            geometry,
            name=roi_name,
            selection_name=selection_name,
            class_name=class_name,
            poly_kwargs=poly_kwargs,
            source=os.fspath(geojson_path) if source is None else source,
            units=units,
            metadata=metadata,
        )
        roi.metadata.setdefault("area_um2", area_um2)
        return roi

    def __getitem__(self, key):
        legacy = {
            "geometry": self.geometry,
            "centroid": self.centroid,
            "points": self.points,
            "poly_kwargs": self.poly_kwargs,
            "source": self.source,
            "area_um2": self.area_um2,
            "selection_name": self.selection_name,
            "class_name": self.class_name,
        }
        if key in legacy:
            return legacy[key]
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)


@dataclass
class ROIClass:
    name: str
    rois: list[ROI] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, roi: ROI):
        self.rois.append(roi)

    @property
    def bounds(self):
        if not self.rois:
            return (0.0, 0.0, 0.0, 0.0)
        xmin = min(r.bounds[0] for r in self.rois)
        xmax = max(r.bounds[1] for r in self.rois)
        ymin = min(r.bounds[2] for r in self.rois)
        ymax = max(r.bounds[3] for r in self.rois)
        return (xmin, xmax, ymin, ymax)

    @property
    def union(self):
        from shapely.ops import unary_union

        geom = unary_union([roi.geometry for roi in self.rois]) if self.rois else None
        if geom is None:
            raise ValueError(f"ROI class '{self.name}' has no geometries.")
        return ROI.from_geometry(
            geom,
            name=self.name,
            class_name=self.name,
            poly_kwargs=self.rois[0].poly_kwargs if self.rois else None,
            source=self.rois[0].source if self.rois else None,
            metadata=dict(self.metadata),
        )

    def plot(self, ax, **kwargs):
        for roi in self.rois:
            roi.plot(ax, **kwargs)
        return ax

    def __len__(self):
        return len(self.rois)

    def __iter__(self):
        return iter(self.rois)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rois[key]
        for roi in self.rois:
            if roi.name == key or roi.selection_name == key:
                return roi
        raise KeyError(key)


@dataclass
class ROIGroup:
    """Named logical collection of ROIs with a combined union geometry."""

    name: str
    rois: list[ROI]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rois:
            raise ValueError("An ROIGroup must contain at least one ROI.")
        if len({roi.units for roi in self.rois}) > 1:
            raise ValueError("All ROIGroup members must use the same coordinate units.")

    @property
    def member_names(self):
        return [roi.name for roi in self.rois]

    @property
    def union(self):
        from shapely.ops import unary_union

        metadata = dict(self.metadata)
        metadata.update({"roi_group": self.name, "members": self.member_names})
        return ROI.from_geometry(
            unary_union([roi.geometry for roi in self.rois]),
            name=self.name,
            poly_kwargs=dict(self.rois[0].poly_kwargs),
            source=self.rois[0].source,
            units=self.rois[0].units,
            metadata=metadata,
        )

    @property
    def geometry(self):
        return self.union.geometry

    @property
    def bounds(self):
        return self.union.bounds

    @property
    def area(self):
        return self.union.area

    def contains_points(self, x, y):
        return self.union.contains_points(x, y)

    def plot(self, ax, **kwargs):
        for roi in self.rois:
            roi.plot(ax, **kwargs)
        return ax

    def to_geojson_feature(self, include_name=True):
        return self.union.to_geojson_feature(include_name=include_name)

    def __len__(self):
        return len(self.rois)

    def __iter__(self):
        return iter(self.rois)


def _short(value, width):
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _count_label(count, singular, plural=None):
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _format_roi_area(roi):
    area = float(roi.area)
    if roi.units == "micron":
        if area >= 1_000_000:
            return f"{area / 1_000_000:.2f} mm²"
        return f"{area:,.1f} µm²"
    if roi.units == "pixel":
        return f"{area:,.1f} px²"
    return f"{area:,.1f} {roi.units}²"


@dataclass(repr=False)
class ROICollection:
    classes: dict[str, ROIClass] = field(default_factory=dict)
    flat_named: dict[str, ROI] = field(default_factory=dict)
    groups: dict[str, ROIGroup] = field(default_factory=dict)

    def add(self, roi: ROI):
        key = str(roi.name) if roi.name is not None else f"roi_{len(self.flat_named)}"
        roi.name = key
        self.flat_named[key] = roi
        class_name = roi.class_name
        if class_name is not None:
            cls = self.classes.get(class_name)
            if cls is None:
                cls = ROIClass(name=class_name)
                self.classes[class_name] = cls
            cls.add(roi)

    def class_names(self):
        return list(self.classes.keys())

    def selection_names(self):
        return list(self.flat_named.keys())

    def all_rois(self):
        return list(self.flat_named.values())

    def create_group(self, name, selectors, *, metadata=None, overwrite=False):
        """Create a logical ROI group without modifying its member ROIs."""
        name = str(name)
        if name in self.flat_named or name in self.classes:
            raise ValueError(f"ROI group name '{name}' conflicts with an ROI selection or class.")
        if name in self.groups and not overwrite:
            raise ValueError(f"ROI group '{name}' already exists; pass overwrite=True to replace it.")
        if isinstance(selectors, (str, int, ROI, ROIClass, ROIGroup)):
            selectors = [selectors]
        members = [self.resolve(selector) for selector in selectors]
        group = ROIGroup(
            name=name,
            rois=members,
            metadata={} if metadata is None else dict(metadata),
        )
        self.groups[name] = group
        return group

    def group_names(self):
        return list(self.groups.keys())

    @property
    def union(self):
        from shapely.ops import unary_union

        rois = self.all_rois()
        if not rois:
            raise ValueError("ROI collection is empty.")
        geom = unary_union([roi.geometry for roi in rois])
        return ROI.from_geometry(
            geom,
            name="all_rois",
            poly_kwargs=rois[0].poly_kwargs if rois else None,
            source=rois[0].source if rois else None,
            metadata=self.summary(),
        )

    def get_union(self, class_name):
        return self.classes[class_name].union

    def plot(
        self,
        ax,
        level: Literal["selection", "class", "group"] = "selection",
        *,
        labels=False,
        label_kwargs=None,
        **kwargs,
    ):
        """
        Draw ROI selections, classes, or groups on an existing axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Existing image, transcript, or other spatial axes.
        level : {'selection', 'class', 'group'}
            Which collection level to draw.
        labels : bool, default False
            Add names at ROI centroids. Classes and groups are labeled once at
            the centroid of their union geometry.
        label_kwargs : dict, optional
            Keyword arguments forwarded to ``ax.text``. Defaults are derived
            from the outline color and may be overridden here.
        **kwargs
            Outline options forwarded to :meth:`ROI.plot`, including
            ``edgecolor`` and ``linewidth``.
        """
        if level == "selection":
            plot_items = list(self.flat_named.items())
            for _, roi in plot_items:
                roi.plot(ax, **kwargs)
        elif level == "class":
            plot_items = [(name, roi_class.union) for name, roi_class in self.classes.items()]
            for roi_class in self.classes.values():
                roi_class.plot(ax, **kwargs)
        elif level == "group":
            plot_items = [(name, roi_group.union) for name, roi_group in self.groups.items()]
            for roi_group in self.groups.values():
                roi_group.plot(ax, **kwargs)
        else:
            raise ValueError("level must be 'selection', 'class', or 'group'.")

        if labels:
            text_kwargs = {
                "color": kwargs.get("edgecolor", "yellow"),
                "fontsize": 9,
                "fontweight": "bold",
                "ha": "center",
                "va": "center",
                "clip_on": True,
            }
            if label_kwargs:
                text_kwargs.update(label_kwargs)
            for name, roi in plot_items:
                x, y = roi.centroid
                ax.text(x, y, str(name), **text_kwargs)
        return ax

    def items(self):
        return self.flat_named.items()

    def keys(self):
        return self.flat_named.keys()

    def values(self):
        return self.flat_named.values()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        self.classes.clear()
        self.flat_named.clear()
        self.groups.clear()

    def summary(self):
        n_classified = sum(roi.class_name is not None for roi in self.flat_named.values())
        return {
            "n_selections": len(self.flat_named),
            "n_classes": len(self.classes),
            "n_groups": len(self.groups),
            "n_classified": n_classified,
            "n_unclassified": len(self.flat_named) - n_classified,
            "classes": {name: len(cls.rois) for name, cls in self.classes.items()},
            "groups": {name: group.member_names for name, group in self.groups.items()},
        }

    def __repr__(self):
        summary = self.summary()
        lines = [
            "ROICollection",
            (
                f"  {_count_label(summary['n_selections'], 'selection')} · "
                f"{summary['n_classified']} classified · {summary['n_unclassified']} unclassified"
            ),
            (
                f"  {_count_label(summary['n_classes'], 'class', 'classes')} · "
                f"{_count_label(summary['n_groups'], 'group')}"
            ),
        ]
        if not self.flat_named and not self.groups:
            return "\n".join(lines + ["  (empty)"])

        memberships = {name: [] for name in self.flat_named}
        for group_name, group in self.groups.items():
            for member_name in group.member_names:
                if member_name in memberships:
                    memberships[member_name].append(group_name)

        if self.flat_named:
            lines.extend(["", "Selections", "  name                 class          groups               geometry        area"])
            selections = list(self.flat_named.items())
            for name, roi in selections[:12]:
                group_text = ", ".join(memberships[name]) or "—"
                lines.append(
                    f"  {_short(name, 20):<20} {_short(roi.class_name or '—', 14):<14} "
                    f"{_short(group_text, 20):<20} {_short(roi.geometry.geom_type, 15):<15} "
                    f"{_format_roi_area(roi)}"
                )
            if len(selections) > 12:
                lines.append(f"  … {len(selections) - 12} more")

        if self.groups:
            lines.extend(["", "Groups", "  name                 members  geometry        area"])
            group_items = list(self.groups.items())
            for name, group in group_items[:12]:
                union = group.union
                lines.append(
                    f"  {_short(name, 20):<20} {len(group):>7}  "
                    f"{_short(union.geometry.geom_type, 15):<15} {_format_roi_area(union)}"
                )
            if len(group_items) > 12:
                lines.append(f"  … {len(group_items) - 12} more")

        if self.classes or summary["n_unclassified"]:
            lines.extend(["", "Classes"])
            for name, roi_class in list(self.classes.items())[:12]:
                lines.append(f"  {_short(name, 20):<20} {_count_label(len(roi_class), 'selection')}")
            if summary["n_unclassified"]:
                lines.append(
                    f"  {'Unclassified':<20} "
                    f"{_count_label(summary['n_unclassified'], 'selection')}"
                )
        return "\n".join(lines)

    def _repr_html_(self):
        from html import escape

        summary = self.summary()
        memberships = {name: [] for name in self.flat_named}
        for group_name, group in self.groups.items():
            for member_name in group.member_names:
                if member_name in memberships:
                    memberships[member_name].append(group_name)

        parts = [
            "<div><strong>ROICollection</strong><br>",
            f"{escape(_count_label(summary['n_selections'], 'selection'))} &middot; ",
            f"{summary['n_classified']} classified &middot; ",
            f"{summary['n_unclassified']} unclassified<br>",
            f"{escape(_count_label(summary['n_classes'], 'class', 'classes'))} &middot; ",
            f"{escape(_count_label(summary['n_groups'], 'group'))}</div>",
        ]
        if self.flat_named:
            parts.append("<h4>Selections</h4><table><thead><tr><th>Name</th><th>Class</th><th>Groups</th><th>Geometry</th><th>Area</th></tr></thead><tbody>")
            selections = list(self.flat_named.items())
            for name, roi in selections[:12]:
                values = (
                    name,
                    roi.class_name or "—",
                    ", ".join(memberships[name]) or "—",
                    roi.geometry.geom_type,
                    _format_roi_area(roi),
                )
                parts.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
            if len(selections) > 12:
                parts.append(f"<tr><td colspan='5'><em>… {len(selections) - 12} more</em></td></tr>")
            parts.append("</tbody></table>")
        if self.groups:
            parts.append("<h4>Groups</h4><table><thead><tr><th>Name</th><th>Members</th><th>Geometry</th><th>Area</th></tr></thead><tbody>")
            for name, group in list(self.groups.items())[:12]:
                union = group.union
                values = (name, len(group), union.geometry.geom_type, _format_roi_area(union))
                parts.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
            parts.append("</tbody></table>")
        if self.classes or summary["n_unclassified"]:
            parts.append("<h4>Classes</h4><table><tbody>")
            for name, roi_class in list(self.classes.items())[:12]:
                parts.append(
                    f"<tr><td>{escape(name)}</td>"
                    f"<td>{escape(_count_label(len(roi_class), 'selection'))}</td></tr>"
                )
            if summary["n_unclassified"]:
                parts.append(
                    "<tr><td>Unclassified</td>"
                    f"<td>{escape(_count_label(summary['n_unclassified'], 'selection'))}</td></tr>"
                )
            parts.append("</tbody></table>")
        return "".join(parts)

    def import_file(self, roi_file, roi_name=None, pixel_size=1.0, scale_geojson=True, append=True):
        if not append:
            self.clear()
        return _import_roi_records(
            roi_store=self,
            roi_file=roi_file,
            roi_name=roi_name,
            pixel_size=pixel_size,
            scale_geojson=scale_geojson,
        )

    def to_geojson_feature_collection(self, level: Literal["selection", "class", "group"] = "selection"):
        if level == "selection":
            features = [roi.to_geojson_feature() for roi in self.flat_named.values()]
        elif level == "class":
            features = [roi_class.union.to_geojson_feature() for roi_class in self.classes.values()]
        elif level == "group":
            features = [roi_group.to_geojson_feature() for roi_group in self.groups.values()]
        else:
            raise ValueError("level must be 'selection', 'class', or 'group'.")
        return {"type": "FeatureCollection", "features": features}

    def resolve(self, selector):
        if isinstance(selector, ROI):
            return selector
        if isinstance(selector, ROIClass):
            return selector.union
        if isinstance(selector, ROIGroup):
            return selector.union
        if isinstance(selector, int):
            names = list(self.flat_named.keys())
            if selector < 0 or selector >= len(names):
                raise KeyError(selector)
            return self.flat_named[names[selector]]
        if isinstance(selector, str):
            if selector in self.flat_named:
                return self.flat_named[selector]
            if selector in self.classes:
                return self.classes[selector].union
            if selector in self.groups:
                return self.groups[selector].union
        raise KeyError(selector)

    def __getitem__(self, key):
        if key in self.flat_named:
            return self.flat_named[key]
        if key in self.classes:
            return self.classes[key]
        if key in self.groups:
            return self.groups[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.flat_named or key in self.classes or key in self.groups

    def __len__(self):
        return len(self.flat_named)

    def __iter__(self):
        return iter(self.flat_named)

    def __bool__(self):
        return bool(self.flat_named or self.classes or self.groups)


def read_roi_from_csv(csv_path, selection=None):
    csv_path = Path(csv_path)
    header_lines = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            else:
                break

    names_line = next((ln for ln in header_lines if ln.lower().startswith("#selection names")), None)
    areas_line = next((ln for ln in header_lines if "areas" in ln.lower()), None)

    def _parse_list_from_comment(line):
        return [s.strip() for s in line.split(":", 1)[1].split(",")] if line and ":" in line else []

    def _parse_float_list_from_comment(line):
        vals = _parse_list_from_comment(line)
        out = []
        for v in vals:
            v_clean = re.sub(r"[^\d.+\-eE]", "", v)
            try:
                out.append(float(v_clean))
            except ValueError:
                out.append(np.nan)
        return out

    header_names = _parse_list_from_comment(names_line)
    header_areas = _parse_float_list_from_comment(areas_line)
    area_map = {}
    if header_names and header_areas and len(header_names) == len(header_areas):
        area_map = dict(zip(header_names, header_areas))

    df = pd.read_csv(csv_path, comment="#")
    if not {"Selection", "X", "Y"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Selection', 'X', 'Y'.")

    unique_selections = df["Selection"].astype(str).unique().tolist()
    if selection is None:
        if len(unique_selections) == 1:
            selection = unique_selections[0]
        else:
            raise ValueError(
                "Multiple regions found. Please specify one of: "
                + ", ".join(unique_selections)
            )

    sel = str(selection)
    chosen = None
    if sel in unique_selections:
        chosen = sel
    else:
        ci_map = {s.lower(): s for s in unique_selections}
        if sel.lower() in ci_map:
            chosen = ci_map[sel.lower()]
        else:
            cands = [s for s in unique_selections if s.lower().startswith(sel.lower())]
            if not cands:
                cands = [s for s in unique_selections if sel.lower() in s.lower()]
            if len(cands) == 1:
                chosen = cands[0]
            else:
                raise ValueError(
                    f"Selection '{selection}' not uniquely matched. "
                    f"Available options: {', '.join(unique_selections)}"
                )

    sub = df[df["Selection"].astype(str) == chosen]
    if sub.empty:
        raise ValueError(f"No coordinates found for selection '{chosen}'.")

    coords = sub[["X", "Y"]].to_numpy(dtype=float)
    return coords, area_map.get(chosen)


def read_roi_from_geojson(geojson_path, feature=None, return_gdf=False, scale_factor=1.0):
    rois = gpd.read_file(geojson_path)
    if rois.empty:
        raise ValueError(f"No ROI features found in GeoJSON file: {geojson_path}")
    if "geometry" not in rois:
        raise ValueError(f"GeoJSON file does not contain a geometry column: {geojson_path}")

    rois = rois.loc[rois.geometry.notna()].copy()
    if rois.empty:
        raise ValueError(f"GeoJSON file contains no valid geometries: {geojson_path}")

    if scale_factor != 1.0:
        rois["geometry"] = rois.scale(xfact=scale_factor, yfact=scale_factor, origin=(0, 0))

    if feature is None:
        roi_geometry = rois.geometry.union_all()
    elif isinstance(feature, int):
        if feature < 0 or feature >= len(rois):
            raise IndexError(f"Feature index {feature} is out of bounds for {len(rois)} ROI feature(s).")
        roi_geometry = rois.geometry.iloc[feature]
    elif isinstance(feature, str):
        if "name" not in rois.columns:
            raise ValueError("GeoJSON feature selection by name requires a 'name' property.")
        matches = rois.loc[rois["name"].astype(str) == feature]
        if matches.empty:
            detected = ", ".join(rois["name"].dropna().astype(str).unique().tolist())
            raise ValueError(f'ROI feature "{feature}" not found. Detected names: {detected}')
        roi_geometry = matches.geometry.union_all()
    else:
        raise TypeError("feature must be None, a string feature name, or an integer feature index.")

    area_um2 = roi_geometry.area
    if return_gdf:
        return roi_geometry, area_um2, rois
    return roi_geometry, area_um2


def _coerce_roi_geometry(roi_like, selection=None, pixel_size=1.0, scale_geojson=True):
    from shapely.geometry import Polygon

    if isinstance(roi_like, ROI):
        return roi_like.geometry
    if isinstance(roi_like, (ROIClass, ROIGroup)):
        return roi_like.union.geometry
    if hasattr(roi_like, "geom_type"):
        return roi_like
    if isinstance(roi_like, str):
        if not os.path.exists(roi_like):
            raise FileNotFoundError(f"ROI file not found: {roi_like}")
        if roi_like.lower().endswith((".geojson", ".json")):
            scale_factor = pixel_size if scale_geojson else 1.0
            roi_geometry, _ = read_roi_from_geojson(roi_like, feature=selection, scale_factor=scale_factor)
            return roi_geometry
        roi_like, _ = read_roi_from_csv(roi_like, selection=selection)
    if isinstance(roi_like, np.ndarray):
        if roi_like.ndim != 2 or roi_like.shape[1] != 2:
            raise ValueError("ROI must be a 2D numpy array with shape (n, 2).")
        return Polygon(roi_like)
    if isinstance(roi_like, pd.DataFrame):
        if roi_like.shape[1] != 2:
            raise ValueError("ROI must be a DataFrame with 2 columns.")
        return Polygon(roi_like.values)
    raise TypeError(
        "ROI must be a shapely geometry, a path to a ROI file, a 2D numpy array, or a 2-column DataFrame."
    )


def _coerce_roi(roi_like, selection=None, pixel_size=1.0, scale_geojson=True, name=None, source=None, poly_kwargs=None):
    if isinstance(roi_like, ROI):
        return roi_like
    if isinstance(roi_like, (ROIClass, ROIGroup)):
        return roi_like.union
    geometry = _coerce_roi_geometry(
        roi_like,
        selection=selection,
        pixel_size=pixel_size,
        scale_geojson=scale_geojson,
    )
    return ROI.from_geometry(
        geometry,
        name=name,
        source=source,
        poly_kwargs=poly_kwargs,
    )


def _geometry_to_roi_points(geometry):
    if geometry.geom_type == "Polygon":
        return np.asarray(geometry.exterior.coords)
    if geometry.geom_type == "MultiPolygon":
        largest = max(geometry.geoms, key=lambda geom: geom.area)
        return np.asarray(largest.exterior.coords)
    raise TypeError(f"Unsupported ROI geometry type: {geometry.geom_type}")


def _geometry_polygons(geometry):
    """Return every component of a Polygon or MultiPolygon."""
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    raise TypeError(f"Unsupported ROI geometry type: {geometry.geom_type}")


def _normalize_roi_property(value):
    if isinstance(value, dict):
        for key in ("name", "label", "class", "classification", "value", "title"):
            if key in value and value[key] not in (None, ""):
                return str(value[key])
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _normalize_roi_property(value[0])
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _geojson_class_name(row):
    for key in (
        "class_name",
        "class",
        "classification",
        "annotation_class",
        "region_class",
        "group",
        "object_type",
        "type",
    ):
        if key in row.index:
            value = _normalize_roi_property(row.get(key))
            if value is not None:
                return value
    return None


def _geojson_selection_name(row, idx):
    for key in (
        "selection_name",
        "selection",
        "name",
        "label",
        "region_name",
        "annotation_name",
    ):
        if key in row.index:
            value = _normalize_roi_property(row.get(key))
            if value is not None:
                return value
    return f"selection_{idx}"


def _make_roi_name(class_name, selection_name, idx):
    if class_name and selection_name:
        return f"{class_name}:{selection_name}"
    if selection_name:
        return selection_name
    if class_name:
        return f"{class_name}_{idx}"
    return f"selection_{idx}"


def _register_roi_record(
    roi_store,
    name,
    geometry,
    selection_name=None,
    class_name=None,
    poly_kwargs=None,
    source=None,
    area_um2=None,
    metadata=None,
):
    roi = ROI.from_geometry(
        geometry,
        name=str(name),
        selection_name=selection_name,
        class_name=class_name,
        poly_kwargs=poly_kwargs,
        source=source,
        metadata={
            "area_um2": geometry.area if area_um2 is None else area_um2,
            **({} if metadata is None else metadata),
        },
    )
    if isinstance(roi_store, ROICollection):
        roi_store.add(roi)
    else:
        roi_store[str(name)] = roi


def _select_geojson_features(rois, roi_name=None):
    if roi_name is None:
        return [(str(row.get("name", f"feature_{idx}")), row.geometry) for idx, row in rois.iterrows()]

    selectors = roi_name if isinstance(roi_name, list) else [roi_name]
    selected = []
    used_labels = set()

    for selector in selectors:
        if isinstance(selector, int):
            if selector < 0 or selector >= len(rois):
                raise IndexError(f"Feature index {selector} is out of bounds for {len(rois)} ROI feature(s).")
            row = rois.iloc[selector]
            label = str(row.get("name", f"feature_{selector}"))
            if label not in used_labels:
                selected.append((label, row.geometry))
                used_labels.add(label)
            continue

        selector = str(selector)
        if "name" not in rois.columns:
            raise ValueError("GeoJSON feature selection by name requires a 'name' property.")
        matches = rois.loc[rois["name"].astype(str) == selector]
        if matches.empty:
            detected = ", ".join(rois["name"].dropna().astype(str).unique().tolist())
            raise ValueError(f'ROI feature "{selector}" not found. Detected names: {detected}')
        for _, row in matches.iterrows():
            label = str(row.get("name", selector))
            if label not in used_labels:
                selected.append((label, row.geometry))
                used_labels.add(label)

    return selected


def _match_geojson_row(row, selector):
    if selector is None:
        return True
    if isinstance(selector, int):
        return False
    selector = str(selector)
    candidates = {
        _normalize_roi_property(row.get("name")) if "name" in row.index else None,
        _normalize_roi_property(row.get("selection_name")) if "selection_name" in row.index else None,
        _normalize_roi_property(row.get("selection")) if "selection" in row.index else None,
        _geojson_selection_name(row, 0),
        _geojson_class_name(row),
    }
    candidates = {c for c in candidates if c}
    return selector in candidates


def _import_roi_records(roi_store, roi_file, roi_name=None, pixel_size=1.0, scale_geojson=True):
    roi_path = os.fspath(roi_file)
    if roi_path.lower().endswith((".geojson", ".json")):
        scale_factor = pixel_size if scale_geojson else 1.0
        _, _, rois = read_roi_from_geojson(roi_path, return_gdf=True, scale_factor=scale_factor)
        imported_names = []
        selectors = roi_name if isinstance(roi_name, list) else ([roi_name] if roi_name is not None else None)
        class_counts = {}
        for idx, row in rois.iterrows():
            if selectors is not None:
                matched = False
                for selector in selectors:
                    if isinstance(selector, int) and selector == idx:
                        matched = True
                        break
                    if _match_geojson_row(row, selector):
                        matched = True
                        break
                if not matched:
                    continue
            class_name = _geojson_class_name(row)
            selection_name = _geojson_selection_name(row, idx)
            if class_name:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if selection_name == f"selection_{idx}":
                    selection_name = f"{class_name}_{class_counts[class_name]:03d}"
            roi_name_full = _make_roi_name(class_name, selection_name, idx)
            metadata = {
                col: row[col]
                for col in rois.columns
                if col != "geometry" and row[col] is not None
            }
            _register_roi_record(
                roi_store,
                name=roi_name_full,
                geometry=row.geometry,
                selection_name=selection_name,
                class_name=class_name,
                source=roi_path,
                metadata=metadata,
            )
            imported_names.append(roi_name_full)
        return imported_names

    polydf = pd.read_csv(roi_path, comment="#")
    if "Selection" not in polydf.columns:
        raise KeyError(
            'The provided ROI CSV file does not contain a "Selection" column. Please ensure it is a valid selections file produced by Xenium Analyzer.'
        )

    polydf["Selection"] = polydf["Selection"].astype("str")
    detected_roi_names = polydf["Selection"].unique().tolist()

    if isinstance(roi_name, str):
        if roi_name not in detected_roi_names:
            raise ValueError(f'ROI name "{roi_name}" not found in file. Detected ROIs: {detected_roi_names}')
        selected_names = [roi_name]
    elif isinstance(roi_name, list):
        selected_names = [str(name) for name in roi_name]
        for name in selected_names:
            if name not in detected_roi_names:
                raise ValueError(f'ROI name "{name}" not found in file. Detected ROIs: {detected_roi_names}')
    elif roi_name is None:
        selected_names = detected_roi_names
    else:
        raise TypeError("CSV ROI selection must be None, a string ROI name, or a list of ROI names.")

    for name in selected_names:
        pts = polydf.loc[polydf["Selection"] == name, ["X", "Y"]].to_numpy()
        geometry = _coerce_roi_geometry(pts)
        _register_roi_record(
            roi_store,
            name=name,
            geometry=geometry,
            selection_name=name,
            source=roi_path,
        )

    return selected_names


def _resolve_roi_selection_selector(roi_collection, imported_names, selector):
    if selector is None or selector is False:
        return None
    if selector is True:
        if len(imported_names) != 1:
            raise ValueError(
                "crop_to_selection=True requires the ROI file to contain exactly one "
                f"imported selection; found {len(imported_names)}."
            )
        return imported_names[0]
    if isinstance(selector, int):
        if selector < 0 or selector >= len(imported_names):
            raise IndexError(
                f"crop_to_selection index {selector} is out of bounds for "
                f"{len(imported_names)} imported selection(s)."
            )
        return imported_names[selector]
    if isinstance(selector, str):
        return selector
    return selector


def _roi_bounds_um(roi_geometry):
    if isinstance(roi_geometry, ROI):
        return roi_geometry.bounds
    minx, miny, maxx, maxy = _shapely_bounds(roi_geometry)
    return (minx, maxx, miny, maxy)


def _shapely_bounds(roi_geometry):
    if isinstance(roi_geometry, ROI):
        return roi_geometry.shapely_bounds
    return tuple(map(float, roi_geometry.bounds))
