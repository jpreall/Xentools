"""User-facing gene-set picker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .filtering import (
    coverage_summary,
    filter_to_data,
    infer_database_species,
    infer_xendata_species,
)
from .library import GeneSetLibrary, load_curated_gene_sets
from .orthologs import convert_gene_sets_species

_SPLAT_COLORS = ("red", "green", "blue")


@dataclass
class _Selection:
    label: str
    database: str
    gene_set: str
    raw_genes: list[str]
    source_species: str


class GeneSetPicker:
    """
    Select gene sets and prepare them for a reference ``XenData`` object.

    The main output is ``picker.gene_sets``: an unrestricted
    ``dict[str, list[str]]`` filtered to genes present in ``xdata.adata``.
    That dictionary can be passed directly to Scanpy dotplots or to Xentools
    plotting functions.
    """

    def __init__(
        self,
        xdata,
        *,
        library: GeneSetLibrary | None = None,
        path=None,
        source_species: Literal["auto", "human", "mouse"] = "auto",
        target_species: Literal["auto", "human", "mouse"] = "auto",
        ortholog_strategy: Literal["all", "one_to_one", "best"] = "best",
        mouse_case: Literal["mgi", "upper", "lower"] | None = "mgi",
        min_genes: int = 1,
        cache_path=None,
        ortholog_mapping: dict[str, list[str]] | None = None,
    ):
        self.xdata = xdata
        self.library = library if library is not None else GeneSetLibrary(path) if path is not None else load_curated_gene_sets()
        self.source_species = source_species
        self.target_species = infer_xendata_species(xdata) if target_species == "auto" else target_species
        self.ortholog_strategy = ortholog_strategy
        self.mouse_case = mouse_case
        self.min_genes = int(min_genes)
        self.cache_path = cache_path
        self.ortholog_mapping = ortholog_mapping
        self._selected: dict[str, _Selection] = {}
        self._gene_sets: dict[str, list[str]] = {}
        self._converted_gene_sets: dict[str, list[str]] = {}
        self._coverage = pd.DataFrame()
        self._widget = None
        self._widget_state = {}
        self._sync_xdata()

    @property
    def databases(self) -> list[str]:
        """Available source databases."""
        return self.library.databases

    @property
    def raw_gene_sets(self) -> dict[str, list[str]]:
        """Selected gene sets before species conversion and dataset filtering."""
        return {label: list(sel.raw_genes) for label, sel in self._selected.items()}

    @property
    def converted_gene_sets(self) -> dict[str, list[str]]:
        """Selected gene sets after species conversion but before data filtering."""
        return dict(self._converted_gene_sets)

    @property
    def gene_sets(self) -> dict[str, list[str]]:
        """Selected gene sets converted as needed and filtered to ``xdata``."""
        return dict(self._gene_sets)

    @property
    def genes(self) -> list[str]:
        """Flat deduplicated union of genes in ``picker.gene_sets``."""
        seen = set()
        out = []
        for genes in self._gene_sets.values():
            for gene in genes:
                if gene not in seen:
                    out.append(gene)
                    seen.add(gene)
        return out

    @property
    def coverage(self) -> pd.DataFrame:
        """Per-selection coverage after conversion/filtering."""
        return self._coverage.copy()

    @property
    def dropped(self) -> pd.DataFrame:
        """Selected gene sets dropped because too few genes remained."""
        if self._coverage.empty:
            return self._coverage.copy()
        return self._coverage[~self._coverage["retained"]].copy()

    def search(self, query: str, *, database: str | None = None) -> pd.DataFrame:
        """Search available gene-set names."""
        return self.library.search(query, database=database)

    def add(self, database: str, gene_set: str, *, label: str | None = None):
        """
        Add one gene set by database and name.

        ``label`` controls the output key in ``picker.gene_sets``. If omitted,
        the gene-set name is used unless already selected, in which case the
        database name is prefixed to avoid a collision.
        """
        raw = self.library.get(database, gene_set)
        label = self._make_label(database, gene_set, label)
        source = self._resolve_source_species(database, raw)
        self._selected[label] = _Selection(label, database, gene_set, raw, source)
        self._refresh()
        self._refresh_widget_selection()
        return self

    def add_many(self, database: str, gene_sets, *, labels: dict[str, str] | None = None):
        """Add multiple gene sets from one database."""
        for gene_set in gene_sets:
            self.add(database, gene_set, label=None if labels is None else labels.get(gene_set))
        return self

    def remove(self, label: str):
        """Remove one selected gene set by output label."""
        self._selected.pop(label, None)
        self._refresh()
        self._refresh_widget_selection()
        return self

    def clear(self):
        """Remove all selected gene sets."""
        self._selected.clear()
        self._refresh()
        self._refresh_widget_selection()
        return self

    def close(self):
        """Close the underlying gene-set library."""
        self.library.close()

    def show(self):
        """
        Show the interactive picker in Jupyter when widgets are available.

        If ``ipywidgets`` or IPython display machinery is unavailable, this
        prints a short fallback message and returns ``self``. The picker remains
        fully usable via ``search()``, ``add()``, ``remove()``, and ``clear()``.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except Exception:
            print(
                "Interactive widgets are unavailable. Use picker.search(...) "
                "and picker.add(database, gene_set) instead."
            )
            return None

        if self._widget is None:
            self._build_widget(widgets)
        display(self._widget)
        return None

    def _make_label(self, database: str, gene_set: str, label: str | None):
        if label is not None:
            return label
        if gene_set not in self._selected:
            return gene_set
        return f"{database}: {gene_set}"

    def _resolve_source_species(self, database: str, genes: list[str]):
        if self.source_species != "auto":
            return self.source_species
        return infer_database_species(database, genes)

    def _refresh(self):
        converted = {}
        for label, selection in self._selected.items():
            genes = {label: selection.raw_genes}
            if (
                selection.source_species in {"human", "mouse"}
                and self.target_species in {"human", "mouse"}
                and selection.source_species != self.target_species
            ):
                genes = convert_gene_sets_species(
                    genes,
                    source_species=selection.source_species,
                    target_species=self.target_species,
                    strategy=self.ortholog_strategy,
                    mouse_case=self.mouse_case,
                    min_genes=0,
                    cache_path=self.cache_path,
                    ortholog_mapping=self.ortholog_mapping,
                )
            converted.update(genes)

        self._converted_gene_sets = converted
        if not hasattr(self.xdata, "adata") or self.xdata.adata is None:
            raise ValueError("xdata must have an AnnData object at xdata.adata.")
        self._gene_sets = filter_to_data(
            converted,
            self.xdata.adata.var_names,
            min_genes=self.min_genes,
        )
        self._coverage = coverage_summary(converted, self._gene_sets)
        self._sync_xdata()
        self._refresh_widget_status()

    def _sync_xdata(self):
        if self.xdata is not None:
            self.xdata.picked_gene_sets = self.gene_sets

    def _build_widget(self, widgets):
        dbs = self.databases
        default_db = "MSigDB_Hallmark_2020" if "MSigDB_Hallmark_2020" in dbs else dbs[0]
        db_dropdown = widgets.Dropdown(options=dbs, value=default_db, description="Database")
        search_box = widgets.Text(placeholder="Filter gene sets")
        results = widgets.SelectMultiple(options=[], rows=14, layout=widgets.Layout(width="420px"))
        selected = widgets.SelectMultiple(options=[], rows=14, layout=widgets.Layout(width="420px"))
        match_label = widgets.HTML()
        status = widgets.HTML()
        selected_label = widgets.HTML(value="<b>Selected gene sets</b>")
        splat_status = widgets.HTML()
        splat_output = widgets.Output()
        force_all_genes = widgets.Checkbox(
            value=False,
            description="Use all genes",
            indent=False,
            tooltip="Disable plot_splat's large-signature clipping for preview.",
            layout=widgets.Layout(width="120px"),
        )
        splat_button = widgets.Button(
            description="Preview splat",
            button_style="info",
            tooltip="Plot the first three retained gene sets as red, green, and blue.",
            layout=widgets.Layout(width="130px"),
        )

        def refresh_results(*_):
            query = search_box.value.strip()
            database = db_dropdown.value
            if query:
                table = self.search(query, database=database)
            else:
                table = self.library.list_sets(database).head(250)
            options = [
                (f"{row.gene_set} ({int(row.n_genes)} genes)", row.gene_set)
                for row in table.itertuples(index=False)
                if row.gene_set not in self._selected
            ]
            results.options = options
            match_label.value = f"{len(options):,} shown"

        def add_clicked(_):
            for gene_set in list(results.value):
                self.add(db_dropdown.value, gene_set)
            results.value = ()
            refresh_results()

        def remove_clicked(_):
            for label in list(selected.value):
                self.remove(label)
            selected.value = ()
            refresh_results()

        def clear_clicked(_):
            self.clear()
            selected.value = ()
            refresh_results()

        def preview_splat_clicked(_):
            if len(self._gene_sets) == 0:
                splat_status.value = "<span style='color:#b66;'>No retained gene sets to plot.</span>"
                return
            if len(self._gene_sets) > 3:
                splat_status.value = (
                    "<span style='color:#b66;'>Splat preview supports at most 3 retained gene sets. "
                    "Remove selections or call xdata.plot_splat(...) manually with a subset.</span>"
                )
                return
            try:
                splat_output.clear_output(wait=True)
                with splat_output:
                    from IPython.display import display
                    import matplotlib.pyplot as plt

                    ax = self.xdata.plot_splat(
                        genes=self._gene_sets,
                        force_all_genes=bool(force_all_genes.value),
                    )
                    display(ax.figure)
                    plt.close(ax.figure)
                splat_status.value = "<span style='color:#5a8;'>Splat preview rendered.</span>"
            except Exception as exc:
                splat_status.value = f"<span style='color:#b66;'>Splat preview failed: {exc}</span>"

        add_button = widgets.Button(description="Add ->", button_style="primary")
        remove_button = widgets.Button(description="<- Remove", button_style="warning")
        clear_button = widgets.Button(description="Clear", button_style="danger")
        add_button.on_click(add_clicked)
        remove_button.on_click(remove_clicked)
        clear_button.on_click(clear_clicked)
        splat_button.on_click(preview_splat_clicked)
        db_dropdown.observe(refresh_results, names="value")
        search_box.observe(refresh_results, names="value")

        self._widget_state = {
            "selected": selected,
            "selected_label": selected_label,
            "status": status,
            "splat_button": splat_button,
            "splat_status": splat_status,
            "splat_output": splat_output,
            "force_all_genes": force_all_genes,
        }
        controls = widgets.VBox([db_dropdown, search_box, match_label])
        buttons = widgets.VBox([add_button, remove_button, clear_button])
        selected_panel = widgets.VBox([selected_label, selected])
        panels = widgets.HBox([results, buttons, selected_panel])
        plot_controls = widgets.HBox([splat_button, force_all_genes, splat_status])
        self._widget = widgets.VBox([controls, panels, status, plot_controls, splat_output])
        refresh_results()
        self._refresh_widget_selection()
        self._refresh_widget_status()

    def _refresh_widget_selection(self):
        widget = self._widget_state.get("selected")
        if widget is not None:
            current_labels = set(widget.value)
            options = []
            current_values = []
            for label in self._selected:
                n_raw = len(self.raw_gene_sets.get(label, []))
                n_after = len(self._gene_sets.get(label, []))
                display = f"{label} ({n_after}/{n_raw} genes)"
                if n_after == 0:
                    display += " - dropped"
                options.append((display, label))
                if label in current_labels:
                    current_values.append(label)
            widget.options = options
            widget.value = tuple(current_values)

    def _refresh_widget_status(self):
        widget = self._widget_state.get("status")
        if widget is None:
            return
        n_raw = len(self._selected)
        n_kept = len(self._gene_sets)
        n_dropped = max(n_raw - n_kept, 0)
        n_genes = len(self.genes)
        if self._coverage.empty:
            detail = "No gene sets selected."
        else:
            rows = []
            for row in self._coverage.itertuples(index=False):
                status = "kept" if row.retained else "dropped"
                rows.append(
                    f"<li><b>{row.gene_set}</b>: {int(row.n_after)}/{int(row.n_before)} genes {status}</li>"
                )
            detail = "<ul style='margin: 4px 0 0 18px; padding: 0;'>" + "".join(rows) + "</ul>"
        widget.value = (
            f"<b>{n_kept}/{n_raw}</b> gene sets retained"
            + (f" (<b>{n_dropped}</b> dropped)" if n_dropped else "")
            + f"; <b>{n_genes:,}</b> unique genes in xdata."
            + detail
        )
        self._refresh_splat_preview_status()

    def _refresh_splat_preview_status(self):
        button = self._widget_state.get("splat_button")
        status = self._widget_state.get("splat_status")
        if button is None or status is None:
            return
        retained_items = list(self._gene_sets.items())
        n_retained = len(retained_items)
        button.disabled = n_retained == 0 or n_retained > 3

        if n_retained == 0:
            status.value = "<span style='color:#777;'>Select retained gene sets to preview as a splat.</span>"
            return
        chips = []
        for i, (name, genes) in enumerate(retained_items[:3]):
            color = _SPLAT_COLORS[i]
            chips.append(
                "<span style='display:inline-block;margin-right:10px;'>"
                f"<span style='display:inline-block;width:11px;height:11px;"
                f"background:{color};border-radius:50%;margin-right:4px;'></span>"
                f"{color}: <b>{name}</b> ({len(genes)} genes)"
                "</span>"
            )
        if n_retained > 3:
            status.value = (
                "".join(chips)
                + f"<br><span style='color:#b66;'>Splat preview disabled: {n_retained} retained gene sets selected; "
                "use 3 or fewer for RGB preview.</span>"
            )
        else:
            status.value = "".join(chips)

    def _ipython_display_(self):
        """
        Display the interactive picker when evaluated in an IPython notebook.

        ``__repr__`` intentionally remains side-effect free for logs, terminals,
        and debugging. IPython calls this rich-display hook when the picker is
        the final expression in a notebook cell.
        """
        self.show()

    def __repr__(self):
        return (
            "GeneSetPicker("
            f"{len(self._selected)} selected, "
            f"{len(self._gene_sets)} retained, "
            f"target_species={self.target_species!r})"
        )


def pick(
    xdata,
    *,
    path=None,
    library: GeneSetLibrary | None = None,
    source_species: Literal["auto", "human", "mouse"] = "auto",
    target_species: Literal["auto", "human", "mouse"] = "auto",
    ortholog_strategy: Literal["all", "one_to_one", "best"] = "best",
    mouse_case: Literal["mgi", "upper", "lower"] | None = "mgi",
    min_genes: int = 1,
    cache_path=None,
    ortholog_mapping: dict[str, list[str]] | None = None,
) -> GeneSetPicker:
    """Create a gene-set picker tied to a reference ``XenData`` object."""
    return GeneSetPicker(
        xdata,
        library=library,
        path=path,
        source_species=source_species,
        target_species=target_species,
        ortholog_strategy=ortholog_strategy,
        mouse_case=mouse_case,
        min_genes=min_genes,
        cache_path=cache_path,
        ortholog_mapping=ortholog_mapping,
    )
