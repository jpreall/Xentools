"""
Interactive gene set picker for Jupyter notebooks.

Usage
-----
from enrichr_picker import GeneSetPicker

picker = GeneSetPicker("enrichr.h5")
picker.show()

# After making selections:
gene_sets = picker.selection   # {set_name: [genes], ...}
gene_list  = picker.genes      # flat deduplicated list of all selected genes
"""

from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

from build_enrichr_h5 import EnrichrH5, DEFAULT_OUT

# Databases with more sets than this require typing before results appear
_SHOW_ALL_THRESHOLD = 500
# Never show more than this many options in the results list (keeps widget snappy)
_MAX_DISPLAY = 300
# Characters required before searching a large database
_MIN_QUERY_CHARS = 2


class GeneSetPicker:
    """
    Interactive two-panel gene set selector.

    Parameters
    ----------
    h5_path : path to enrichr.h5 (defaults to enrichr.h5 in the same folder)
    default_db : database to pre-select on open

    Attributes
    ----------
    selection : dict[str, list[str]]
        Currently selected gene sets as {name: [genes]}.
    genes : list[str]
        Flat deduplicated union of genes across all selected sets.
    """

    def __init__(self, h5_path=DEFAULT_OUT, default_db: str | None = None):
        self._db = EnrichrH5(h5_path)
        self._selected: dict[str, list[str]] = {}   # name → gene list
        self._names_cache: dict[str, list[str]] = {}  # db → list of set names
        self._build_ui(default_db)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_names(self, db_name: str) -> list[str]:
        if db_name not in self._names_cache:
            grp = self._db._h5[db_name]
            self._names_cache[db_name] = list(self._db._decode_names(grp))
        return self._names_cache[db_name]

    def _refresh_results(self, *_):
        db_name = self._db_dropdown.value
        query = self._search_box.value.strip().lower()
        names = self._get_names(db_name)
        large = len(names) > _SHOW_ALL_THRESHOLD

        if large and len(query) < _MIN_QUERY_CHARS:
            self._results_list.options = []
            self._match_label.value = (
                f'<span style="color:#888;font-size:0.85em;">'
                f'{len(names):,} sets — type {_MIN_QUERY_CHARS}+ characters to search'
                f'</span>'
            )
            self._update_status()
            return

        if query:
            filtered = [n for n in names if query in n.lower()]
        else:
            filtered = names  # small db, no query → show all

        # Hide sets already in the right-hand panel
        filtered = [n for n in filtered if n not in self._selected]

        overflow = max(0, len(filtered) - _MAX_DISPLAY)
        shown = filtered[:_MAX_DISPLAY]

        self._results_list.options = shown

        count_str = f"{len(filtered):,} match{'es' if len(filtered) != 1 else ''}"
        if not query:
            count_str = f"{len(filtered):,} sets"
        overflow_str = f", showing first {_MAX_DISPLAY:,}" if overflow else ""
        self._match_label.value = (
            f'<span style="color:#555;font-size:0.85em;">'
            f'{count_str}{overflow_str}'
            f'</span>'
        )
        self._update_status()

    def _update_status(self):
        n_sets = len(self._selected)
        if n_sets == 0:
            self._status.value = '<span style="color:#aaa;font-size:0.85em;">No sets selected</span>'
            return
        unique_genes = len({g for genes in self._selected.values() for g in genes})
        self._status.value = (
            f'<b>{n_sets}</b> gene set{"s" if n_sets != 1 else ""} selected'
            f' &nbsp;·&nbsp; <b>{unique_genes:,}</b> unique genes'
        )

    def _on_db_change(self, *_):
        self._search_box.value = ""
        self._refresh_results()

    def _on_search_change(self, *_):
        self._refresh_results()

    def _on_add(self, *_):
        db_name = self._db_dropdown.value
        for name in self._results_list.value:  # currently highlighted in left panel
            name = str(name)
            if name and name not in self._selected:
                self._selected[name] = self._db.get(db_name, name)
        self._selected_list.options = list(self._selected.keys())
        self._refresh_results()  # removes added items from left panel

    def _on_remove(self, *_):
        for name in self._selected_list.value:
            self._selected.pop(name, None)
        self._selected_list.options = list(self._selected.keys())
        self._refresh_results()  # adds removed items back to left panel

    def _on_clear(self, *_):
        self._selected.clear()
        self._selected_list.options = []
        self._refresh_results()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, default_db: str | None):
        dbs = self._db.databases

        # ── top row: database selector ──────────────────────────────────
        self._db_dropdown = widgets.Dropdown(
            options=dbs,
            value=default_db if default_db in dbs else dbs[0],
            layout=widgets.Layout(width="420px"),
        )
        db_row = widgets.HBox(
            [widgets.Label("Database:", layout=widgets.Layout(width="70px")),
             self._db_dropdown],
        )

        # ── search row ──────────────────────────────────────────────────
        self._search_box = widgets.Text(
            placeholder="Filter gene sets…",
            layout=widgets.Layout(width="420px"),
        )
        search_row = widgets.HBox(
            [widgets.Label("Search:", layout=widgets.Layout(width="70px")),
             self._search_box],
        )

        # ── left panel: results ─────────────────────────────────────────
        self._match_label = widgets.HTML(value="")
        self._results_list = widgets.SelectMultiple(
            options=[],
            rows=14,
            layout=widgets.Layout(width="380px"),
        )
        left_panel = widgets.VBox(
            [self._match_label, self._results_list],
            layout=widgets.Layout(margin="0 8px 0 0"),
        )

        # ── middle: buttons ─────────────────────────────────────────────
        add_btn = widgets.Button(
            description="Add →",
            button_style="primary",
            layout=widgets.Layout(width="110px"),
        )
        remove_btn = widgets.Button(
            description="← Remove",
            button_style="warning",
            layout=widgets.Layout(width="110px"),
        )
        clear_btn = widgets.Button(
            description="Clear all",
            button_style="danger",
            layout=widgets.Layout(width="110px"),
        )
        btn_col = widgets.VBox(
            [add_btn, remove_btn, clear_btn],
            layout=widgets.Layout(
                justify_content="center",
                align_items="center",
                margin="20px 8px 0 8px",
            ),
        )

        # ── right panel: selection ──────────────────────────────────────
        selected_label = widgets.HTML(
            value='<span style="color:#555;font-size:0.85em;">Selected sets</span>'
        )
        self._selected_list = widgets.SelectMultiple(
            options=[],
            rows=14,
            layout=widgets.Layout(width="380px"),
        )
        right_panel = widgets.VBox(
            [selected_label, self._selected_list],
            layout=widgets.Layout(margin="0 0 0 8px"),
        )

        # ── status bar ──────────────────────────────────────────────────
        self._status = widgets.HTML(value="")

        # ── wire events ─────────────────────────────────────────────────
        self._db_dropdown.observe(self._on_db_change, names="value")
        self._search_box.observe(self._on_search_change, names="value")
        add_btn.on_click(self._on_add)
        remove_btn.on_click(self._on_remove)
        clear_btn.on_click(self._on_clear)

        # ── assemble ────────────────────────────────────────────────────
        panels = widgets.HBox([left_panel, btn_col, right_panel])
        self._widget = widgets.VBox(
            [db_row, search_row, panels, self._status],
            layout=widgets.Layout(padding="10px"),
        )

        # Initial population
        self._refresh_results()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self):
        """Render the picker in the current Jupyter cell."""
        display(self._widget)

    @property
    def selection(self) -> dict[str, list[str]]:
        """Selected gene sets as {name: [genes]}."""
        return dict(self._selected)

    @property
    def genes(self) -> list[str]:
        """Flat deduplicated union of all selected gene sets."""
        seen = set()
        out = []
        for gene_list in self._selected.values():
            for g in gene_list:
                if g not in seen:
                    seen.add(g)
                    out.append(g)
        return out

    def close(self):
        """Release the HDF5 file handle."""
        self._db.close()

    def __repr__(self):
        n = len(self._selected)
        return f"GeneSetPicker({n} set{'s' if n != 1 else ''} selected)"
