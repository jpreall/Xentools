from __future__ import annotations


def test_curated_gene_set_library_opens():
    import xentools

    with xentools.gene_sets.load_curated_gene_sets() as lib:
        assert "MSigDB_Hallmark_2020" in lib.databases
        assert lib.info()["gene_sets"] > 0
        genes = lib.get("MSigDB_Hallmark_2020", "Hypoxia")
        assert "VEGFA" in genes


def test_picker_search_add_and_filter_to_xendata(xdata):
    import xentools

    picker = xentools.gene_sets.pick(xdata, source_species="mouse", target_species="mouse")
    hits = picker.search("Hypoxia", database="MSigDB_Hallmark_2020")
    assert not hits.empty

    picker.add("MSigDB_Hallmark_2020", "Hypoxia")
    assert "Hypoxia" in picker.raw_gene_sets
    assert isinstance(picker.gene_sets, dict)
    assert set(picker.gene_sets.get("Hypoxia", [])).issubset(set(xdata.adata.var_names))
    picker.close()


def test_picker_mouse_library_matches_dataset_symbol_case(xdata):
    import xentools

    picker = xentools.gene_sets.pick(xdata)
    row = picker.search("Fibroblast", database="Tabula_Muris").iloc[0]
    picker.add("Tabula_Muris", row.gene_set)

    assert row.gene_set in picker.gene_sets
    assert len(picker.gene_sets[row.gene_set]) > 0
    assert set(picker.gene_sets[row.gene_set]).issubset(set(xdata.adata.var_names))
    picker.close()


def test_picker_remove_and_clear_update_outputs(xdata):
    import xentools

    picker = xentools.gene_sets.pick(xdata)
    row = picker.search("Fibroblast", database="Tabula_Muris").iloc[0]
    picker.add("Tabula_Muris", row.gene_set)
    assert picker.gene_sets

    picker.remove(row.gene_set)
    assert picker.gene_sets == {}

    picker.add("Tabula_Muris", row.gene_set)
    assert picker.gene_sets
    picker.clear()
    assert picker.gene_sets == {}
    picker.close()


def test_picker_splat_preview_color_order_is_rgb(xdata):
    import xentools

    picker = xentools.gene_sets.pick(xdata)
    rows = picker.search("Fibroblast", database="Tabula_Muris").head(3)
    for row in rows.itertuples(index=False):
        picker.add("Tabula_Muris", row.gene_set)

    assert list(picker.gene_sets.keys()) == list(rows["gene_set"])
    picker.close()


def test_picker_widget_defaults_to_hallmark_when_available(xdata):
    import xentools

    picker = xentools.gene_sets.pick(xdata)
    picker.show()

    db_dropdown = picker._widget.children[0].children[0]
    assert db_dropdown.value == "MSigDB_Hallmark_2020"
    picker.close()


def test_picker_can_convert_then_filter_with_supplied_mapping(xdata):
    import xentools

    # Use the preparation function directly here so the test does not depend
    # on a particular curated library entry being present in the tiny Xenium panel.
    target_gene = str(xdata.adata.var_names[0])
    prepared = xentools.gene_sets.prepare_for_xendata(
        {"test": ["HUMAN_A", "HUMAN_B"]},
        xdata,
        source_species="human",
        target_species="mouse",
        ortholog_mapping={
            "HUMAN_A": [target_gene],
            "HUMAN_B": ["not_in_dataset"],
        },
        min_genes=1,
    )
    assert prepared == {"test": [target_gene]}


def test_build_gene_set_h5_from_enrichr_txt(tmp_path):
    from xentools.gene_sets import GeneSetLibrary, build_gene_set_h5

    source = tmp_path / "Example_DB.txt"
    source.write_text(
        "Set A\t\tGENE1\tGENE2\tGENE2\n"
        "Set B\t\tGENE2,0.5\tGENE3,1.0\n",
        encoding="utf-8",
    )
    out = build_gene_set_h5(tmp_path, tmp_path / "gene_sets.h5")

    with GeneSetLibrary(out) as lib:
        assert lib.databases == ["Example_DB"]
        assert lib.get("Example_DB", "Set A") == ["GENE1", "GENE2"]
        assert lib.get("Example_DB", "Set B") == ["GENE2", "GENE3"]
