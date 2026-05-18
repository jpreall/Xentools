from __future__ import annotations

import pandas as pd


def _first_tile_bounds(trans):
    key, meta = next(iter(trans._tile_meta.items()))
    if {"x0", "x1", "y0", "y1"}.issubset(meta):
        return key, (meta["x0"], meta["x1"], meta["y0"], meta["y1"])

    dx, dy = trans._tile_size
    return key, (
        meta["x"] - dx / 2,
        meta["x"] + dx / 2,
        meta["y"] - dy / 2,
        meta["y"] + dy / 2,
    )


def test_lazy_transcripts_iter_tiles_matches_query(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    trans = xdata.trans
    _key, bounds = _first_tile_bounds(trans)

    tile_frames = list(
        trans.iter_tiles(
            xmin=bounds[0],
            xmax=bounds[1],
            ymin=bounds[2],
            ymax=bounds[3],
            quality="all",
        )
    )
    queried = trans.query(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=bounds[2],
        ymax=bounds[3],
        quality="all",
    )

    assert tile_frames
    assert all(isinstance(tile_key, str) for tile_key, _df in tile_frames)
    assert all(isinstance(df, pd.DataFrame) for _tile_key, df in tile_frames)
    assert sum(len(df) for _tile_key, df in tile_frames) == len(queried)


def test_lazy_transcripts_repr_and_lightweight_accessors(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    trans = xdata.trans
    text = repr(trans)

    assert "LazyTranscripts" in text
    assert "shape:" in text
    assert "genes/features:" in text
    assert "spatial extent:" in text
    assert "optional columns: z_location via include_z=True" in text
    assert "query with:" in text
    assert list(trans.columns) == ["x_location", "y_location", "feature_name"]
    assert len(trans.gene_names) == trans.n_genes
    assert trans.shape == (len(trans), len(trans.columns))


def test_lazy_transcripts_include_z_is_opt_in(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    trans = xdata.trans
    _key, bounds = _first_tile_bounds(trans)

    default = trans.query(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=bounds[2],
        ymax=bounds[3],
        quality="all",
    )
    with_z = trans.query(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=bounds[2],
        ymax=bounds[3],
        quality="all",
        include_z=True,
    )

    assert "z_location" not in default.columns
    assert list(with_z.columns) == ["x_location", "y_location", "z_location", "feature_name"]
    assert len(with_z) == len(default)
    assert with_z["z_location"].notna().all()


def test_lazy_transcripts_iter_tiles_can_include_z(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    trans = xdata.trans
    _key, bounds = _first_tile_bounds(trans)

    tile_frames = list(
        trans.iter_tiles(
            xmin=bounds[0],
            xmax=bounds[1],
            ymin=bounds[2],
            ymax=bounds[3],
            quality="all",
            include_z=True,
        )
    )

    assert tile_frames
    assert all("z_location" in df.columns for _tile_key, df in tile_frames)


def test_lazy_transcripts_query_diagnostics_report_tile_pruning(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    trans = xdata.trans
    _key, bounds = _first_tile_bounds(trans)

    diagnostic = trans.diagnose_query(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=bounds[2],
        ymax=bounds[3],
        quality="all",
    )

    assert diagnostic["candidate_tiles"] >= 1
    assert diagnostic["candidate_transcripts"] >= diagnostic["returned_transcripts"]
    assert 0.0 <= diagnostic["survival_fraction"] <= 1.0
    assert diagnostic["tile_index_mode"] in {"explicit", "sampled", "unknown"}
    assert diagnostic["tile_size"][0] > 0
    assert diagnostic["tile_size"][1] > 0


def test_lazy_transcripts_missing_gene_yields_no_tiles(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    trans = xdata.trans

    assert list(trans.iter_tiles(genes=["definitely_not_a_gene"])) == []
    diagnostic = trans.diagnose_query(genes=["definitely_not_a_gene"])
    assert diagnostic["candidate_tiles"] == 0
    assert diagnostic["returned_transcripts"] == 0


def test_lazy_transcripts_cache_info_and_clear_cache(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
    )
    trans = xdata.trans
    _key, bounds = _first_tile_bounds(trans)

    trans.clear_cache()
    assert trans.cache_info()["entries"] == 0

    trans.query(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=bounds[2],
        ymax=bounds[3],
        quality="all",
    )
    info = trans.cache_info()
    assert info["entries"] == 1
    assert info["bytes"] > 0
    assert info["max_bytes"] == 512_000_000

    trans.clear_cache()
    assert trans.cache_info()["entries"] == 0
    assert trans.cache_info()["bytes"] == 0


def test_lazy_transcripts_cache_byte_cap_can_skip_large_result(xenium_testdata):
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
        cache_max_bytes=1,
    )
    trans = xdata.trans
    _key, bounds = _first_tile_bounds(trans)

    trans.query(
        xmin=bounds[0],
        xmax=bounds[1],
        ymin=bounds[2],
        ymax=bounds[3],
        quality="all",
    )

    assert trans.cache_info()["entries"] == 0


def test_lazy_transcripts_cache_evicts_lru_entry(xenium_testdata):
    import pandas as pd
    import xentools

    xdata = xentools.XenData(
        str(xenium_testdata),
        verbose=False,
        transcript_source="zarr",
        cache_max_bytes=None,
    )
    trans = xdata.trans
    trans.clear_cache()

    df1 = pd.DataFrame({"x_location": [1.0], "y_location": [2.0], "feature_name": ["A"]})
    df2 = pd.DataFrame({"x_location": [3.0], "y_location": [4.0], "feature_name": ["B"]})
    n1 = int(df1.memory_usage(index=True, deep=True).sum())
    n2 = int(df2.memory_usage(index=True, deep=True).sum())

    trans.cache_max_bytes = n1 + n2 - 1
    trans._cache_query_result(("first",), df1)
    trans._cache_query_result(("second",), df2)

    info = trans.cache_info()
    assert info["entries"] == 1
    assert info["bytes"] == n2
    assert ("first",) not in trans._query_cache
    assert ("second",) in trans._query_cache
