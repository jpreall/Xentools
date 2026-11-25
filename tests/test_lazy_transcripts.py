from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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


@pytest.fixture
def tiled_transcripts(tmp_path):
    import zarr

    from xentools.core.transcripts import LazyTranscripts, _open_zarr_group_compat

    path = tmp_path / "transcripts.zarr.zip"
    tiles = [
        (
            [
                (0.5, 0.5), (0.5, 0.5), (0.75, 0.75), (0.125, 0.5), (1.5, 2.875),
                (1.5, 1.5), (1.5, 1.5), (1.5, 0.125), (1.75, 2.75), (1.5, 1.5),
            ],
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 99],
            [(0, 1, 1, 5), (5, 6, 6, 8), (0, 0, 8, 9)],
        ),
        (
            [(2.25, 0.25), (3.75, 2.75), (3.875, 2.75), (2.5, 1.5)],
            [0, 1, 1, 2],
            [(0, 0, 0, 1), (0, 0, 1, 3), (3, 6, 0, 0)],
        ),
    ]
    with zarr.storage.ZipStore(str(path), mode="w") as store:
        root = _open_zarr_group_compat(zarr, store, mode="w", force_v2=True)
        grid = root.create_group("grids").create_group("0")
        for col, (coords, genes, offsets) in enumerate(tiles):
            tile = grid.create_group(f"{col},0")
            locations = np.column_stack(
                [np.asarray(coords, dtype=np.float32), np.zeros(len(coords), dtype=np.float32)]
            )
            gene_ids = np.asarray(genes, dtype=np.uint32)[:, None]
            gene_offsets = np.asarray(offsets, dtype=np.uint32)
            create = tile.create_array if hasattr(tile, "create_array") else tile.create_dataset
            create("location", data=locations, chunks=(3, 3))
            create("gene_identity", data=gene_ids, chunks=(4, 1))
            create("gene_offset", data=gene_offsets, chunks=gene_offsets.shape)

    return LazyTranscripts(path, ["A", "B", "C"], verbose=False)


@pytest.mark.parametrize("workers", [1, 2])
def test_lazy_rasterize_channels_preserves_quality_assignment_and_bounds(tiled_transcripts, monkeypatch, workers):
    from zipfile import ZipFile, ZipInfo

    trans = tiled_transcripts
    bounds = (0.25, 3.75, 0.25, 2.75)
    channels = [["A", "B"], ["B", "C", "missing"]]
    high = np.zeros((3, 4, 2), dtype=np.float32)
    high[0, 0, 0] = 2
    high[0, 2, 0] = 1
    high[1, 1, 1] = 1
    high[2, 1, 1] = 1
    high[2, 3, 1] = 1
    low = np.zeros_like(high)
    low[0, 0, 0] = 1
    low[1, 1, 1] = 1
    low[1, 2, 1] = 1

    shared_chunk = "grids/0/0,0/gene_identity/1.0"
    chunk_reads = []
    original_open = ZipFile.open

    def record_chunk_reads(archive, name, *args, **kwargs):
        member = name.filename if isinstance(name, ZipInfo) else name
        if archive.filename == trans._path and member == shared_chunk:
            chunk_reads.append(member)
        return original_open(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", record_chunk_reads)
    for quality, expected in [("high", high), ("low", low), ("all", high + low)]:
        chunk_reads.clear()
        result = trans.rasterize_channels(channels, bounds, quality=quality, workers=workers)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float32
        assert result.flags.c_contiguous
        # High-quality ranges [1, 5) and [6, 9) both touch gene rows [4, 8).
        if quality == "high":
            assert chunk_reads == [shared_chunk]

    all_genes = trans.rasterize_channels([None], bounds, quality="high", workers=workers)
    np.testing.assert_array_equal(all_genes, (high + low).sum(axis=2, keepdims=True))
    missing = trans.rasterize_channels([["missing"]], bounds, workers=workers)
    np.testing.assert_array_equal(missing, np.zeros((3, 4, 1), dtype=np.float32))

    # The last low-quality range extends past the end of its tile.
    without_b = high + low
    without_b[1, 1, 1] = 0
    without_b[2, 3, 1] = 0
    clipped = trans.rasterize_channels([["A"], ["C"]], bounds, quality="all", workers=workers)
    np.testing.assert_array_equal(clipped, without_b)


def test_lazy_rasterize_worker_failure_waits_before_closing_store(tiled_transcripts, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import zarr

    trans = tiled_transcripts
    monkeypatch.setattr(trans, "_candidate_tiles", lambda *bounds: ["0,0", "1,0"])
    second_started = Event()
    allow_second_to_finish = Event()
    second_finished = Event()
    closed_after_workers = []
    select = zarr.Array.get_orthogonal_selection
    close = zarr.storage.ZipStore.close
    shutdown = ThreadPoolExecutor.shutdown

    def select_with_failure(array, *args, **kwargs):
        if array.path == "grids/0/0,0/location":
            assert second_started.wait(timeout=5)
            raise RuntimeError("tile read failed")
        if array.path == "grids/0/1,0/location":
            second_started.set()
            assert allow_second_to_finish.wait(timeout=5)
            result = select(array, *args, **kwargs)
            second_finished.set()
            return result
        return select(array, *args, **kwargs)

    def release_worker_at_shutdown(pool, *args, **kwargs):
        allow_second_to_finish.set()
        return shutdown(pool, *args, **kwargs)

    def record_store_close(store):
        closed_after_workers.append(second_finished.is_set())
        return close(store)

    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", select_with_failure)
    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", release_worker_at_shutdown)
    monkeypatch.setattr(zarr.storage.ZipStore, "close", record_store_close)
    with pytest.raises(RuntimeError, match="tile read failed"):
        trans.rasterize_channels([None], (0.25, 3.75, 0.25, 2.75), workers=2)

    assert closed_after_workers == [True]


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
