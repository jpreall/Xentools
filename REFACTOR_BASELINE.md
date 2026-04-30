# Refactor Baseline

## Phase 0 decisions

### Compatibility entrypoints

During the refactor:

- `xentools.py` remains the primary compatibility facade
- package-root `__init__.py` continues to re-export the current public API
- public `xentools.io` compatibility should continue to point to `io_utils`
  until the migration is complete

This means the early refactor phases should move implementation, not public
imports.

### Current user-facing imports to preserve

At minimum, preserve:

- `import xentools`
- `xentools.XenData`
- `xentools.LazyTranscripts`
- ROI classes and common helpers exposed from the current top-level module
- `xentools.io`

### Current package caveat

Package-style import (`import xentools` from the parent directory) was broken
before Phase 0 because `io_utils.__init__` still imported the deleted
`_read_zarr.py` module. Phase 0 removes that stale import so import stability can
be used as a meaningful baseline during the refactor.

## Smoke test plan

Run these after each refactor phase:

```bash
conda run -n xentools python -m compileall xentools.py __init__.py core analysis io plotting utils io_utils
conda run -n xentools python -c "import sys; sys.path.insert(0, '..'); import xentools; print(xentools.__file__)"
```

Representative functional smoke tests:

```python
import xentools
xdata = xentools.XenData("files/xenium_v1_testdata")
xdata.show_image("DAPI", level=2)
xdata.create_binned_adata()
print(xdata)
```

Additional checks to rerun when relevant code moves:

- ROI import/subset
- protein-channel image display
- boundary plotting
- explorer export

## Phase 0 deliverable

- internal module skeleton created
- compatibility stance documented
- broken package import baseline fixed
