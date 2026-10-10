"""The movie's mask (`mask.tif`/`mask.json`) against the one its results were made with."""

import numpy as np
import polars as pl

from napari_gemscape2.regions import Region, Regions
from napari_gemscape2.results import (
    build_manifest,
    has_mask,
    load_mask,
    load_regions,
    mask_is_stale,
    save_mask,
    write_result,
)


def _mask(value: int = 1, name: str = "cell") -> tuple[np.ndarray, Regions]:
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[2:5, 2:5] = value
    return labels, Regions(table={value: Region(name)})


def _write_results(result_dir, labels=None, regions=None) -> None:
    table = pl.DataFrame({"x": [1.0], "y": [1.0]})
    manifest = build_manifest(result_id=result_dir.name, source_image_path="movie.tif", params={})
    write_result(result_dir, table, table, manifest, labels=labels, regions=regions)


def test_no_mask(tmp_path):
    assert not has_mask(tmp_path)
    assert load_mask(tmp_path) == (None, None)
    assert not mask_is_stale(tmp_path)


def test_mask_before_results(tmp_path):
    labels, regions = _mask()
    save_mask(tmp_path, labels, regions)
    assert has_mask(tmp_path)
    got_labels, got_regions = load_mask(tmp_path)
    assert np.array_equal(got_labels, labels)
    assert got_regions.to_json() == regions.to_json()
    # No results yet, so nothing to be out of date.
    assert not mask_is_stale(tmp_path)


def test_results_with_the_same_mask_are_current(tmp_path):
    labels, regions = _mask()
    save_mask(tmp_path, labels, regions)
    _write_results(tmp_path, *load_mask(tmp_path))
    assert not mask_is_stale(tmp_path)


def test_editing_the_mask_keeps_the_results_mask(tmp_path):
    labels, regions = _mask()
    save_mask(tmp_path, labels, regions)
    _write_results(tmp_path, labels, regions)

    repainted, _ = _mask()
    repainted[0, 0] = 1
    save_mask(tmp_path, repainted, regions)
    assert mask_is_stale(tmp_path)
    assert np.array_equal(load_regions(tmp_path)[0], labels)
    assert np.array_equal(load_mask(tmp_path)[0], repainted)


def test_renaming_a_region_makes_results_stale(tmp_path):
    labels, regions = _mask()
    _write_results(tmp_path, labels, regions)
    save_mask(tmp_path, labels, Regions(table={1: Region("nucleus")}))
    assert mask_is_stale(tmp_path)


def test_masking_an_unmasked_run_makes_it_stale(tmp_path):
    _write_results(tmp_path)
    assert not mask_is_stale(tmp_path)
    save_mask(tmp_path, *_mask())
    assert mask_is_stale(tmp_path)


def test_erasing_the_mask_is_recorded(tmp_path):
    labels, regions = _mask()
    _write_results(tmp_path, labels, regions)
    save_mask(tmp_path, np.zeros_like(labels), Regions())
    assert not has_mask(tmp_path)
    assert load_mask(tmp_path) == (None, None)
    assert mask_is_stale(tmp_path)


def test_results_mask_stands_in_without_a_saved_mask(tmp_path):
    # A result dir from before masks were saved on their own.
    labels, regions = _mask()
    _write_results(tmp_path, labels, regions)
    assert has_mask(tmp_path)
    assert np.array_equal(load_mask(tmp_path)[0], labels)
    assert not mask_is_stale(tmp_path)
