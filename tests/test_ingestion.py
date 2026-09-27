"""Unit tests: GeoTIFF ingestion + GSD-from-metadata (resource.md GeoTIFF hardening)."""
import os

from core.ingestion import ingest_file

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "geotiff_sample.tif")


def test_geotiff_sample_exists():
    assert os.path.exists(SAMPLE), "samples/geotiff_sample.tif missing"


def test_geotiff_ingest_reports_format_and_bands():
    scene, report = ingest_file(SAMPLE, gsd=10.0, safe_name="geotiff_sample.tif")
    assert report.format == "TIFF"
    assert report.bands == 3
    assert report.status == "ok"


def test_gsd_from_geotiff_metadata_overrides_user_input():
    # The sample carries a ModelPixelScaleTag of 10 m/px; a caller passing a
    # different user GSD should still get the metadata value back, with the
    # source labelled accordingly (never silently trusting the form field
    # when the file itself says otherwise).
    scene, report = ingest_file(SAMPLE, gsd=999.0, safe_name="geotiff_sample.tif")
    assert report.gsd_m == 10.0
    assert "GeoTIFF" in report.gsd_source or "raster metadata" in report.gsd_source
    assert "pixel-scale" in report.resolution_note or "raster metadata" in report.resolution_note


def test_gsd_falls_back_to_user_input_without_metadata():
    from PIL import Image
    import tempfile
    import numpy as np

    arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plain.tif")
        Image.fromarray(arr, "RGB").save(p)
        scene, report = ingest_file(p, gsd=15.0, safe_name="plain.tif")
        assert report.gsd_m == 15.0
        assert report.gsd_source == "user input (GSD form field)"
