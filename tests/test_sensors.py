import pytest

from core import sensors


def test_sentinel2_spec():
    s = sensors.STANDARD_SENSORS["Sentinel-2"]
    assert s.bands == 13
    assert 10.0 in s.resolutions


def test_sentinel1_spec():
    s = sensors.STANDARD_SENSORS["Sentinel-1"]
    assert s.bands == 2
    assert 10.0 in s.resolutions


def test_cartosat_spec():
    s = sensors.STANDARD_SENSORS["Cartosat-2S"]
    assert s.bands == 5
    assert 0.6 in s.resolutions


def test_risat_spec():
    s = sensors.STANDARD_SENSORS["RISAT-1"]
    assert s.bands == 1
    assert 3.0 in s.resolutions
