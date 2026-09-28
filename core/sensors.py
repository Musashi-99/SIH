from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SensorSpec:
    name: str
    sensor_type: str  # 'optical' | 'sar'
    bands: int
    band_names: List[str]
    resolutions: List[float]  # meters
    notes: str = ""


# A small registry of common sensors with conservative/default specs.
STANDARD_SENSORS: Dict[str, SensorSpec] = {
    "Sentinel-2": SensorSpec(
        name="Sentinel-2",
        sensor_type="optical",
        bands=13,
        band_names=[
            "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
        ],
        resolutions=[10.0, 20.0, 60.0],
        notes="Multispectral MSI (ESA) standard band set",
    ),
    "Sentinel-1": SensorSpec(
        name="Sentinel-1",
        sensor_type="sar",
        bands=2,
        band_names=["VV", "VH"],
        resolutions=[10.0],
        notes="C-band SAR GRD typical dual-polarisation (VV/VH)",
    ),
    "Cartosat-2S": SensorSpec(
        name="Cartosat-2S",
        sensor_type="optical",
        bands=5,
        band_names=["PAN", "B2", "B3", "B4", "B5"],
        resolutions=[0.6, 2.5],
        notes=(
            "High-resolution PAN (≈0.6m) + multispectral (~2.5m). "
            "Vendor/resolution may vary by product/processing."
        ),
    ),
    "RISAT-1": SensorSpec(
        name="RISAT-1",
        sensor_type="sar",
        bands=1,
        band_names=["HH"],
        resolutions=[3.0],
        notes=(
            "C-band SAR (RISAT-1) — mode-dependent resolution; default conservative value = 3m"
        ),
    ),
}


def get_sensor(name: str) -> SensorSpec:
    """Return a SensorSpec by name.

    Lookup is case-insensitive and tolerates underscores or dashes.
    Examples that resolve to the same spec: "SENTINEL_2", "sentinel-2", "Sentinel-2".
    Raises KeyError if no matching sensor is found.
    """
    # fast path: exact key
    if name in STANDARD_SENSORS:
        return STANDARD_SENSORS[name]

    def _normalize(s: str) -> str:
        return s.strip().lower().replace("_", "-").replace(" ", "-")

    # build normalized map once (module-level cache via function attribute)
    if not hasattr(get_sensor, "_norm_map"):
        norm_map = { _normalize(k): k for k in STANDARD_SENSORS.keys() }
        # also allow keys without the dash (e.g. 'sentinel2')
        for k in list(STANDARD_SENSORS.keys()):
            nk = _normalize(k).replace("-", "")
            if nk not in norm_map:
                norm_map[nk] = k
        get_sensor._norm_map = norm_map

    key = _normalize(name)
    norm_map = getattr(get_sensor, "_norm_map")
    if key in norm_map:
        return STANDARD_SENSORS[norm_map[key]]
    # fallback: try removing dashes/underscores
    key_nodash = key.replace("-", "")
    if key_nodash in norm_map:
        return STANDARD_SENSORS[norm_map[key_nodash]]

    raise KeyError(f"Unknown sensor: {name}")


# ---------------------------------------------------------------------------
# Backwards-compatible lightweight band-mapping utilities (original repo)
# ---------------------------------------------------------------------------

# Logical band names used across the pipeline.
BLUE = "blue"
GREEN = "green"
RED = "red"
NIR = "nir"
SWIR1 = "swir1"
SWIR2 = "swir2"
ALPHA = "alpha"


@dataclass
class BandMapping:
    sensor: str
    bands: Dict[str, int]          # logical name -> raster band index (0-based)
    band_count: int
    verified: bool                 # True only if actually tested end-to-end
    notes: str = ""

    def has(self, *names: str) -> bool:
        return all(n in self.bands for n in names)

    def available_bands(self) -> List[str]:
        return sorted(self.bands)


SENSOR_REGISTRY: Dict[str, BandMapping] = {
    # --- Verified: exercised by the test-suite and sample scenes. ---
    "generic_rgb": BandMapping(
        sensor="generic_rgb", bands={RED: 0, GREEN: 1, BLUE: 2},
        band_count=3, verified=True,
        notes="Plain 3-band true-colour imagery. RGB proxies are used for "
              "vegetation/water cues; NIR indices are reported unavailable."),
    "generic_rgb_nir": BandMapping(
        sensor="generic_rgb_nir", bands={RED: 0, GREEN: 1, BLUE: 2, NIR: 3},
        band_count=4, verified=True,
        notes="4-band product with NIR as the 4th band (e.g. stacked GeoTIFF). "
              "True NDVI/NDWI family indices are computed."),
    "sar_generic": BandMapping(
        sensor="sar_generic", bands={"amplitude": 0},
        band_count=1, verified=True,
        notes="Single-band SAR amplitude/intensity (e.g. RISAT, Sentinel-1 GRD). "
              "Routed through core.sar: log/dB backscatter stretch + median "
              "speckle filter + water/built-up rule head — not the optical "
              "k-means land-cover pipeline."),
    # --- Planned: band tables reserved for future ingestion work. ---
    "sentinel2": BandMapping(
        sensor="sentinel2",
        bands={BLUE: 1, GREEN: 2, RED: 3, NIR: 7, SWIR1: 10, SWIR2: 11},
        band_count=12, verified=False,
        notes="Planned: Sentinel-2 L1C/L2A band layout (0-based indices into "
              "B01..B12). Not yet wired to a reader."),
    "landsat89": BandMapping(
        sensor="landsat89",
        bands={BLUE: 1, GREEN: 2, RED: 3, NIR: 4, SWIR1: 5, SWIR2: 6},
        band_count=7, verified=False,
        notes="Planned: Landsat 8/9 OLI band layout."),
    "planet": BandMapping(
        sensor="planet", bands={BLUE: 0, GREEN: 1, RED: 2, NIR: 3},
        band_count=4, verified=False,
        notes="Planned: PlanetScope 4-band layout."),
    "cartosat": BandMapping(
        sensor="cartosat", bands={RED: 0, GREEN: 0, BLUE: 0},
        band_count=1, verified=False,
        notes="Planned: Cartosat panchromatic — single band; multispectral "
              "indices do not apply."),
    "resourcesat": BandMapping(
        sensor="resourcesat", bands={GREEN: 0, RED: 1, NIR: 2},
        band_count=3, verified=False,
        notes="Planned: Resourcesat LISS-III/IV green/red/NIR layout."),
}


def resolve_mapping(band_count: int, mode: str = "",
                    hint: Optional[str] = None) -> BandMapping:
    """Pick the honest band mapping for an ingested raster.

    * ``hint`` may name a registry sensor, but unverified sensors are never
      silently used — the generic mapping is returned with a note instead.
    * RGBA input maps to ``generic_rgb`` (alpha is ignored, never mistaken
      for NIR).
    """
    if hint and hint in SENSOR_REGISTRY:
        m = SENSOR_REGISTRY[hint]
        if m.verified:
            return m
        # Fall through to the generic mapping; the caller surfaces the note.
    if band_count >= 4 and mode not in ("RGBA", "RGBa"):
        return SENSOR_REGISTRY["generic_rgb_nir"]
    return SENSOR_REGISTRY["generic_rgb"]


def supported_sensors() -> List[Dict[str, object]]:
    return [{"sensor": m.sensor, "verified": m.verified,
             "bands": m.available_bands(), "notes": m.notes}
            for m in SENSOR_REGISTRY.values()]
