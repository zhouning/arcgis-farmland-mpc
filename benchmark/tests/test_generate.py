import json
from pathlib import Path
import geopandas as gpd

from generator.generate import generate_dataset
from generator.schema import load_preset
import pytest


@pytest.fixture
def small_preset(tmp_path):
    yaml_text = """
preset_id: smoke
n_blocks_target: 60
terrain:
  type: plain
  dem_amplitude_m: 5.0
  dem_lengthscale_m: 600.0
parcels:
  parcels_per_block_mean: 6
  parcels_per_block_std: 2
  area_distribution: lognormal
  area_mean_m2: 1200.0
landuse:
  farmland_frac: 0.7
  forest_frac: 0.2
  other_frac: 0.1
fragmentation:
  grf_lengthscale: 400.0
  patch_threshold: 0.4
"""
    p = tmp_path / "smoke.yaml"
    p.write_text(yaml_text)
    return load_preset(p)


def test_end_to_end_writes_expected_files(small_preset, tmp_out_dir):
    generate_dataset(small_preset, seed=0, out_dir=tmp_out_dir)
    assert (tmp_out_dir / "DLTB_with_slope.gpkg").exists()
    assert (tmp_out_dir / "manifest.json").exists()
    township_dirs = list(tmp_out_dir.glob("township_*"))
    assert len(township_dirs) >= 1
    for td in township_dirs:
        assert (td / "block_compositions.json").exists()
        assert (td / "block_features.json").exists()


def test_manifest_contains_metadata(small_preset, tmp_out_dir):
    generate_dataset(small_preset, seed=42, out_dir=tmp_out_dir)
    manifest = json.loads((tmp_out_dir / "manifest.json").read_text())
    assert manifest["preset_id"] == "smoke"
    assert manifest["seed"] == 42
    assert "n_blocks" in manifest
    assert "n_parcels" in manifest
    assert manifest["n_blocks"] > 0
    assert manifest["n_parcels"] > manifest["n_blocks"]


def test_geopackage_loadable_by_geopandas(small_preset, tmp_out_dir):
    generate_dataset(small_preset, seed=0, out_dir=tmp_out_dir)
    gdf = gpd.read_file(tmp_out_dir / "DLTB_with_slope.gpkg")
    assert {"DLBM", "QSDWDM", "slope_mean"}.issubset(gdf.columns)
    assert len(gdf) > 0
