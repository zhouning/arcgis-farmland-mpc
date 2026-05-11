import pytest
from generator.schema import PresetConfig, load_preset


def test_load_preset_roundtrip(tmp_path):
    yaml_text = """
preset_id: test_preset
n_blocks_target: 800
terrain:
  type: plain
  dem_amplitude_m: 5.0
  dem_lengthscale_m: 800.0
parcels:
  parcels_per_block_mean: 28
  parcels_per_block_std: 8
  area_distribution: lognormal
  area_mean_m2: 1200.0
landuse:
  farmland_frac: 0.78
  forest_frac: 0.15
  other_frac: 0.07
fragmentation:
  grf_lengthscale: 250.0
  patch_threshold: 0.4
adjacency:
  median_degree_target: 4
"""
    p = tmp_path / "test.yaml"
    p.write_text(yaml_text)
    cfg = load_preset(p)
    assert isinstance(cfg, PresetConfig)
    assert cfg.preset_id == "test_preset"
    assert cfg.n_blocks_target == 800
    assert cfg.terrain.type == "plain"
    assert cfg.landuse.farmland_frac == pytest.approx(0.78)


def test_landuse_fractions_sum_to_one():
    from generator.schema import LanduseConfig
    with pytest.raises(ValueError, match="must sum to 1"):
        LanduseConfig(farmland_frac=0.5, forest_frac=0.3, other_frac=0.3)


def test_terrain_type_validated():
    from generator.schema import TerrainConfig
    with pytest.raises(ValueError, match="terrain.type"):
        TerrainConfig(type="mountain", dem_amplitude_m=50, dem_lengthscale_m=500)
