"""Preset schema for the synthetic farmland benchmark generator."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

TERRAIN_TYPES = ("hilly", "mixed", "plain")
AREA_DISTRIBUTIONS = ("lognormal", "uniform")


@dataclass
class TerrainConfig:
    type: str
    dem_amplitude_m: float
    dem_lengthscale_m: float

    def __post_init__(self):
        if self.type not in TERRAIN_TYPES:
            raise ValueError(f"terrain.type must be one of {TERRAIN_TYPES}, got {self.type!r}")
        if self.dem_amplitude_m < 0:
            raise ValueError(f"dem_amplitude_m must be >= 0, got {self.dem_amplitude_m!r}")
        if self.dem_lengthscale_m <= 0:
            raise ValueError(f"dem_lengthscale_m must be > 0, got {self.dem_lengthscale_m!r}")


@dataclass
class ParcelsConfig:
    parcels_per_block_mean: int
    parcels_per_block_std: int
    area_distribution: str
    area_mean_m2: float

    def __post_init__(self):
        if self.area_distribution not in AREA_DISTRIBUTIONS:
            raise ValueError(f"parcels.area_distribution must be one of {AREA_DISTRIBUTIONS}, got {self.area_distribution!r}")
        if self.parcels_per_block_mean <= 0:
            raise ValueError(f"parcels_per_block_mean must be > 0, got {self.parcels_per_block_mean!r}")


@dataclass
class LanduseConfig:
    farmland_frac: float
    forest_frac: float
    other_frac: float

    def __post_init__(self):
        total = self.farmland_frac + self.forest_frac + self.other_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"landuse fractions must sum to 1, got {total}")


@dataclass
class FragmentationConfig:
    grf_lengthscale: float
    patch_threshold: float


@dataclass
class AdjacencyConfig:
    median_degree_target: int = 4


@dataclass
class PresetConfig:
    preset_id: str
    n_blocks_target: int
    terrain: TerrainConfig
    parcels: ParcelsConfig
    landuse: LanduseConfig
    fragmentation: FragmentationConfig
    adjacency: AdjacencyConfig = field(default_factory=AdjacencyConfig)


def load_preset(path: str | Path) -> PresetConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PresetConfig(
        preset_id=raw["preset_id"],
        n_blocks_target=int(raw["n_blocks_target"]),
        terrain=TerrainConfig(**raw["terrain"]),
        parcels=ParcelsConfig(**raw["parcels"]),
        landuse=LanduseConfig(**raw["landuse"]),
        fragmentation=FragmentationConfig(**raw["fragmentation"]),
        adjacency=AdjacencyConfig(**raw.get("adjacency", {})),
    )
