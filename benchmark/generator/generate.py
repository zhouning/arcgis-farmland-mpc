"""End-to-end orchestrator: PresetConfig + seed -> synthetic dataset on disk."""
from __future__ import annotations
import json
import time
from pathlib import Path
import argparse

import numpy as np

from .schema import PresetConfig, load_preset
from .tessellation import tessellate_domain, subdivide_block_into_parcels
from .dem import synthesize_dem, derive_slope_degrees, sample_block_slopes
from .landuse import assign_landuse_per_block, FARMLAND_CODE, FOREST_CODE, OTHER_CODE
from .features import polsby_popper_compactness
from .io import write_synthetic_dataset


COUNTY_CODE = "999999"


def _assign_blocks_to_townships(
    n_blocks: int, target_blocks_per_township: int = 130
) -> tuple[list[int], list[str]]:
    """Partition blocks into townships of ~target_blocks_per_township each."""
    n_townships = max(1, int(round(n_blocks / target_blocks_per_township)))
    boundaries = np.linspace(0, n_blocks, n_townships + 1, dtype=int)
    block_township_id = [0] * n_blocks
    for ti in range(n_townships):
        for bi in range(boundaries[ti], boundaries[ti + 1]):
            block_township_id[bi] = ti
    township_codes = [f"{COUNTY_CODE}{ti + 1:03d}" for ti in range(n_townships)]
    return block_township_id, township_codes


def generate_dataset(
    cfg: PresetConfig,
    seed: int,
    out_dir: str | Path,
) -> dict:
    """Generate one synthetic dataset and write to out_dir. Returns manifest dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)

    blocks, domain = tessellate_domain(cfg.n_blocks_target, seed=seed)
    n_blocks = len(blocks)
    minx, miny, maxx, maxy = domain.bounds

    dem, transform = synthesize_dem(
        bounds=(minx, miny, maxx, maxy),
        resolution_m=50.0,
        amplitude_m=cfg.terrain.dem_amplitude_m,
        lengthscale_m=cfg.terrain.dem_lengthscale_m,
        seed=seed,
    )
    slope_raster = derive_slope_degrees(dem, pixel_size_m=50.0)
    block_slopes = sample_block_slopes(blocks, slope_raster, transform)

    block_codes = assign_landuse_per_block(
        blocks,
        cfg.landuse.farmland_frac,
        cfg.landuse.forest_frac,
        cfg.landuse.other_frac,
        cfg.fragmentation.grf_lengthscale,
        cfg.fragmentation.patch_threshold,
        seed=seed,
    )

    block_compactness = [polsby_popper_compactness(b) for b in blocks]

    parcels_per_block: list[list] = []
    parcel_dlbm: list[list[str]] = []
    parcel_slope: list[list[float]] = []
    for bi, b in enumerate(blocks):
        n_par = max(3, int(rng.normal(cfg.parcels.parcels_per_block_mean,
                                       cfg.parcels.parcels_per_block_std)))
        parcels = subdivide_block_into_parcels(b, n_par, seed=seed + 1000 * bi)
        parcels_per_block.append(parcels)
        dlbm_block: list[str] = []
        slope_block: list[float] = []
        for _ in parcels:
            r = rng.random()
            if r < 0.85:
                dlbm_block.append(block_codes[bi])
            elif r < 0.92:
                dlbm_block.append(FOREST_CODE if block_codes[bi] != FOREST_CODE else FARMLAND_CODE)
            else:
                dlbm_block.append(OTHER_CODE)
            slope_block.append(float(block_slopes[bi] + rng.normal(0, 1.0)))
        parcel_dlbm.append(dlbm_block)
        parcel_slope.append([max(0.0, s) for s in slope_block])

    block_township_id, township_codes = _assign_blocks_to_townships(n_blocks)

    dataset = {
        "blocks": blocks,
        "parcels_per_block": parcels_per_block,
        "parcel_dlbm": parcel_dlbm,
        "parcel_slope": parcel_slope,
        "block_township_id": block_township_id,
        "block_compactness": block_compactness,
        "township_codes": township_codes,
    }
    write_synthetic_dataset(dataset, out_dir)

    n_parcels = sum(len(ps) for ps in parcels_per_block)
    elapsed = time.time() - t0
    manifest = {
        "preset_id": cfg.preset_id,
        "seed": seed,
        "n_blocks": n_blocks,
        "n_parcels": n_parcels,
        "n_townships": len(township_codes),
        "domain_bounds": [minx, miny, maxx, maxy],
        "elapsed_seconds": elapsed,
        "generator_version": "0.1.0",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True, type=Path, help="path to preset YAML")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args()
    cfg = load_preset(args.preset)
    manifest = generate_dataset(cfg, args.seed, args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
