"""End-to-end orchestrator: PresetConfig + seed -> synthetic dataset on disk."""
from __future__ import annotations
import json
import sys
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

ANCHOR_TARGETS = {
    "bishan_clone": {
        "init_slope_deg": 9.6157,
        "init_contiguity": 3.5852,
        "init_baimu_count": 109.0,
        "init_baimu_area_ha": 46843.65,
    },
    "neijiang_clone": {
        "init_slope_deg": 10.55,
        "init_contiguity": 2.63,
        "init_baimu_count": 384.0,
        "init_baimu_area_ha": 74342.0,
    },
}


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


def _maybe_run_calibration(cfg: PresetConfig, out_dir: Path) -> None:
    """For anchor presets, build env, snapshot init stats, write calibration_report.json."""
    if cfg.preset_id not in ANCHOR_TARGETS:
        return
    # Late imports: calibration depends on county_env (heavy)
    from .calibration import (
        AnchorTargets,
        compute_init_stats,
        verify_anchor_within_tolerance,
    )
    # synthetic_env_loader lives at benchmark root (sibling of generator/)
    benchmark_root = Path(out_dir).resolve().parents[0]
    # Walk upward until we find synthetic_env_loader.py; fall back to two levels up
    for candidate in [Path(out_dir).resolve(), *Path(out_dir).resolve().parents]:
        if (candidate / "synthetic_env_loader.py").exists():
            benchmark_root = candidate
            break
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    from synthetic_env_loader import make_synthetic_env

    env = make_synthetic_env(out_dir, total_budget=20, swaps_per_step=2)
    env.reset(seed=0)
    actual = compute_init_stats(env)
    target = AnchorTargets(name=cfg.preset_id, **ANCHOR_TARGETS[cfg.preset_id])
    report = verify_anchor_within_tolerance(target, actual, tolerance=0.5)
    with open(Path(out_dir) / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if not report["passed"]:
        print(f"WARNING: anchor calibration FAILED for {cfg.preset_id}")
        print(json.dumps(report, indent=2))


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
    _maybe_run_calibration(cfg, out_dir)
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
