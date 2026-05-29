"""Neijiang Dongxing variant of CountyLevelEnv for Paper 9 cross-region test.

Strategy: import the research-side county_env.py, then swap the module-level
constants (DLTB_PATH, BLOCK_DIR, ALL_TOWNSHIPS, TOWNSHIP_CODES) before
instantiating CountyLevelEnv. This keeps the original Bishan env untouched.

NOTE (public release): this script is provided as a configuration reference,
not as a runnable artifact. It depends on (a) the research-side `county_env.py`
module (not part of this package) and (b) the raw Third National Land Survey
GeoPackage for Neijiang Dongxing, which cannot be redistributed under existing
data-governance restrictions (see the paper's Data Availability statement).
Set the paths below to your own checkout/data before use.

Usage:
    from neijiang_cross_region.county_env_neijiang import make_neijiang_env
    env = make_neijiang_env(total_budget=500, swaps_per_step=5, ...)
"""
import os, sys

# Path to the research-side checkout that contains county_env.py
SCRIPT_DIR = os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Raw 三调 GeoPackage with per-parcel slope — RESTRICTED, supply your own
NEIJIANG_GPKG = os.environ.get("NEIJIANG_GPKG", "/path/to/neijiang_DLTB_with_slope.gpkg")
NEIJIANG_BLOCK_DIR = os.environ.get(
    "NEIJIANG_BLOCK_DIR",
    os.path.join(os.path.dirname(__file__), "blocks"),
)

NEIJIANG_TOWNSHIPS = {
    '511011001': 'N01',
    '511011002': 'N02',
    '511011003': 'N03',
    '511011100': 'N04',
    '511011101': 'N05',
    '511011102': 'N06',
    '511011103': 'N07',
    '511011104': 'N08',
    '511011105': 'N09',
    '511011106': 'N10',
    '511011107': 'N11',
    '511011108': 'N12',
    '511011109': 'N13',
    '511011110': 'N14',
    '511011111': 'N15',
    '511011200': 'N16',
    '511011201': 'N17',
    '511011202': 'N18',
    '511011203': 'N19',
    '511011204': 'N20',
    '511011205': 'N21',
    '511011206': 'N22',
    '511011207': 'N23',
    '511011208': 'N24',
    '511011209': 'N25',
    '511011210': 'N26',
    '511011211': 'N27',
    '511011212': 'N28',
    '511011213': 'N29',
}


def make_neijiang_env(**kwargs):
    """Instantiate a CountyLevelEnv configured for Neijiang Dongxing."""
    import county_env
    # Patch module-level constants
    county_env.DLTB_PATH = NEIJIANG_GPKG
    county_env.BLOCK_DIR = NEIJIANG_BLOCK_DIR
    county_env.ALL_TOWNSHIPS = dict(NEIJIANG_TOWNSHIPS)
    county_env.TOWNSHIP_CODES = sorted(NEIJIANG_TOWNSHIPS.keys())

    from county_env import CountyLevelEnv
    return CountyLevelEnv(**kwargs)


if __name__ == '__main__':
    print("Creating Neijiang CountyLevelEnv (smoke test)...")
    import time
    t0 = time.time()
    env = make_neijiang_env(total_budget=500, swaps_per_step=5)
    print(f"\n>>> Env init complete in {time.time()-t0:.1f}s <<<")
    print(f"  n_blocks: {env.n_blocks}")
    print(f"  obs dim: {env.observation_space.shape}")
    print(f"  action dim: {env.action_space.n}")

    t1 = time.time()
    obs, info = env.reset(seed=0)
    print(f"  reset: {time.time()-t1:.2f}s, initial slope={env.avg_farmland_slope:.4f}")
    print(f"  initial contiguity={env.contiguity:.4f}")
    print(f"  initial baimu count={env.baimu_count}, area={env.baimu_total_area/10000:.0f} ha")

    # Do 3 random steps
    import numpy as np
    rng = np.random.default_rng(42)
    for step in range(3):
        mask = env.action_masks()
        valid = np.where(mask)[0]
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {step+1}: action={action}, reward={reward:.3f}, slope={env.avg_farmland_slope:.4f}")

    print("\n>>> Smoke test PASS <<<")
