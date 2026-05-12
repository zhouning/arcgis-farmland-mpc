import pytest
from generator.landuse import assign_landuse_per_block
from generator.tessellation import tessellate_domain

FARMLAND_CODE = "0110"
FOREST_CODE = "0310"
OTHER_CODE = "0510"


def test_landuse_codes_close_to_target_fractions(rng_seed):
    blocks, _ = tessellate_domain(n_blocks_target=500, seed=rng_seed)
    codes = assign_landuse_per_block(
        blocks,
        farmland_frac=0.78,
        forest_frac=0.15,
        other_frac=0.07,
        grf_lengthscale_m=300.0,
        patch_threshold=0.4,
        seed=rng_seed,
    )
    assert len(codes) == len(blocks)
    n = len(codes)
    n_farm = sum(c == FARMLAND_CODE for c in codes)
    n_forest = sum(c == FOREST_CODE for c in codes)
    n_other = sum(c == OTHER_CODE for c in codes)
    assert abs(n_farm / n - 0.78) < 0.05
    assert abs(n_forest / n - 0.15) < 0.05
    assert abs(n_other / n - 0.07) < 0.05


def test_landuse_codes_are_valid_strings(rng_seed):
    blocks, _ = tessellate_domain(n_blocks_target=100, seed=rng_seed)
    codes = assign_landuse_per_block(
        blocks, 0.7, 0.2, 0.1,
        grf_lengthscale_m=400.0, patch_threshold=0.4, seed=rng_seed,
    )
    assert all(c in {FARMLAND_CODE, FOREST_CODE, OTHER_CODE} for c in codes)


def test_low_lengthscale_yields_more_patches(rng_seed):
    blocks, _ = tessellate_domain(n_blocks_target=400, seed=rng_seed)
    from generator.adjacency import build_block_adjacency

    def count_transitions(codes, blocks):
        adj = build_block_adjacency(blocks)
        t = 0
        for i, nbrs in enumerate(adj):
            for j in nbrs:
                if j > i and codes[i] != codes[j]:
                    t += 1
        return t

    codes_frag = assign_landuse_per_block(
        blocks, 0.7, 0.2, 0.1, grf_lengthscale_m=150.0, patch_threshold=0.4, seed=rng_seed,
    )
    codes_cons = assign_landuse_per_block(
        blocks, 0.7, 0.2, 0.1, grf_lengthscale_m=1000.0, patch_threshold=0.4, seed=rng_seed,
    )
    assert count_transitions(codes_frag, blocks) > count_transitions(codes_cons, blocks)
