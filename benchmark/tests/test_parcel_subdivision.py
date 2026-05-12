from shapely.geometry import box
from generator.tessellation import subdivide_block_into_parcels


def test_parcel_count_matches_request(rng_seed):
    block = box(0, 0, 1000, 1000)
    parcels = subdivide_block_into_parcels(block, n_parcels=20, seed=rng_seed)
    assert 18 <= len(parcels) <= 22


def test_parcels_tile_block(rng_seed):
    block = box(0, 0, 1000, 1000)
    parcels = subdivide_block_into_parcels(block, n_parcels=15, seed=rng_seed)
    total = sum(p.area for p in parcels)
    assert total > 0.99 * block.area


def test_parcels_are_valid(rng_seed):
    block = box(0, 0, 500, 800)
    parcels = subdivide_block_into_parcels(block, n_parcels=10, seed=rng_seed)
    for p in parcels:
        assert p.is_valid
        assert p.area > 0
