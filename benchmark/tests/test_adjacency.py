import numpy as np
from generator.adjacency import build_block_adjacency
from generator.tessellation import tessellate_domain


def test_adjacency_is_symmetric(rng_seed):
    blocks, _ = tessellate_domain(n_blocks_target=200, seed=rng_seed)
    adj = build_block_adjacency(blocks)
    assert len(adj) == len(blocks)
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            assert i in adj[j], f"asymmetry at ({i},{j})"


def test_median_degree_close_to_target(rng_seed):
    blocks, _ = tessellate_domain(n_blocks_target=500, seed=rng_seed)
    adj = build_block_adjacency(blocks)
    degrees = [len(a) for a in adj]
    median = float(np.median(degrees))
    assert 3.0 <= median <= 7.0


def test_no_self_loops(rng_seed):
    blocks, _ = tessellate_domain(n_blocks_target=100, seed=rng_seed)
    adj = build_block_adjacency(blocks)
    for i, nbrs in enumerate(adj):
        assert i not in nbrs
