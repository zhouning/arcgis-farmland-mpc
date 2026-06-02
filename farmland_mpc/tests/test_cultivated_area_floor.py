"""Unit tests for the optional cumulative cultivated-area floor constraint.

Run with: python -m farmland_mpc.tests.test_cultivated_area_floor
"""

from __future__ import annotations

import numpy as np

from farmland_mpc.county_env import CountyLevelEnv, FARMLAND, FOREST


def _minimal_env(area_floor_m2):
    env = CountyLevelEnv.__new__(CountyLevelEnv)
    env.cultivated_area_floor_m2 = area_floor_m2
    env.baimu_area_floor_m2 = None
    env.delta_conn = 0.0
    env.gamma_conn = 0.0
    env.n_parcels = 4
    env.baimu_threshold_m2 = 100.0
    env.land_use = np.array([FARMLAND, FARMLAND, FOREST, FOREST], dtype=np.int8)
    env.initial_types = env.land_use.copy()
    env.slopes = np.array([10.0, 9.0, 1.0, 2.0])
    env.areas = np.array([100.0, 80.0, 60.0, 90.0])
    env.total_farm_area = 180.0
    env.total_weighted_slope = 1720.0
    env.n_farmland = 2
    env.n_forest = 2
    env.adjacency = [[], [], [], []]
    env.farmland_nbr_count = np.zeros(4, dtype=np.int32)
    env.total_farmland_adj = 0
    env.block_parcels = [np.arange(4, dtype=np.int32)]
    env.swapped = np.zeros(4, dtype=bool)
    env._block_farm_avail = np.array([2], dtype=np.int32)
    env._block_forest_avail = np.array([2], dtype=np.int32)
    return env


def test_constraint_selects_feasible_pair_instead_of_default_best_pair():
    env = _minimal_env(area_floor_m2=180.0)
    completed = env._execute_greedy_in_block(0, max_swaps=1)

    assert completed == 1
    assert env.total_farm_area == 190.0
    assert env.land_use.tolist() == [FARMLAND, FOREST, FOREST, FARMLAND]


def test_constraint_masks_block_without_feasible_pair():
    env = _minimal_env(area_floor_m2=181.0)
    env.areas[3] = 79.0

    assert env.action_masks().tolist() == [False]
    completed = env._execute_greedy_in_block(0, max_swaps=1)
    assert completed == 0
    assert env.total_farm_area == 180.0


def test_disabled_constraint_preserves_original_greedy_selection():
    env = _minimal_env(area_floor_m2=None)
    completed = env._execute_greedy_in_block(0, max_swaps=1)

    assert completed == 1
    assert env.total_farm_area == 140.0
    assert env.land_use.tolist() == [FOREST, FARMLAND, FARMLAND, FOREST]


def test_baimu_area_floor_rejects_pair_that_breaks_qualifying_area():
    env = _minimal_env(area_floor_m2=None)
    env.baimu_area_floor_m2 = 180.0
    env.adjacency = [
        np.array([1], dtype=np.intp),
        np.array([0], dtype=np.intp),
        np.array([], dtype=np.intp),
        np.array([], dtype=np.intp),
    ]

    completed = env._execute_greedy_in_block(0, max_swaps=1)

    assert completed == 0
    assert env.land_use.tolist() == [FARMLAND, FARMLAND, FOREST, FOREST]


if __name__ == "__main__":
    test_constraint_selects_feasible_pair_instead_of_default_best_pair()
    test_constraint_masks_block_without_feasible_pair()
    test_disabled_constraint_preserves_original_greedy_selection()
    test_baimu_area_floor_rejects_pair_that_breaks_qualifying_area()
    print("ALL CULTIVATED-AREA FLOOR TESTS PASSED")
