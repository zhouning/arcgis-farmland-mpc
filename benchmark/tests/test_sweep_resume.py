import json
import sys
from pathlib import Path
import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from sweep.manifest import build_manifest, load_state
from sweep.runner import run_sweep, METHOD_DISPATCH


@pytest.fixture
def tiny_setup(tmp_path):
    """Build manifest with 1 preset x 2 methods x 2 seeds = 4 cells, all
    Random/Greedy (cheapest baselines so the test stays fast)."""
    from generator.schema import load_preset
    from generator.generate import generate_dataset
    cfg = load_preset(BENCH_ROOT / "presets" / "plain_small_cons.yaml")
    cfg.n_blocks_target = 60
    cfg.parcels.parcels_per_block_mean = 6
    cfg.parcels.parcels_per_block_std = 2
    data_root = tmp_path / "data"
    for seed in (0, 1):
        generate_dataset(cfg, seed=seed,
                         out_dir=data_root / f"plain_small_cons_seed{seed}")
    csv_path = tmp_path / "manifest.csv"
    state_path = tmp_path / "state.json"
    results_root = tmp_path / "results"
    build_manifest(["plain_small_cons"], ["Random", "Greedy"], [0, 1],
                   out_csv=csv_path)
    return {
        "csv": csv_path, "state": state_path, "data_root": data_root,
        "results_root": results_root,
    }


def test_full_sweep_marks_all_done(tiny_setup):
    run_sweep(
        manifest_csv=tiny_setup["csv"],
        state_path=tiny_setup["state"],
        data_root=tiny_setup["data_root"],
        results_root=tiny_setup["results_root"],
        method_kwargs={"Random": {"total_budget": 20, "swaps_per_step": 2},
                       "Greedy": {"total_budget": 20, "swaps_per_step": 2}},
    )
    state = load_state(tiny_setup["state"])
    assert all(c["status"] == "done" for c in state["cells"])
    n = sum(1 for _ in tiny_setup["results_root"].rglob("*.json"))
    assert n == 4


def test_resume_picks_up_after_partial(tiny_setup):
    # First run with limit_time=0.01s -- checks gate before each cell and
    # exits after the first one completes.
    run_sweep(
        manifest_csv=tiny_setup["csv"],
        state_path=tiny_setup["state"],
        data_root=tiny_setup["data_root"],
        results_root=tiny_setup["results_root"],
        method_kwargs={"Random": {"total_budget": 20, "swaps_per_step": 2},
                       "Greedy": {"total_budget": 20, "swaps_per_step": 2}},
        limit_time_s=0.01,
    )
    s1 = load_state(tiny_setup["state"])
    n_done_1 = sum(c["status"] == "done" for c in s1["cells"])
    assert n_done_1 < len(s1["cells"])  # didn't finish all

    # Resume with no time limit -- should finish everything.
    run_sweep(
        manifest_csv=tiny_setup["csv"],
        state_path=tiny_setup["state"],
        data_root=tiny_setup["data_root"],
        results_root=tiny_setup["results_root"],
        method_kwargs={"Random": {"total_budget": 20, "swaps_per_step": 2},
                       "Greedy": {"total_budget": 20, "swaps_per_step": 2}},
        resume=True,
    )
    s2 = load_state(tiny_setup["state"])
    assert all(c["status"] == "done" for c in s2["cells"])


def test_method_dispatch_contains_all_five():
    assert METHOD_DISPATCH == {"Random", "Greedy", "GA", "PPO", "MPC"}
