import json
import sys
from pathlib import Path
import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, "D:/test")


@pytest.fixture
def toy_dataset(tmp_path):
    from generator.schema import load_preset
    from generator.generate import generate_dataset
    cfg = load_preset(BENCH_ROOT / "presets" / "plain_small_cons.yaml")
    cfg.n_blocks_target = 60
    cfg.parcels.parcels_per_block_mean = 6
    cfg.parcels.parcels_per_block_std = 2
    out = tmp_path / "ds"
    generate_dataset(cfg, seed=0, out_dir=out)
    return out


def test_random_runner_writes_valid_result(toy_dataset, tmp_path):
    from baselines.run_random import run_random
    from eval.metrics import RESULT_SCHEMA_KEYS

    result_path = tmp_path / "random.json"
    result = run_random(
        dataset_dir=toy_dataset,
        preset_id="plain_small_cons",
        seed=0,
        out_path=result_path,
        total_budget=20, swaps_per_step=2,
    )
    assert set(result.keys()) == RESULT_SCHEMA_KEYS
    on_disk = json.loads(result_path.read_text())
    assert on_disk["method"] == "Random-Block"
    assert on_disk["seed"] == 0


def test_greedy_runner_writes_valid_result(toy_dataset, tmp_path):
    from baselines.run_greedy import run_greedy
    from eval.metrics import RESULT_SCHEMA_KEYS

    result = run_greedy(
        dataset_dir=toy_dataset,
        preset_id="plain_small_cons",
        seed=0,
        out_path=tmp_path / "greedy.json",
        total_budget=20, swaps_per_step=2,
    )
    assert set(result.keys()) == RESULT_SCHEMA_KEYS
    assert result["method"] == "Greedy-Sequential"


def test_ga_runner_writes_valid_result(toy_dataset, tmp_path):
    from baselines.run_ga import run_ga_baseline
    from eval.metrics import RESULT_SCHEMA_KEYS

    result = run_ga_baseline(
        dataset_dir=toy_dataset,
        preset_id="plain_small_cons",
        seed=0,
        out_path=tmp_path / "ga.json",
        total_budget=10, swaps_per_step=2,
        pop_size=10, generations=10,  # smoke values
    )
    assert set(result.keys()) == RESULT_SCHEMA_KEYS
    assert result["method"] == "GA"


def test_ppo_runner_writes_valid_result(toy_dataset, tmp_path):
    from baselines.run_ppo import run_ppo
    from eval.metrics import RESULT_SCHEMA_KEYS

    result = run_ppo(
        dataset_dir=toy_dataset,
        preset_id="plain_small_cons",
        seed=0,
        out_path=tmp_path / "ppo.json",
        total_budget=20, swaps_per_step=2,
        total_timesteps=500,  # smoke
        device="cpu",
    )
    assert set(result.keys()) == RESULT_SCHEMA_KEYS
    assert result["method"] == "PPO-Centralized"
