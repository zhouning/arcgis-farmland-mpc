import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


def _load_runner():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "scirep_reward_weight_sensitivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scirep_reward_weight_sensitivity", script
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plan_runs_in_fresh_python_process(tmp_path):
    runner = _load_runner()
    plan_dir = tmp_path / "plan"
    summary_path = plan_dir / "mpc_summary.json"
    calls = []

    plan_config = {
        "ensemble_dir": str(tmp_path / "tool3"),
        "out_dir": str(plan_dir),
        "prepared_dir": str(tmp_path / "prepared"),
        "horizon": 5,
        "top_k": 50,
        "gamma": 0.99,
        "n_episodes": 1,
        "continuation": "greedy",
        "scoring": "reward",
        "threads": 0,
        "seed_offset": 0,
        "env_kind": "county",
        "cultivated_area_floor_delta_ha": 0.0,
        "max_steps": None,
    }

    def fake_run(cmd, cwd, check):
        calls.append((cmd, cwd, check))
        assert cmd[0] == sys.executable
        assert cmd[1] == "-c"
        assert "from farmland_mpc.mpc_plan import run" in cmd[2]
        assert "cultivated_area_floor_delta_ha" in cmd[2]
        assert check is False
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"results": [{"slope_change_pct": -0.1}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    summary = runner._run_plan_in_subprocess(
        plan_config, summary_path, run_cmd=fake_run
    )

    assert summary["results"][0]["slope_change_pct"] == -0.1
    assert len(calls) == 1


def test_write_outputs_uses_region_label(tmp_path):
    runner = _load_runner()
    args = Namespace(
        region_label="Neijiang",
        prepared_template=tmp_path / "prepared_neijiang",
        out_json=tmp_path / "reward_weight_sensitivity.json",
        out_md=tmp_path / "reward_weight_sensitivity.md",
        horizon=5,
        top_k=50,
        gamma=0.99,
        mpc_batch_size=256,
        max_steps=None,
        n_transition_episodes=60,
        n_pairwise_states=1000,
        n_pairwise_actions=50,
        seed=0,
        n_members=3,
        epochs=30,
        patience=8,
        lambda_rank=5.0,
        margin=0.1,
        batch_size=256,
        train_seed_base=0,
    )
    rows = [
        {
            "label": "Default reward",
            "slope_change_pct": -0.1,
            "cont_change": 0.02,
            "baimu_count_change": 1,
            "baimu_area_change_ha": 2.5,
            "cultivated_area_change_ha": 0.3,
            "final_ranking_acc_mean": 0.9,
            "sample_reward_std_median": 0.4,
        }
    ]

    runner._write_outputs(args, rows)

    report = json.loads(args.out_json.read_text(encoding="utf-8"))
    md = args.out_md.read_text(encoding="utf-8")
    assert report["region"] == "neijiang"
    assert md.startswith("# Retrained Neijiang reward-weight sensitivity")
