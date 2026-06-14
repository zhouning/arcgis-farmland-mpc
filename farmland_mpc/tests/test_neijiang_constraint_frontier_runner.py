from pathlib import Path
import sys

import pytest


def test_neijiang_frontier_defaults_point_to_neijiang_assets():
    from scripts import pareto_sweep_neijiang_constraints as runner

    parser = runner.build_arg_parser()
    args = parser.parse_args([])

    assert args.prepared_dir == Path("runs/scirep_extra/prepared_neijiang")
    assert args.ensemble_dir == Path("runs/scirep_extra/onnx/neijiang/ensemble_seed0")
    assert runner.slope_source(args) == (
        Path("runs/scirep_extra/prepared_neijiang")
        / "dem_slope_analysis"
        / "output"
        / "DLTB_with_slope.gpkg"
    )
    assert args.out_json == Path("paper/submission_scirep_corrected/neijiang_constraint_frontier.json")
    assert args.out_md == Path("paper/submission_scirep_corrected/neijiang_constraint_frontier.md")


def test_neijiang_frontier_markdown_uses_region_title(tmp_path):
    from scripts import pareto_sweep_neijiang_constraints as runner

    out = tmp_path / "frontier.md"
    rows = [
        {
            "id": "no_net_loss",
            "label": "No net cultivated loss",
            "slope_change_pct": -0.492,
            "cont_change": 0.0373,
            "steep_tail_change_ha": -330.4,
            "baimu_count_change": 0,
            "baimu_area_change_ha": 252.6,
            "cultivated_area_change_ha": 62.2,
            "total_reward": 185.0,
            "swaps": 466,
            "runtime_s": 2700.0,
        }
    ]

    runner.write_markdown(out, rows)

    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Neijiang Dongxing execution-constraint frontier")
    assert "| No net cultivated loss | -0.492 | +0.0373 | -330.4 | +0 | +252.6 | +62.2 | +185.00 | 466 | 2700.0 |" in text


def test_neijiang_frontier_can_add_repo_root_to_import_path(monkeypatch):
    from scripts import pareto_sweep_neijiang_constraints as runner

    repo_root = str(Path(runner.__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != repo_root])

    runner.ensure_repo_root_on_path()

    assert sys.path[0] == repo_root


def test_neijiang_frontier_can_filter_configs():
    from scripts import pareto_sweep_neijiang_constraints as runner

    parser = runner.build_arg_parser()
    args = parser.parse_args(["--only", "unconstrained", "no_net_loss"])

    ids = [cfg["id"] for cfg in runner.selected_configs(args)]
    assert ids == ["unconstrained", "no_net_loss"]


def test_neijiang_frontier_supports_skipping_expensive_audits():
    from scripts import pareto_sweep_neijiang_constraints as runner

    parser = runner.build_arg_parser()
    args = parser.parse_args(["--skip-validation", "--skip-policy-audit"])

    assert args.skip_validation is True
    assert args.skip_policy_audit is True


def test_neijiang_frontier_detects_existing_plan_outputs(tmp_path):
    from scripts import pareto_sweep_neijiang_constraints as runner

    cell_dir = tmp_path / "no_net_loss"
    cell_dir.mkdir()
    (cell_dir / "mpc_summary.json").write_text("{}", encoding="utf-8")
    (cell_dir / "optimized.shp").write_text("placeholder", encoding="utf-8")

    assert runner.has_plan_outputs(cell_dir) is True


def test_neijiang_frontier_reuses_plan_outputs_and_backfills_audits(tmp_path, monkeypatch):
    from scripts import pareto_sweep_neijiang_constraints as runner

    scratch = tmp_path / "scratch"
    cell_dir = scratch / "no_net_loss"
    cell_dir.mkdir(parents=True)
    (cell_dir / "mpc_summary.json").write_text(
        '{"results":[{"slope_change_pct":-0.1,"cont_change":0.02,'
        '"cultivated_area_change_ha":12.0,"cultivated_area_change_pct":0.3,'
        '"baimu_count_change":1,"baimu_area_change_ha":20.0,'
        '"total_reward":5.0,"swaps_completed":4,"steps_run":5,'
        '"total_time_s":123.0}]}',
        encoding="utf-8",
    )
    (cell_dir / "optimized.shp").write_text("placeholder", encoding="utf-8")

    calls = {"plan": 0, "validation": 0, "policy": 0}

    def fail_plan(*args, **kwargs):
        calls["plan"] += 1
        raise AssertionError("run_plan should not be called when plan outputs exist")

    def fake_validation(args, actual_cell_dir):
        calls["validation"] += 1
        assert actual_cell_dir == cell_dir
        return {
            "overall_pass": True,
            "delta_recomputed": {
                "slope_pct": -0.1,
                "cont": 0.02,
                "cultivated_area_ha": 12.0,
                "cultivated_area_pct": 0.3,
                "baimu_ha": 20.0,
            },
        }

    def fake_policy(args, cfg, actual_cell_dir):
        calls["policy"] += 1
        assert cfg["id"] == "no_net_loss"
        assert actual_cell_dir == cell_dir
        return {
            "source_optimized_shp": str(cell_dir / "optimized.shp"),
            "slope_bands": {"delta_ha": {"15_25": -2.0, "gt25": -3.0}},
            "cultivated_area": {"delta_ha": 12.0, "delta_pct": 0.3},
            "baimu_fang": {"delta_count": 1, "delta_area_ha": 20.0},
            "swap_area_totals": {"farm_to_forest_count": 4},
        }

    monkeypatch.setattr(runner, "run_plan", fail_plan)
    monkeypatch.setattr(runner, "run_validation", fake_validation)
    monkeypatch.setattr(runner, "run_policy_audit", fake_policy)
    monkeypatch.setattr(runner, "write_plot", lambda *args, **kwargs: None)

    rc = runner.main(
        [
            "--only",
            "no_net_loss",
            "--scratch-dir",
            str(scratch),
            "--out-json",
            str(tmp_path / "frontier.json"),
            "--out-md",
            str(tmp_path / "frontier.md"),
            "--out-fig-pdf",
            str(tmp_path / "frontier.pdf"),
            "--out-fig-png",
            str(tmp_path / "frontier.png"),
        ]
    )

    assert rc == 0
    assert calls == {"plan": 0, "validation": 1, "policy": 1}
