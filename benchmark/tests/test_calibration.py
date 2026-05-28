import sys
from pathlib import Path
import pytest

from generator.calibration import (
    AnchorTargets,
    compute_init_stats,
    verify_anchor_within_tolerance,
)

BISHAN_TARGETS = AnchorTargets(
    name="bishan",
    init_slope_deg=9.80,
    init_contiguity=3.59,
    init_baimu_count=None,
    init_baimu_area_ha=None,
)

NEIJIANG_TARGETS = AnchorTargets(
    name="neijiang",
    init_slope_deg=10.55,
    init_contiguity=2.63,
    init_baimu_count=384,
    init_baimu_area_ha=74342.0,
)


def test_within_tolerance_passes_when_close():
    actual = {
        "init_slope_deg": 10.0,
        "init_contiguity": 2.7,
        "init_baimu_count": 380,
        "init_baimu_area_ha": 74000.0,
    }
    report = verify_anchor_within_tolerance(NEIJIANG_TARGETS, actual, tolerance=0.5)
    assert report["passed"] is True
    for k, d in report["deltas"].items():
        if d["target"] is not None:
            assert d["within_tolerance"] is True


def test_within_tolerance_fails_when_off():
    actual = {
        "init_slope_deg": 30.0,
        "init_contiguity": 2.7,
        "init_baimu_count": 380,
        "init_baimu_area_ha": 74000.0,
    }
    report = verify_anchor_within_tolerance(NEIJIANG_TARGETS, actual, tolerance=0.5)
    assert report["passed"] is False
    assert report["deltas"]["init_slope_deg"]["within_tolerance"] is False


def test_compute_init_stats_keys_present(tmp_out_dir):
    """compute_init_stats requires an env; synthetic_env_loader builds one."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generator.schema import load_preset
    from generator.generate import generate_dataset
    preset_path = Path(__file__).resolve().parents[1] / "presets" / "plain_small_cons.yaml"
    cfg = load_preset(preset_path)
    cfg.n_blocks_target = 60
    cfg.parcels.parcels_per_block_mean = 6
    cfg.parcels.parcels_per_block_std = 2
    generate_dataset(cfg, seed=0, out_dir=tmp_out_dir)
    from synthetic_env_loader import make_synthetic_env
    env = make_synthetic_env(tmp_out_dir, total_budget=20, swaps_per_step=2)
    env.reset(seed=0)
    stats = compute_init_stats(env)
    for k in ("init_slope_deg", "init_contiguity", "init_baimu_count", "init_baimu_area_ha"):
        assert k in stats


def test_none_targets_always_pass():
    """When a target field is None, it must be treated as passing."""
    partial_targets = AnchorTargets(
        name="partial",
        init_slope_deg=10.0,
        init_contiguity=3.0,
        init_baimu_count=None,
        init_baimu_area_ha=None,
    )
    actual = {
        "init_slope_deg": 10.5,
        "init_contiguity": 3.1,
        "init_baimu_count": 0,
        "init_baimu_area_ha": 0.0,
    }
    report = verify_anchor_within_tolerance(partial_targets, actual, tolerance=0.5)
    assert report["passed"] is True
    assert report["deltas"]["init_baimu_count"]["within_tolerance"] is True
    assert report["deltas"]["init_baimu_count"]["target"] is None
