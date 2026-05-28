import sys
from pathlib import Path

D_TEST = Path("D:/test")
if str(D_TEST) not in sys.path:
    sys.path.insert(0, str(D_TEST))


def _quick_synth(tmp_out_dir):
    """Generate a tiny dataset suitable for env smoke test."""
    from generator.schema import load_preset
    from generator.generate import generate_dataset
    preset_path = Path(__file__).resolve().parents[1] / "presets" / "plain_small_cons.yaml"
    cfg = load_preset(preset_path)
    cfg.n_blocks_target = 60
    cfg.parcels.parcels_per_block_mean = 6
    cfg.parcels.parcels_per_block_std = 2
    generate_dataset(cfg, seed=0, out_dir=tmp_out_dir)


def test_loader_imports_county_env(tmp_out_dir):
    _quick_synth(tmp_out_dir)
    from synthetic_env_loader import make_synthetic_env
    env = make_synthetic_env(tmp_out_dir, total_budget=20, swaps_per_step=2)
    assert env.n_blocks > 0
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")


def test_env_reset_runs(tmp_out_dir):
    _quick_synth(tmp_out_dir)
    from synthetic_env_loader import make_synthetic_env
    env = make_synthetic_env(tmp_out_dir, total_budget=20, swaps_per_step=2)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.avg_farmland_slope > 0


def test_env_step_runs(tmp_out_dir):
    _quick_synth(tmp_out_dir)
    from synthetic_env_loader import make_synthetic_env
    env = make_synthetic_env(tmp_out_dir, total_budget=20, swaps_per_step=2)
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
