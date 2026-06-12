import numpy as np

from farmland_mpc import sample


class _FakeEnv:
    n_blocks = 3
    n_parcels = 5
    max_steps = 2


def _fake_transitions(_env, _n_transition_episodes, seed_offset, say):
    say(f"fake transitions seed={seed_offset}")
    return {
        "block_features": np.zeros((1, 3, 17), dtype=np.float32),
        "global_features": np.zeros((1, 12), dtype=np.float32),
        "actions": np.array([1], dtype=np.int64),
        "rewards": np.array([1.5], dtype=np.float32),
        "next_block_features": np.ones((1, 3, 17), dtype=np.float32),
        "next_global_features": np.ones((1, 12), dtype=np.float32),
    }


def _fake_pairwise(_env, _n_states, n_actions, seed, max_outer_episodes, say):
    say(f"fake pairwise seed={seed} outer={max_outer_episodes}")
    return {
        "states_bf": np.zeros((1, 3, 17), dtype=np.float32),
        "states_gf": np.zeros((1, 12), dtype=np.float32),
        "actions": np.arange(n_actions, dtype=np.int64).reshape(1, n_actions),
        "rewards": np.linspace(0.0, 1.0, n_actions, dtype=np.float32).reshape(1, n_actions),
    }


def test_sample_forwards_reward_weight_overrides_to_county_env(tmp_path, monkeypatch):
    calls = []

    def fake_import_make_env(env_kind):
        assert env_kind == "county"

        def fake_make_env(**kwargs):
            calls.append(kwargs)
            return _FakeEnv()

        return fake_make_env

    monkeypatch.setattr(sample, "_import_make_env", fake_import_make_env)
    monkeypatch.setattr(sample, "_collect_transitions", _fake_transitions)
    monkeypatch.setattr(sample, "_collect_pairwise", _fake_pairwise)

    summary = sample.run(
        prepared_dir=tmp_path,
        n_transition_episodes=1,
        n_pairwise_states=1,
        n_pairwise_actions=2,
        seed=7,
        proj_crs="EPSG:32648",
        slope_weight=4100.0,
        cont_weight=600.0,
        baimu_weight=2300.0,
        baimu_bonus=9.0,
        baimu_area_penalty=3100.0,
    )

    assert calls == [
        {
            "prepared_dir": str(tmp_path),
            "proj_crs": "EPSG:32648",
            "slope_weight": 4100.0,
            "cont_weight": 600.0,
            "baimu_weight": 2300.0,
            "baimu_bonus": 9.0,
            "baimu_area_penalty": 3100.0,
        }
    ]
    assert summary["config"]["reward_overrides"] == {
        "slope_weight": 4100.0,
        "cont_weight": 600.0,
        "baimu_weight": 2300.0,
        "baimu_bonus": 9.0,
        "baimu_area_penalty": 3100.0,
    }
