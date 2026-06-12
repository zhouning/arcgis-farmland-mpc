import numpy as np

from farmland_mpc.ensemble_runner import EnsembleOrtRunner
from farmland_mpc.mpc_plan import _greedy_1step_actions, _run_episode, mpc_select_action


class _FakeSession:
    def __init__(self, offset):
        self.offset = np.float32(offset)

    def run(self, _outputs, feeds):
        bf = feeds["block_features"]
        gf = feeds["global_features"]
        batch = bf.shape[0]
        return (
            bf + self.offset,
            gf + self.offset * 10,
            np.full((batch, 1), self.offset, dtype=np.float32),
        )


def test_batch_predict_streams_member_outputs_without_stack(monkeypatch):
    runner = EnsembleOrtRunner.__new__(EnsembleOrtRunner)
    runner._sessions = [_FakeSession(1), _FakeSession(2), _FakeSession(3)]

    def fail_stack(*_args, **_kwargs):
        raise AssertionError("member outputs should be reduced without np.stack")

    monkeypatch.setattr("farmland_mpc.ensemble_runner.np.stack", fail_stack)

    bf = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    gf = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    actions = np.array([0, 2], dtype=np.int64)

    nbf, ngf, reward_mean, reward_std = runner.batch_predict(bf, gf, actions)

    np.testing.assert_allclose(nbf, bf + 2)
    np.testing.assert_allclose(ngf, gf + 20)
    np.testing.assert_allclose(reward_mean, np.full(2, 2.0, dtype=np.float32))
    np.testing.assert_allclose(
        reward_std,
        np.full(2, np.std([1.0, 2.0, 3.0], dtype=np.float32), dtype=np.float32),
    )


def test_batch_predict_normalizes_next_states_in_place(monkeypatch):
    runner = EnsembleOrtRunner.__new__(EnsembleOrtRunner)
    runner._sessions = [_FakeSession(1), _FakeSession(2)]
    in_place_divides = []
    real_divide = np.divide

    def tracking_divide(a, b, *args, **kwargs):
        out = kwargs.get("out")
        if isinstance(a, np.ndarray) and a.shape == (2, 3, 4):
            in_place_divides.append(out is a)
        return real_divide(a, b, *args, **kwargs)

    monkeypatch.setattr("farmland_mpc.ensemble_runner.np.divide", tracking_divide)

    bf = np.ones((2, 3, 4), dtype=np.float32)
    gf = np.ones((2, 5), dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int64)

    runner.batch_predict(bf, gf, actions)

    assert in_place_divides
    assert all(in_place_divides)


class _BatchLimitedEnsemble:
    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.seen_batches = []

    def batch_predict(self, block_features, global_features, actions):
        batch = len(actions)
        self.seen_batches.append(batch)
        if batch > self.max_batch:
            raise AssertionError(f"batch {batch} exceeds limit {self.max_batch}")

        next_block = block_features.copy()
        next_global = global_features.copy()
        reward = actions.astype(np.float32)
        next_global[:, 4] = reward
        return next_block, next_global, reward, np.zeros(batch, dtype=np.float32)


def test_mpc_select_action_chunks_stage1_and_greedy_rollout_batches():
    ensemble = _BatchLimitedEnsemble(max_batch=4)
    block_features = np.zeros((7, 17), dtype=np.float32)
    global_features = np.zeros(12, dtype=np.float32)
    action_mask = np.ones(7, dtype=bool)

    chosen, info = mpc_select_action(
        ensemble,
        block_features,
        global_features,
        action_mask,
        horizon=2,
        top_k=3,
        gamma=0.99,
        continuation="greedy",
        greedy_sample=5,
        scoring="reward",
        rng=np.random.default_rng(0),
        batch_size=4,
    )

    assert chosen == 6
    assert info["n_valid"] == 7
    assert info["n_candidates"] == 3
    assert max(ensemble.seen_batches) <= 4


class _OneStepEnv:
    max_steps = 1

    def reset(self, seed):
        self.seed = seed

    def _get_block_features(self):
        return np.zeros((10, 17), dtype=np.float32)

    def _get_global_features(self):
        return np.zeros(12, dtype=np.float32)

    def action_masks(self):
        return np.ones(10, dtype=bool)

    def step(self, action):
        return None, 0.0, True, False, {"budget_used": 1}


def test_run_episode_forwards_mpc_batch_size():
    ensemble = _BatchLimitedEnsemble(max_batch=4)

    info = _run_episode(
        _OneStepEnv(),
        ensemble,
        horizon=1,
        top_k=3,
        gamma=0.99,
        continuation="greedy",
        scoring="reward",
        seed=0,
        batch_size=4,
    )

    assert info["steps_run"] == 1
    assert max(ensemble.seen_batches) <= 4


class _RewardOnlyEnsemble:
    def __init__(self):
        self.reward_batches = []

    def batch_predict(self, *_args, **_kwargs):
        raise AssertionError("greedy continuation should not request next states")

    def batch_predict_rewards(self, block_features, global_features, actions):
        batch = len(actions)
        self.reward_batches.append(batch)
        return actions.astype(np.float32), np.zeros(batch, dtype=np.float32)


def test_greedy_continuation_uses_reward_only_predictions():
    ensemble = _RewardOnlyEnsemble()
    cur_bf = np.zeros((3, 7, 17), dtype=np.float32)
    cur_gf = np.zeros((3, 12), dtype=np.float32)
    valid_actions = np.arange(6, dtype=np.int64)

    actions = _greedy_1step_actions(
        ensemble,
        cur_bf,
        cur_gf,
        valid_actions,
        n_sample=6,
        rng=np.random.default_rng(0),
        batch_size=4,
    )

    np.testing.assert_array_equal(actions, np.full(3, 5, dtype=np.int64))
    assert max(ensemble.reward_batches) <= 4
