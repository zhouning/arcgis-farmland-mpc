"""Generate pairwise ranking data from Neijiang Dongxing environment.

Identical logic to D:/test/paper9_contrastive/generate_pairwise_data.py,
but with make_neijiang_env() so the module-level constants point to
Neijiang's DLTB_with_slope.gpkg and block directory.

Output: D:/test/neijiang_cross_region/pairwise_data_neijiang.npz
  - states_bf: (N_STATES, n_blocks=3711, K_BLOCK=17)
  - states_gf: (N_STATES, K_GLOBAL=12)
  - actions:   (N_STATES, 50) int64
  - rewards:   (N_STATES, 50) float32
"""
import os, sys
import time
import logging
import numpy as np

sys.path.insert(0, os.environ.get("P9_ADK_DIR", "/path/to/adk"))
sys.path.insert(0, os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from county_env_neijiang import make_neijiang_env  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_STATES = 1000
N_ACTIONS = 50
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_data_neijiang.npz")

STATE_ATTRS = [
    'land_use', 'swapped', 'budget_used', 'step_count', 'swaps_in_block',
    'n_farmland', 'n_forest', 'total_weighted_slope', 'total_farm_area',
    'farmland_nbr_count', 'total_farmland_adj',
    '_block_farm_avail', '_block_forest_avail',
    'baimu_count', 'baimu_total_area',
    'prev_slope', 'prev_cont', 'prev_baimu_count', 'prev_baimu_area',
]


def snapshot(env):
    snap = {}
    for attr in STATE_ATTRS:
        val = getattr(env, attr)
        if isinstance(val, np.ndarray):
            snap[attr] = val.copy()
        else:
            snap[attr] = val
    return snap


def restore(env, snap):
    for attr in STATE_ATTRS:
        val = snap[attr]
        if isinstance(val, np.ndarray):
            getattr(env, attr)[:] = val
        else:
            setattr(env, attr, val)


def main():
    logger.info("Creating Neijiang environment...")
    env = make_neijiang_env()
    obs, _ = env.reset(seed=0)
    n_blocks = env.n_blocks
    logger.info("  n_blocks=%d, max_steps=%d", n_blocks, env.max_steps)

    from county_env import K_BLOCK, K_GLOBAL_COUNTY

    rng = np.random.default_rng(42)
    out_bf = np.zeros((N_STATES, n_blocks, K_BLOCK), dtype=np.float32)
    out_gf = np.zeros((N_STATES, K_GLOBAL_COUNTY), dtype=np.float32)
    out_actions = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    out_rewards = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)

    t0 = time.time()
    state_count = 0
    episode = 0
    while state_count < N_STATES:
        obs, _ = env.reset(seed=episode)
        for step in range(env.max_steps):
            if state_count >= N_STATES:
                break

            should_sample = True
            if should_sample:
                bf = env._get_block_features()
                gf = env._get_global_features()
                mask = env.action_masks()
                valid_actions = np.where(mask)[0]
                if len(valid_actions) < 2:
                    should_sample = False

            if should_sample:
                out_bf[state_count] = bf
                out_gf[state_count] = gf

                if len(valid_actions) >= N_ACTIONS:
                    sampled = rng.choice(valid_actions, size=N_ACTIONS, replace=False)
                else:
                    sampled = rng.choice(valid_actions, size=N_ACTIONS, replace=True)
                out_actions[state_count] = sampled

                snap = snapshot(env)
                for j, a in enumerate(sampled):
                    _, r, _, _, _ = env.step(int(a))
                    out_rewards[state_count, j] = r
                    restore(env, snap)

                state_count += 1
                if state_count % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / state_count * (N_STATES - state_count)
                    logger.info("  %d/%d states done (%.1fs elapsed, ETA %.1fs)",
                                state_count, N_STATES, elapsed, eta)

            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            a = int(rng.choice(valid))
            _, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break
        episode += 1
        if episode > 60:
            logger.warning("Too many episodes, stopping at %d states", state_count)
            break

    elapsed = time.time() - t0
    logger.info("Done! %d states × %d actions in %.1fs", state_count, N_ACTIONS, elapsed)

    out_bf = out_bf[:state_count]
    out_gf = out_gf[:state_count]
    out_actions = out_actions[:state_count]
    out_rewards = out_rewards[:state_count]

    reward_stds = out_rewards.std(axis=1)
    reward_means = out_rewards.mean(axis=1)
    logger.info("Reward std across actions: mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
                reward_stds.mean(), np.median(reward_stds), reward_stds.min(), reward_stds.max())
    logger.info("Reward mean: %.4f, range: [%.4f, %.4f]",
                reward_means.mean(), out_rewards.min(), out_rewards.max())

    np.savez_compressed(OUT_PATH,
                        states_bf=out_bf, states_gf=out_gf,
                        actions=out_actions, rewards=out_rewards)
    logger.info("Saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()
