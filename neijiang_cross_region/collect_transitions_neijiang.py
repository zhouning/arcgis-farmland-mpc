"""Collect 6K transitions from Neijiang Dongxing env for contrastive trainer's MSE loss.

Random policy, 60 episodes × 100 steps = 6000 transitions.
Output matches D:/adk/results_dual_dreamer_real/trajectories_6k.npz schema:
  block_features, global_features, actions, rewards, next_block_features, next_global_features
"""
import os, sys
import time
import logging
import numpy as np

sys.path.insert(0, os.environ.get("P9_ADK_DIR", "/path/to/adk"))
sys.path.insert(0, os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from county_env_neijiang import make_neijiang_env  # noqa: E402
from data_agent.transition_model import TrajectoryCollector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_EPISODES = 60
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories_6k_neijiang.npz")


def main():
    logger.info("Creating Neijiang env...")
    t0 = time.time()
    env = make_neijiang_env()
    logger.info("env init in %.1fs, n_blocks=%d", time.time() - t0, env.n_blocks)

    coll = TrajectoryCollector()
    t0 = time.time()
    coll.collect(env, n_episodes=N_EPISODES, policy='random', n_blocks=env.n_blocks)
    logger.info("Collected %d transitions in %.1fs", coll.size, time.time() - t0)

    coll.save(OUT_PATH)
    logger.info("Saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()
