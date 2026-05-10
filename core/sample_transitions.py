"""Tool 2: Sample transitions + pairwise ranking data.

Two outputs under <prepared_dir>/tool2/:

    transitions.npz  (for contrastive trainer's MSE loss)
        block_features        (N, n_blocks, K_BLOCK)
        global_features       (N, K_GLOBAL)
        actions               (N,) int64
        rewards               (N,) float32
        next_block_features   (N, n_blocks, K_BLOCK)
        next_global_features  (N, K_GLOBAL)

    pairwise.npz     (for contrastive trainer's ranking loss)
        states_bf   (M, n_blocks, K_BLOCK)
        states_gf   (M, K_GLOBAL)
        actions     (M, N_ACTIONS) int64
        rewards     (M, N_ACTIONS) float32

Runtime estimate:
    - transitions: N_episodes * 100 steps * ~10ms/step ~= N_episodes seconds
        at county scale (2600 blocks). For 60 episodes ~= 10-20 min.
    - pairwise: N_states * N_actions snapshot/restore cycles. Dominant cost is
        env.step which is ~10-30 ms; for 1000x50 = 50K steps ~= 10-25 min.
    Total county scale: 30-60 min CPU.
    Small regions (30 blocks): ~1 min total.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Mutable env-state attributes. Used for snapshot/restore in pairwise pass.
# Keep in sync with county_env.CountyLevelEnv's state surface.
_STATE_ATTRS = [
    "land_use", "swapped", "budget_used", "step_count", "swaps_in_block",
    "n_farmland", "n_forest", "total_weighted_slope", "total_farm_area",
    "farmland_nbr_count", "total_farmland_adj",
    "_block_farm_avail", "_block_forest_avail",
    "baimu_count", "baimu_total_area",
    "prev_slope", "prev_cont", "prev_baimu_count", "prev_baimu_area",
]


def _snapshot(env):
    snap = {}
    for attr in _STATE_ATTRS:
        val = getattr(env, attr)
        snap[attr] = val.copy() if isinstance(val, np.ndarray) else val
    return snap


def _restore(env, snap):
    for attr in _STATE_ATTRS:
        val = snap[attr]
        if isinstance(val, np.ndarray):
            getattr(env, attr)[:] = val
        else:
            setattr(env, attr, val)


def _collect_transitions(env, n_episodes: int, seed_offset: int, say):
    """Random-policy rollouts. Mirrors TrajectoryCollector.collect but
    doesn't drag in gym/adk imports."""
    from county_env import K_BLOCK  # patched; safe to import after make_env
    K_BLOCK  # used below via env shape

    bf_list, gf_list, a_list, r_list, nbf_list, ngf_list = [], [], [], [], [], []
    t0 = time.time()

    # K_GLOBAL may differ between variants (K_GLOBAL_COUNTY=12 in Paper 9);
    # infer from env.observation_space shape and env.n_blocks.
    obs_dim = env.observation_space.shape[0]
    k_global = obs_dim - env.n_blocks * 17  # K_BLOCK is 17 in Paper 9

    rng = np.random.default_rng(seed_offset)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_steps = 0
        while not done:
            mask = env.action_masks() if hasattr(env, "action_masks") else None
            valid = np.where(mask)[0] if mask is not None else np.arange(env.n_blocks)
            if len(valid) == 0:
                break
            action = int(rng.choice(valid))

            # Split current obs into bf / gf
            bf = obs[: env.n_blocks * 17].reshape(env.n_blocks, 17)
            gf = obs[env.n_blocks * 17: env.n_blocks * 17 + k_global]

            next_obs, reward, terminated, truncated, _ = env.step(action)
            nbf = next_obs[: env.n_blocks * 17].reshape(env.n_blocks, 17)
            ngf = next_obs[env.n_blocks * 17: env.n_blocks * 17 + k_global]

            bf_list.append(bf.copy())
            gf_list.append(gf.copy())
            a_list.append(action)
            r_list.append(reward)
            nbf_list.append(nbf.copy())
            ngf_list.append(ngf.copy())

            obs = next_obs
            ep_steps += 1
            done = terminated or truncated

        say(f"    episode {ep + 1}/{n_episodes} done ({ep_steps} steps, "
            f"total {len(a_list)} transitions, {time.time() - t0:.1f}s)")

    return {
        "block_features":      np.array(bf_list,  dtype=np.float32),
        "global_features":     np.array(gf_list,  dtype=np.float32),
        "actions":             np.array(a_list,   dtype=np.int64),
        "rewards":             np.array(r_list,   dtype=np.float32),
        "next_block_features": np.array(nbf_list, dtype=np.float32),
        "next_global_features": np.array(ngf_list, dtype=np.float32),
    }


def _collect_pairwise(env, n_states: int, n_actions: int, seed: int,
                      max_outer_episodes: int, say):
    """For each sampled state, snapshot env, try n_actions random valid
    actions (restoring state between each), record true rewards."""
    n_blocks = env.n_blocks
    # Same K_GLOBAL inference as transitions collector
    obs_dim = env.observation_space.shape[0]
    k_global = obs_dim - n_blocks * 17

    rng = np.random.default_rng(seed)
    out_bf = np.zeros((n_states, n_blocks, 17), dtype=np.float32)
    out_gf = np.zeros((n_states, k_global),     dtype=np.float32)
    out_a  = np.zeros((n_states, n_actions),    dtype=np.int64)
    out_r  = np.zeros((n_states, n_actions),    dtype=np.float32)

    t0 = time.time()
    state_count = 0
    episode = 0

    while state_count < n_states and episode < max_outer_episodes:
        env.reset(seed=seed + episode)
        for step in range(env.max_steps):
            if state_count >= n_states:
                break

            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) < 2:
                # advance anyway
                if len(valid) == 0:
                    break
                _, _, terminated, truncated, _ = env.step(int(valid[0]))
                if terminated or truncated:
                    break
                continue

            bf = env._get_block_features()
            gf = env._get_global_features()

            # Sample n_actions valid actions (with-replacement if shortfall)
            if len(valid) >= n_actions:
                sampled = rng.choice(valid, size=n_actions, replace=False)
            else:
                sampled = rng.choice(valid, size=n_actions, replace=True)

            out_bf[state_count] = bf
            out_gf[state_count] = gf
            out_a[state_count]  = sampled

            snap = _snapshot(env)
            for j, a in enumerate(sampled):
                _, r, _, _, _ = env.step(int(a))
                out_r[state_count, j] = r
                _restore(env, snap)

            state_count += 1
            if state_count % 100 == 0 or state_count == n_states:
                elapsed = time.time() - t0
                eta = elapsed / state_count * (n_states - state_count) if state_count else 0
                say(f"    pairwise: {state_count}/{n_states} states "
                    f"({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")

            # Advance state with a random step
            action = int(rng.choice(valid))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        episode += 1

    return {
        "states_bf": out_bf[:state_count],
        "states_gf": out_gf[:state_count],
        "actions":   out_a[:state_count],
        "rewards":   out_r[:state_count],
    }


def run(prepared_dir: str,
        n_transition_episodes: int = 60,
        n_pairwise_states: int = 1000,
        n_pairwise_actions: int = 50,
        seed: int = 0,
        proj_crs: Optional[str] = None,
        messages=None):
    """Sample transitions + pairwise data from the env built on prepared_dir.

    Parameters
    ----------
    prepared_dir : str
        Output of Tool 1.
    n_transition_episodes : int
        How many random-policy episodes to collect. Reference run used
        60 on a county-scale environment (~6000 transitions).
    n_pairwise_states : int
        How many state snapshots to record (each with n_pairwise_actions
        rewards). Reference run used 1000.
    n_pairwise_actions : int
        How many actions to evaluate per state. Reference run used 50.
        Capped to env.n_blocks automatically.
    seed : int
        Base seed. Transitions use seeds [seed, seed + n_episodes);
        pairwise uses seed + 10000 to avoid overlap.
    proj_crs : str or None
        Forwarded to make_env. Default = whatever make_env picks.
    messages : arcpy messages or None
    """
    def _say(msg, level="info"):
        if messages is not None:
            getattr(messages, "addMessage" if level == "info"
                    else "addWarningMessage")(msg)
        logger.info(msg)
        print(msg, flush=True)

    prepared_dir = Path(prepared_dir)
    out_dir = prepared_dir / "tool2"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "sample_transitions.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    try:
        # Ensure toolbox dir + core dir on sys.path
        toolbox_dir = str(Path(__file__).resolve().parent.parent)
        if toolbox_dir not in sys.path:
            sys.path.insert(0, toolbox_dir)
        core_dir = str(Path(__file__).resolve().parent)
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)

        _say(f"[Tool 2] Sampling data under {out_dir}")
        _say(f"  n_transition_episodes = {n_transition_episodes}")
        _say(f"  n_pairwise_states     = {n_pairwise_states}")
        _say(f"  n_pairwise_actions    = {n_pairwise_actions}")
        _say(f"  seed                  = {seed}")

        # Build env
        _say("\n[Tool 2] Building env via make_env ...")
        t0 = time.time()
        from core.blocks_env import make_env
        env = make_env(prepared_dir=str(prepared_dir), proj_crs=proj_crs)
        _say(f"  env built in {time.time() - t0:.1f}s; "
             f"n_blocks={env.n_blocks}, n_parcels={env.n_parcels}, "
             f"max_steps={env.max_steps}")

        # Cap n_pairwise_actions to env.n_blocks
        if n_pairwise_actions > env.n_blocks:
            _say(f"  capping n_pairwise_actions {n_pairwise_actions} -> "
                 f"{env.n_blocks} (env has {env.n_blocks} blocks)",
                 level="warn")
            n_pairwise_actions = env.n_blocks

        summary = {
            "config": {
                "prepared_dir": str(prepared_dir),
                "n_transition_episodes": n_transition_episodes,
                "n_pairwise_states": n_pairwise_states,
                "n_pairwise_actions": n_pairwise_actions,
                "seed": seed, "proj_crs": proj_crs,
                "n_blocks": int(env.n_blocks),
                "n_parcels": int(env.n_parcels),
            },
        }

        # Transitions
        _say("\n[Tool 2] Collecting transitions (random policy) ...")
        t_tr = time.time()
        trans = _collect_transitions(env, n_transition_episodes,
                                     seed_offset=seed, say=_say)
        transitions_path = out_dir / "transitions.npz"
        np.savez_compressed(transitions_path, **trans)
        summary["transitions"] = {
            "elapsed_s": round(time.time() - t_tr, 1),
            "n_transitions": int(len(trans["actions"])),
            "output": str(transitions_path),
        }
        _say(f"[Tool 2] transitions: {summary['transitions']['n_transitions']} "
             f"rows in {summary['transitions']['elapsed_s']}s -> {transitions_path}")

        # Pairwise
        _say("\n[Tool 2] Collecting pairwise data (state + N-actions reward) ...")
        t_pw = time.time()
        # Heuristic: allow up to 2x (n_pairwise_states / env.max_steps) episodes
        max_outer = max(5, int(2 * n_pairwise_states / max(env.max_steps, 1)))
        pw = _collect_pairwise(env, n_pairwise_states, n_pairwise_actions,
                               seed=seed + 10000,
                               max_outer_episodes=max_outer, say=_say)
        pairwise_path = out_dir / "pairwise.npz"
        np.savez_compressed(pairwise_path, **pw)

        # Quick stats
        r = pw["rewards"]
        r_std_per_state = r.std(axis=1)
        summary["pairwise"] = {
            "elapsed_s": round(time.time() - t_pw, 1),
            "n_states": int(len(pw["states_bf"])),
            "n_actions_per_state": int(n_pairwise_actions),
            "reward_std_median": float(np.median(r_std_per_state)) if len(r) else 0.0,
            "reward_mean": float(r.mean()) if r.size else 0.0,
            "reward_range": [float(r.min()), float(r.max())] if r.size else [0.0, 0.0],
            "output": str(pairwise_path),
        }
        _say(f"[Tool 2] pairwise: {summary['pairwise']['n_states']} states "
             f"× {summary['pairwise']['n_actions_per_state']} actions in "
             f"{summary['pairwise']['elapsed_s']}s -> {pairwise_path}")
        _say(f"  reward std per state (median): "
             f"{summary['pairwise']['reward_std_median']:.4f}")

        summary_path = out_dir / "sample_transitions_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _say(f"\n[Tool 2] Done. Summary -> {summary_path}")
        return summary
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
