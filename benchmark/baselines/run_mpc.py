"""Run Contrastive-MPC baseline: gen data, train ensemble, MPC eval.

Production defaults: n_models=5, epochs=30, horizon=5, top_k=50.

Imports (verified 2026-05-13):
- data_agent.transition_model lives under D:/adk/ (paper9 scripts put
  D:/adk first on sys.path -- see replicate_5seeds.py).
- mpc_planner lives at D:/test/mpc_planner.py.
- contrastive_trainer is under D:/test/paper9_contrastive/.
"""
from __future__ import annotations
import time
from pathlib import Path

from .common import build_env, init_snapshot, write_result_json


N_STATES_DEFAULT = 1000
N_ACTIONS_DEFAULT = 50


def _generate_data(env, n_states: int, n_actions: int, seed: int):
    """In-process port of paper9_contrastive/generate_pairwise_data.py main loop.

    Returns (train_data, pairwise_data) dicts ready for
    ContrastiveTransitionTrainer. Keys match what
    ContrastiveTransitionTrainer._prepare_regular expects:
    block_features, global_features, actions, rewards,
    next_block_features, next_global_features.
    Actions use replace=True sampling (like the reference script) so every
    slot holds a real valid action -- no -1 sentinel.
    """
    import numpy as np
    import sys
    sys.path.insert(0, "D:/test/paper9_contrastive")
    from generate_pairwise_data import snapshot, restore

    n_blocks = env.n_blocks
    bf0 = env._get_block_features()
    gf0 = env._get_global_features()
    k_block = bf0.shape[-1]
    k_global = gf0.shape[-1]

    rng = np.random.default_rng(seed)
    out_bf = np.zeros((n_states, n_blocks, k_block), dtype=np.float32)
    out_gf = np.zeros((n_states, k_global), dtype=np.float32)
    out_actions = np.zeros((n_states, n_actions), dtype=np.int64)
    out_rewards = np.zeros((n_states, n_actions), dtype=np.float32)
    tr_bf, tr_gf, tr_act, tr_rew, tr_bf_next, tr_gf_next = [], [], [], [], [], []

    state_count = 0
    episode = 0
    while state_count < n_states:
        env.reset(seed=int(seed * 100 + episode))
        for _ in range(env.max_steps):
            if state_count >= n_states:
                break
            bf = env._get_block_features()
            gf = env._get_global_features()
            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) < 2:
                break
            snap = snapshot(env)
            replace = len(valid) < n_actions
            chosen = rng.choice(valid, size=n_actions, replace=replace)
            rewards = np.zeros(n_actions, dtype=np.float32)
            for j, a in enumerate(chosen):
                restore(env, snap)
                _, r, _, _, _ = env.step(int(a))
                rewards[j] = float(r)
            out_bf[state_count] = bf
            out_gf[state_count] = gf
            out_actions[state_count] = chosen
            out_rewards[state_count] = rewards

            a_taken = int(chosen[0])
            restore(env, snap)
            _, r_taken, term, trunc, _ = env.step(a_taken)
            tr_bf.append(bf); tr_gf.append(gf)
            tr_act.append(a_taken); tr_rew.append(float(r_taken))
            tr_bf_next.append(env._get_block_features())
            tr_gf_next.append(env._get_global_features())
            state_count += 1
            if term or trunc:
                break
        episode += 1

    pairwise_data = {
        "states_bf": out_bf, "states_gf": out_gf,
        "actions": out_actions, "rewards": out_rewards,
    }
    train_data = {
        "block_features": np.array(tr_bf, dtype=np.float32),
        "global_features": np.array(tr_gf, dtype=np.float32),
        "actions": np.array(tr_act, dtype=np.int64),
        "rewards": np.array(tr_rew, dtype=np.float32),
        "next_block_features": np.array(tr_bf_next, dtype=np.float32),
        "next_global_features": np.array(tr_gf_next, dtype=np.float32),
    }
    return train_data, pairwise_data


def run_mpc(
    dataset_dir,
    preset_id: str,
    seed: int,
    out_path,
    total_budget: int = 100,
    swaps_per_step: int = 5,
    n_models: int = 5,
    epochs: int = 30,
    horizon: int = 5,
    top_k: int = 50,
    lambda_rank: float = 5.0,
    n_states: int = N_STATES_DEFAULT,
    n_actions: int = N_ACTIONS_DEFAULT,
    device: str = "cpu",
) -> dict:
    import sys
    sys.path.insert(0, "D:/adk")
    sys.path.insert(0, "D:/test")
    sys.path.insert(0, "D:/test/paper9_contrastive")
    import numpy as np
    import torch
    from data_agent.transition_model import TransitionModel, EnsembleTransitionModel
    from contrastive_trainer import ContrastiveTransitionTrainer
    from mpc_planner import mpc_select_action
    from eval.metrics import extract_run_result

    env = build_env(dataset_dir, total_budget=total_budget,
                    swaps_per_step=swaps_per_step)

    t0 = time.time()
    train_data, pairwise_data = _generate_data(env, n_states=n_states,
                                                n_actions=n_actions, seed=seed)
    data_wall = time.time() - t0

    n_blocks = env.n_blocks
    ensemble = EnsembleTransitionModel(n_blocks, n_models=n_models)
    t0 = time.time()
    for m_idx in range(n_models):
        torch.manual_seed(seed * 1000 + m_idx)
        np.random.seed(seed * 1000 + m_idx)
        m = TransitionModel(n_blocks)
        trainer = ContrastiveTransitionTrainer(
            m, lr=1e-3, epochs=epochs, patience=8,
            lambda_rank=lambda_rank, margin=0.1, n_pairs_per_state=10,
        )
        trainer.train(train_data, pairwise_data)
        ensemble.models[m_idx] = m
    train_wall = time.time() - t0

    env.reset(seed=seed)
    init = init_snapshot(env)
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    eval_t0 = time.time()
    for _ in range(env.max_steps):
        bf = env._get_block_features()
        gf = env._get_global_features()
        mask = env.action_masks()
        action, _ = mpc_select_action(
            ensemble, bf, gf, mask,
            horizon=horizon, top_k=top_k, gamma=0.99,
            n_rollouts=1, continuation="greedy",
            greedy_sample=top_k, scoring="reward", rng=rng,
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    eval_wall = time.time() - eval_t0

    result = extract_run_result(
        preset_id=preset_id, seed=seed, method="Contrastive-MPC",
        env=env, init_snap=init,
        total_reward=total_reward,
        wall_seconds=data_wall + train_wall + eval_wall,
        exchange_pairs_used=int(env.budget_used),
        extra={
            "n_models": n_models,
            "epochs": epochs,
            "horizon": horizon,
            "top_k": top_k,
            "lambda_rank": lambda_rank,
            "n_states": n_states,
            "n_actions": n_actions,
            "device": device,
            "data_wall_seconds": data_wall,
            "train_wall_seconds": train_wall,
            "eval_wall_seconds": eval_wall,
        },
    )
    write_result_json(result, out_path)
    return result
