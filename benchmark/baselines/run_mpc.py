"""Run the contrastive MPC baseline on a synthetic dataset.

Production defaults match Paper 9 contrastive: 5-member ensemble, 30 epochs,
horizon=5, top_k=50, n_states=1000, n_actions=50.

Notes on imports (verified 2026-05-13):
- data_agent.transition_model lives under D:/adk/ (paper9 scripts put
  D:/adk first on sys.path -- see replicate_5seeds.py).
- mpc_planner lives at D:/test/mpc_planner.py.
- contrastive_trainer is under D:/test/paper9_contrastive/.
- common.py centralizes the sys.path setup (reads TEST_SRC_ROOT/ADK_SRC_ROOT
  env vars; falls back to D:/test and D:/adk for local dev).

Checkpoint design (added 2026-05-16 after Colab plain_large_cons cell ran
11.7h without completing, then got killed at the 24h VM limit before the
result JSON could be written):

- `_generate_data` writes an npz snapshot to ckpt_dir every CKPT_EVERY
  states. On resume, it loads the latest snapshot and continues from
  state_count.
- `run_mpc` skips the data-generation phase entirely if a complete
  pairwise.npz / train.npz pair exists.
- Ensemble training is short enough (1-2h) to not bother checkpointing.
- MPC eval is short (~30 min on 4500 blocks) and depends on a deterministic
  env state -- not worth checkpointing.

Checkpoint dir layout:
  out_path.parent / f"_mpc_ckpt/{preset_id}_seed{seed}/" /
      state_<N>.npz    (partial: state_count, out_bf, out_gf, out_actions,
                        out_rewards, tr_*, episode, rng_state)
      pairwise.npz     (final: written when state_count == n_states)
      train.npz        (final: matches pairwise.npz)

When pairwise.npz + train.npz both exist, _generate_data short-circuits.
"""
from __future__ import annotations
import time
from pathlib import Path

from .common import build_env, init_snapshot, write_result_json


N_STATES_DEFAULT = 1000
N_ACTIONS_DEFAULT = 50
CKPT_EVERY = 50


def _load_final_data(ckpt_dir: Path):
    """Return (train_data, pairwise_data) if both final npz files exist."""
    import numpy as np
    pw_path = ckpt_dir / "pairwise.npz"
    tr_path = ckpt_dir / "train.npz"
    if not (pw_path.is_file() and tr_path.is_file()):
        return None
    with np.load(pw_path) as pw:
        pairwise_data = {
            "states_bf": pw["states_bf"].copy(),
            "states_gf": pw["states_gf"].copy(),
            "actions": pw["actions"].copy(),
            "rewards": pw["rewards"].copy(),
        }
    with np.load(tr_path) as tr:
        train_data = {
            "block_features": tr["block_features"].copy(),
            "global_features": tr["global_features"].copy(),
            "actions": tr["actions"].copy(),
            "rewards": tr["rewards"].copy(),
            "next_block_features": tr["next_block_features"].copy(),
            "next_global_features": tr["next_global_features"].copy(),
        }
    return train_data, pairwise_data


def _latest_partial(ckpt_dir: Path):
    """Return path of the highest-numbered state_<N>.npz, or None."""
    if not ckpt_dir.is_dir():
        return None
    snapshots = sorted(ckpt_dir.glob("state_*.npz"),
                       key=lambda p: int(p.stem.split("_")[1]))
    return snapshots[-1] if snapshots else None


def _generate_data(env, n_states: int, n_actions: int, seed: int,
                   ckpt_dir: Path | None = None):
    """In-process port of paper9_contrastive/generate_pairwise_data.py main loop.

    Returns (train_data, pairwise_data). If ckpt_dir is set, writes
    intermediate state_<N>.npz every CKPT_EVERY states and resumes from the
    latest one. Writes final pairwise.npz + train.npz when complete.
    """
    import numpy as np
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

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest = _latest_partial(ckpt_dir)
        if latest is not None:
            print(f"[mpc] resuming pairwise data from {latest.name}", flush=True)
            with np.load(latest, allow_pickle=True) as d:
                state_count = int(d["state_count"])
                out_bf[:state_count] = d["out_bf"][:state_count]
                out_gf[:state_count] = d["out_gf"][:state_count]
                out_actions[:state_count] = d["out_actions"][:state_count]
                out_rewards[:state_count] = d["out_rewards"][:state_count]
                tr_bf = list(d["tr_bf"])
                tr_gf = list(d["tr_gf"])
                tr_act = list(d["tr_act"])
                tr_rew = list(d["tr_rew"])
                tr_bf_next = list(d["tr_bf_next"])
                tr_gf_next = list(d["tr_gf_next"])
                episode = int(d["episode"])
                rng_state = d["rng_state"].item()
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state
            print(f"[mpc] resumed at state_count={state_count}, episode={episode}", flush=True)

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

            if ckpt_dir is not None and state_count % CKPT_EVERY == 0:
                tmp = ckpt_dir / f"state_{state_count}.tmp.npz"
                final = ckpt_dir / f"state_{state_count}.npz"
                # Save SLICED + COMPRESSED so a 50/1000 partial doesn't drag
                # 95% zeros to disk. tr_* lists are already correct-length.
                np.savez_compressed(
                    str(tmp.with_suffix("")),  # auto-appends .npz
                    state_count=state_count,
                    out_bf=out_bf[:state_count],
                    out_gf=out_gf[:state_count],
                    out_actions=out_actions[:state_count],
                    out_rewards=out_rewards[:state_count],
                    tr_bf=np.array(tr_bf, dtype=object),
                    tr_gf=np.array(tr_gf, dtype=object),
                    tr_act=np.array(tr_act),
                    tr_rew=np.array(tr_rew),
                    tr_bf_next=np.array(tr_bf_next, dtype=object),
                    tr_gf_next=np.array(tr_gf_next, dtype=object),
                    episode=episode,
                    rng_state=np.array(rng.bit_generator.state, dtype=object),
                )
                if final.exists():
                    final.unlink()
                tmp.rename(final)
                # Delete older snapshots to save space
                for old in ckpt_dir.glob("state_*.npz"):
                    if old != final and old.suffix == ".npz" and ".tmp" not in old.name:
                        try:
                            n = int(old.stem.split("_")[1])
                            if n < state_count:
                                old.unlink()
                        except (ValueError, IndexError):
                            pass
                print(f"[mpc] ckpt state_count={state_count}/{n_states} "
                      f"({final.stat().st_size//1024} KB)", flush=True)

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

    if ckpt_dir is not None:
        np.savez_compressed(str(ckpt_dir / "pairwise"), **pairwise_data)
        np.savez_compressed(str(ckpt_dir / "train"), **train_data)
        for old in ckpt_dir.glob("state_*.npz"):
            old.unlink()
        print(f"[mpc] final pairwise+train written, partial ckpts cleaned", flush=True)

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
    import numpy as np
    import torch
    from data_agent.transition_model import TransitionModel, EnsembleTransitionModel
    from contrastive_trainer import ContrastiveTransitionTrainer
    from mpc_planner import mpc_select_action
    from eval.metrics import extract_run_result

    out_path = Path(out_path)
    ckpt_dir = out_path.parent / "_mpc_ckpt" / f"{preset_id}_seed{seed}"

    env = build_env(dataset_dir, total_budget=total_budget,
                    swaps_per_step=swaps_per_step)

    t0 = time.time()
    cached = _load_final_data(ckpt_dir)
    if cached is not None:
        print(f"[mpc] loaded cached pairwise+train data from {ckpt_dir}", flush=True)
        train_data, pairwise_data = cached
    else:
        train_data, pairwise_data = _generate_data(
            env, n_states=n_states, n_actions=n_actions, seed=seed,
            ckpt_dir=ckpt_dir,
        )
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

    # Clean up final cached data once result JSON is written
    if ckpt_dir.is_dir():
        for f in ckpt_dir.iterdir():
            f.unlink()
        ckpt_dir.rmdir()

    return result
