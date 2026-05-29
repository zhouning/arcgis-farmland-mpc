"""MPC eval for Neijiang ensembles (baseline or partial transfer).

Loads 5-seed × 3-member ensemble from D:/test/neijiang_cross_region/ensembles/<mode>/,
runs MPC H=5 K=50 greedy with greedy_sample=50 scoring=reward on Neijiang env,
5 episodes × 5 seeds, outputs 5seed_multiobj_results_<mode>.json.
"""
import os, sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("P9_ADK_DIR", "/path/to/adk"))
sys.path.insert(0, os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout"))
sys.path.insert(0, os.path.join(os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout"), "paper9_contrastive"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_agent.transition_model import TransitionModel, EnsembleTransitionModel  # noqa: E402
from mpc_planner import mpc_select_action  # noqa: E402
from county_env_neijiang import make_neijiang_env  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENSEMBLE_ROOT = Path(os.path.dirname(os.path.abspath(__file__))) / "ensembles"
N_MEMBERS = 3
LAMBDA_RANK = 5.0


def run_mpc_eval(ensemble, env, seed, n_episodes=5, horizon=5, top_k=50):
    slopes, rewards, times = [], [], []
    cont_deltas, cont_pcts = [], []
    baimu_count_deltas, baimu_area_deltas_ha = [], []
    for ep in range(n_episodes):
        t0 = time.time()
        obs, _ = env.reset(seed=seed * 100 + ep)
        rng = np.random.default_rng(seed * 100 + ep)
        total_reward = 0.0
        init_cont = float(env.initial_cont)
        init_baimu_count = int(env.initial_baimu_count)
        init_baimu_area = float(env.initial_baimu_area)
        for step in range(env.max_steps):
            bf = env._get_block_features()
            gf = env._get_global_features()
            mask = env.action_masks()
            action, _ = mpc_select_action(
                ensemble, bf, gf, mask,
                horizon=horizon, top_k=top_k, gamma=0.99,
                n_rollouts=1, continuation="greedy",
                greedy_sample=50, scoring="reward",
                rng=rng,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        ep_time = time.time() - t0
        slope_pct = (env.avg_farmland_slope - env.initial_slope) / env.initial_slope * 100
        final_cont = float(env.contiguity)
        cont_delta = final_cont - init_cont
        cont_pct = cont_delta / (abs(init_cont) + 1e-8) * 100
        baimu_count_delta = int(env.baimu_count) - init_baimu_count
        baimu_area_delta_ha = (float(env.baimu_total_area) - init_baimu_area) / 10000.0

        slopes.append(slope_pct)
        rewards.append(total_reward)
        times.append(ep_time)
        cont_deltas.append(cont_delta)
        cont_pcts.append(cont_pct)
        baimu_count_deltas.append(baimu_count_delta)
        baimu_area_deltas_ha.append(baimu_area_delta_ha)
        logger.info(
            "  seed=%d ep=%d: slope=%.4f%% cont_delta=%+.4f (%+.2f%%) "
            "baimu_count%+d area%+.1fha reward=%.2f time=%.1fs",
            seed, ep, slope_pct, cont_delta, cont_pct,
            baimu_count_delta, baimu_area_delta_ha, total_reward, ep_time,
        )
    return {
        'slopes': slopes,
        'rewards': rewards,
        'times': times,
        'cont_deltas': cont_deltas,
        'cont_pcts': cont_pcts,
        'baimu_count_deltas': baimu_count_deltas,
        'baimu_area_deltas_ha': baimu_area_deltas_ha,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['baseline', 'partial'], required=True)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seeds", type=str, default=None)
    args = parser.parse_args()

    ens_dir = ENSEMBLE_ROOT / args.mode
    out_path = ENSEMBLE_ROOT.parent / f"5seed_multiobj_results_{args.mode}.json"

    logger.info("Creating Neijiang env...")
    env = make_neijiang_env()
    env.reset(seed=0)
    n_blocks = env.n_blocks
    logger.info("  n_blocks=%d", n_blocks)

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = list(range(args.n_seeds))

    all_results = {}
    for seed in seeds:
        logger.info("=" * 60)
        logger.info("EVAL SEED %d [%s]", seed, args.mode)
        save_path = ens_dir / f"ensemble_seed{seed}_lam{LAMBDA_RANK}"
        ens = EnsembleTransitionModel(n_blocks, n_models=N_MEMBERS)
        for i in range(N_MEMBERS):
            ens.models[i].load_state_dict(
                torch.load(f"{save_path}_member{i}.pt", map_location='cpu'))
            ens.models[i].eval()
        ep = run_mpc_eval(ens, env, seed, n_episodes=args.eval_episodes)
        all_results[seed] = ep

    per_seed_slope = [float(np.mean(all_results[s]['slopes'])) for s in seeds]
    per_seed_cont_pct = [float(np.mean(all_results[s]['cont_pcts'])) for s in seeds]
    per_seed_cont_delta = [float(np.mean(all_results[s]['cont_deltas'])) for s in seeds]
    per_seed_baimu_count = [float(np.mean(all_results[s]['baimu_count_deltas'])) for s in seeds]
    per_seed_baimu_area = [float(np.mean(all_results[s]['baimu_area_deltas_ha'])) for s in seeds]
    per_seed_reward = [float(np.mean(all_results[s]['rewards'])) for s in seeds]

    def _stats(arr):
        a = np.array(arr, dtype=float)
        return float(a.mean()), float(a.std())

    slope_mean, slope_std = _stats(per_seed_slope)
    cont_pct_mean, cont_pct_std = _stats(per_seed_cont_pct)
    cont_delta_mean, cont_delta_std = _stats(per_seed_cont_delta)
    baimu_count_mean, baimu_count_std = _stats(per_seed_baimu_count)
    baimu_area_mean, baimu_area_std = _stats(per_seed_baimu_area)
    reward_mean, reward_std = _stats(per_seed_reward)

    summary = {
        'region': 'Neijiang Dongxing',
        'mode': args.mode,
        'lambda_rank': LAMBDA_RANK,
        'n_seeds': len(seeds),
        'eval_episodes_per_seed': args.eval_episodes,
        'per_seed': {str(s): all_results[s] for s in seeds},
        'cross_seed': {
            'slope_pct_mean': slope_mean,
            'slope_pct_std': slope_std,
            'cont_pct_mean': cont_pct_mean,
            'cont_pct_std': cont_pct_std,
            'cont_raw_delta_mean': cont_delta_mean,
            'cont_raw_delta_std': cont_delta_std,
            'baimu_count_delta_mean': baimu_count_mean,
            'baimu_count_delta_std': baimu_count_std,
            'baimu_area_delta_ha_mean': baimu_area_mean,
            'baimu_area_delta_ha_std': baimu_area_std,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
        },
        'bishan_reference_slope': -1.289,
        'bishan_reference_slope_std': 0.079,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved to %s", out_path)

    print("\n" + "=" * 90)
    print(f"NEIJIANG CROSS-REGION MPC EVAL  [mode={args.mode}, lambda={LAMBDA_RANK}]")
    print("-" * 90)
    print(f"{'Seed':>4}  {'Slope%':>8}  {'ContDelta':>10}  {'Cont%':>7}  "
          f"{'BaimuCt':>8}  {'BaimuAreaHa':>12}")
    for s in seeds:
        r = all_results[s]
        print(f"{s:>4}  {np.mean(r['slopes']):>8.4f}  "
              f"{np.mean(r['cont_deltas']):>+10.4f}  "
              f"{np.mean(r['cont_pcts']):>+7.2f}  "
              f"{np.mean(r['baimu_count_deltas']):>+8.1f}  "
              f"{np.mean(r['baimu_area_deltas_ha']):>+12.2f}")
    print("-" * 90)
    print(f"Slope:      {slope_mean:+.4f}% ± {slope_std:.4f}")
    print(f"Cont Delta: {cont_delta_mean:+.4f}  ({cont_pct_mean:+.2f}% ± {cont_pct_std:.2f})")
    print(f"Baimu #:    {baimu_count_mean:+.2f}  ± {baimu_count_std:.2f}")
    print(f"Baimu ha:   {baimu_area_mean:+.2f} ha  ± {baimu_area_std:.2f}")
    print(f"Bishan ref: -1.289% ± 0.079")


if __name__ == "__main__":
    main()
