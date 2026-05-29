"""Tool 4: Model-Predictive Control planning.

Uses core.blocks_env.make_env(prepared_dir=...) to instantiate a
region-agnostic CountyLevelEnv, then rolls out top-K candidate blocks
under the ONNX ensemble for H steps and commits the block whose
rollout accumulates the highest reward.

Outputs:
    land_use.npy  -- flattened land-use vector per step
    summary.json  -- aggregate slope / baimu / contiguity metrics
    mpc_run.log   -- per-episode / per-step progress

Optionally writes an optimized DLTB feature class with OPT_DLBM /
OPT_DLMC / CHG_FLAG / ORIG_DLBM fields, mapped back via BSM.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MPC core: lifted from mpc_planner.py, torch dependency removed.
# ---------------------------------------------------------------------------

def _compute_slope_signal(cur_gf, next_gf):
    # global_features[4] = (initial_slope - cur_slope) / initial_slope
    return next_gf[:, 4] - cur_gf[:, 4]


def _greedy_1step_actions(ensemble, cur_bf, cur_gf, valid_actions, n_sample, rng):
    k = cur_bf.shape[0]
    if len(valid_actions) <= n_sample:
        sample_actions = valid_actions
    else:
        sample_actions = rng.choice(valid_actions, n_sample, replace=False)
    n_s = len(sample_actions)

    bf_exp = np.repeat(cur_bf, n_s, axis=0)
    gf_exp = np.repeat(cur_gf, n_s, axis=0)
    a_exp = np.tile(sample_actions, k)
    _, _, r_exp, _ = ensemble.batch_predict(bf_exp, gf_exp, a_exp)
    r_matrix = r_exp.reshape(k, n_s)
    best_local = r_matrix.argmax(axis=1)
    return sample_actions[best_local]


def mpc_select_action(ensemble, block_features, global_features, action_mask,
                      horizon=5, top_k=50, gamma=0.99, n_rollouts=1,
                      continuation="random", greedy_sample=50,
                      scoring="reward", rng=None):
    """Pick the next action by simulating top-K candidates H steps forward."""
    rng = rng or np.random.default_rng()
    valid_actions = np.where(action_mask)[0]
    if len(valid_actions) == 0:
        return 0, {}

    # Stage 1: score every valid action 1-step
    n_valid = len(valid_actions)
    bf_batch = np.tile(block_features[np.newaxis], (n_valid, 1, 1))
    gf_batch = np.tile(global_features[np.newaxis], (n_valid, 1))
    next_bf, next_gf, r1, _ = ensemble.batch_predict(bf_batch, gf_batch, valid_actions)

    score1 = _compute_slope_signal(gf_batch, next_gf) if scoring == "slope" else r1

    k = min(top_k, n_valid)
    top_idx = np.argsort(score1)[-k:]
    candidates = valid_actions[top_idx]
    cand_cumrew = score1[top_idx].copy().astype(np.float64)
    init_bf = next_bf[top_idx]
    init_gf = next_gf[top_idx]

    # Stage 2: H-1 step rollout(s), mean over n_rollouts
    rollout_rewards = np.zeros(k, dtype=np.float64)
    for _ in range(n_rollouts):
        cur_bf = init_bf.copy()
        cur_gf = init_gf.copy()
        prev_gf = init_gf.copy()
        discount = gamma
        for _step in range(1, horizon):
            if continuation == "greedy":
                actions = _greedy_1step_actions(
                    ensemble, cur_bf, cur_gf, valid_actions, greedy_sample, rng)
            else:
                actions = rng.choice(valid_actions, size=k)
            nb, ng, r_step, _ = ensemble.batch_predict(cur_bf, cur_gf, actions)
            step_score = _compute_slope_signal(prev_gf, ng) if scoring == "slope" else r_step
            rollout_rewards += discount * step_score
            discount *= gamma
            prev_gf = cur_gf.copy()
            cur_bf = nb
            cur_gf = ng
    cand_cumrew += rollout_rewards / n_rollouts

    best = int(np.argmax(cand_cumrew))
    chosen = int(candidates[best])
    info = {
        "n_valid": int(n_valid), "n_candidates": int(k),
        "best_cumrew": float(cand_cumrew[best]),
        "mean_cumrew": float(cand_cumrew.mean()),
        "horizon": horizon, "continuation": continuation, "scoring": scoring,
    }
    return chosen, info


# ---------------------------------------------------------------------------
# Episode runner (talks to the Gymnasium env)
# ---------------------------------------------------------------------------

def _run_episode(env, ensemble, horizon, top_k, gamma, continuation,
                 scoring, seed, progress_cb=None):
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    step_times = []
    last_info = {}

    for step in range(env.max_steps):
        bf = env._get_block_features()
        gf = env._get_global_features()
        mask = env.action_masks()

        t0 = time.time()
        action, mpc_info = mpc_select_action(
            ensemble, bf, gf, mask,
            horizon=horizon, top_k=top_k, gamma=gamma,
            n_rollouts=1, continuation=continuation,
            scoring=scoring, rng=rng,
        )
        step_times.append(time.time() - t0)

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        last_info = info

        if progress_cb is not None:
            progress_cb(step + 1, env.max_steps, info, step_times[-1])

        if terminated or truncated:
            break

    last_info["total_reward"] = total_reward
    last_info["mean_step_time"] = float(np.mean(step_times)) if step_times else 0.0
    last_info["total_time"] = float(np.sum(step_times))
    last_info["steps_run"] = len(step_times)
    return last_info


# ---------------------------------------------------------------------------
# Entry point called by the .pyt
# ---------------------------------------------------------------------------

def run(ensemble_dir, out_dir, horizon=5, top_k=50, gamma=0.99,
        threads=0, n_episodes=1, continuation="random", scoring="reward",
        max_steps=None, seed_offset=0,
        prepared_dir=None, proj_crs=None,
        output_fc=None, input_dltb_fc=None,
        farm_dlbm="011", forest_dlbm="031",
        slope_weight=None, cont_weight=None,
        baimu_weight=None, baimu_bonus=None,
        baimu_area_penalty=None,
        messages=None):
    """MPC planning loop (v0.3).

    v0.3 adds optional reward-weight overrides. When any of slope_weight /
    cont_weight / baimu_weight / baimu_bonus is non-None, the env is built
    with that weight (overriding the Paper 9 v6 default). **Important**:
    env reward is what Tool 3's ensemble was trained to predict. Changing
    weights at Tool 4 time means the ensemble's reward head is slightly
    mis-calibrated for the new objective. The MPC still runs -- because
    Paper 9 picks actions by integrating reward over H steps, and the
    ensemble's reward scale is approximately preserved under linear re-
    weighting -- but the result is strictly "in-distribution for Tool 3"
    only at the defaults. For the cleanest results, retrain Tool 3 after
    changing weights.

    Parameters (new vs v0.2)
    ------------------------
    slope_weight : float or None
        Override env.slope_weight (Paper 9 default 4000.0).
    cont_weight : float or None
        Override env.cont_weight (default 500.0).
    baimu_weight : float or None
        Override env.baimu_weight (default 1500.0).
    baimu_bonus : float or None
        Override env.baimu_bonus (default 5.0).
    baimu_area_penalty : float or None
        Override env.baimu_area_penalty (default 2000.0; an asymmetric penalty
        the env applies whenever baimu_fang area decreases between steps,
        NOT documented in the Paper 9 v7 reward equation Eq.1). Note: this
        runtime override only affects the env.step() reward used by
        episode_return reporting; it has NO effect on stage-1/stage-2
        candidate ranking because those use the ONNX ensemble's reward head,
        which was frozen at training time under whatever value of
        baimu_area_penalty was in effect when Tool 2 sampled and Tool 3
        trained. To actually steer planning by a different penalty, re-run
        Tools 2 and 3 with the desired value.

    See v0.2 docstring for the rest.
    """

    def _say(msg, level="info"):
        if messages is not None:
            getattr(messages, "addMessage" if level == "info"
                    else "addWarningMessage")(msg)
        logger.info(msg)
        print(msg, flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "mpc_run.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    try:
        # Ensure the toolbox dir (so 'core/...' is importable) is on
        # sys.path. The .pyt also does this; we repeat here in case
        # core.mpc_plan is invoked from another script.
        toolbox_dir = str(Path(__file__).resolve().parent.parent)
        if toolbox_dir not in sys.path:
            sys.path.insert(0, toolbox_dir)
        core_dir = str(Path(__file__).resolve().parent)
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)

        _say(f"[MPC] horizon={horizon} top_k={top_k} gamma={gamma} "
             f"continuation={continuation} scoring={scoring} "
             f"episodes={n_episodes} threads={threads}")
        _say(f"[MPC] ensemble_dir = {ensemble_dir}")
        _say(f"[MPC] prepared_dir = {prepared_dir}")
        if output_fc:
            _say(f"[MPC] output_fc   = {output_fc}")
            if not input_dltb_fc:
                raise ValueError("output_fc requires input_dltb_fc")

        # Load ensemble
        try:
            from farmland_mpc.ensemble_runner import EnsembleOrtRunner
        except ImportError:
            from core.ensemble_runner import EnsembleOrtRunner
        ensemble = EnsembleOrtRunner(ensemble_dir, n_threads=threads)
        _say(f"[MPC] Loaded {ensemble.n_members} ONNX members: "
             + ", ".join(os.path.basename(p) for p in ensemble.paths))

        # Build env (v0.2: region-agnostic; v0.3: optional reward weights)
        _say("[MPC] Building CountyLevelEnv via blocks_env.make_env "
             "(~30-70s data load)...")
        t_env = time.time()
        try:
            from farmland_mpc.blocks_env import make_env
        except ImportError:
            from core.blocks_env import make_env

        env_kwargs = {}
        reward_overrides = {}
        for name, val in (
            ("slope_weight", slope_weight),
            ("cont_weight", cont_weight),
            ("baimu_weight", baimu_weight),
            ("baimu_bonus", baimu_bonus),
            ("baimu_area_penalty", baimu_area_penalty),
        ):
            if val is not None:
                env_kwargs[name] = float(val)
                reward_overrides[name] = float(val)
        if reward_overrides:
            _say(
                "[MPC] WARNING: overriding env reward weights "
                + ", ".join(f"{k}={v}" for k, v in reward_overrides.items())
                + ". "
                "Effect: Tool 3's ensemble.reward_head predicts reward "
                "under the ORIGINAL training weights, not the overridden "
                "ones. When scoring='reward', MPC ranks candidates by the "
                "ensemble prediction, so the override has no effect on "
                "block selection -- it only changes the env.step() reward "
                "that episode_return etc. report. For the override to "
                "actually steer planning, retrain Tool 3 with these "
                "weights first. (When scoring='slope', MPC uses the env's "
                "slope delta directly, so the weight has no effect at "
                "all.)",
                level="warn",
            )

        env = make_env(prepared_dir=prepared_dir, proj_crs=proj_crs,
                       **env_kwargs)
        if max_steps is not None and max_steps > 0:
            env.max_steps = int(max_steps)
            _say(f"[MPC] env.max_steps capped to {env.max_steps} for smoke test")
        _say(f"[MPC] env built in {time.time() - t_env:.1f}s; "
             f"n_blocks={env.n_blocks}, max_steps={env.max_steps}, "
             f"n_parcels={env.n_parcels}")

        # Verify ensemble was trained for the same n_blocks
        ensemble.assert_compatible(env.n_blocks)

        # Progress callback for arcpy
        try:
            import arcpy
            arcpy.SetProgressor("step", "MPC planning...", 0,
                                env.max_steps * n_episodes, 1)
            arcpy_available = True
        except Exception:
            arcpy_available = False

        progress_total = {"done": 0}

        def _progress(step_idx, total, info, step_time):
            progress_total["done"] += 1
            if arcpy_available:
                arcpy.SetProgressorLabel(
                    f"ep step {step_idx}/{total} | "
                    f"slope {info.get('slope_change_pct', 0):+.3f}% | "
                    f"step {step_time:.2f}s"
                )
                arcpy.SetProgressorPosition(progress_total["done"])
            if step_idx % 10 == 0 or step_idx == total:
                _say(f"    step {step_idx:3d}/{total} "
                     f"slope={info.get('slope_change_pct', 0):+.4f}% "
                     f"cont={info.get('cont_change', 0):+.4f} "
                     f"baimu_ha={info.get('baimu_area_change_ha', 0):+.1f} "
                     f"mpc_step={step_time:.2f}s")

        # Run episodes
        results = []
        for ep in range(n_episodes):
            seed = seed_offset + ep
            _say(f"\n[MPC] === Episode {ep + 1}/{n_episodes} (seed={seed}) ===")
            t0 = time.time()
            info = _run_episode(env, ensemble, horizon, top_k, gamma,
                                continuation, scoring, seed, _progress)
            ep_time = time.time() - t0
            results.append({
                "episode": ep, "seed": seed,
                "slope_change_pct": float(info.get("slope_change_pct", 0.0)),
                "cont_change": float(info.get("cont_change", 0.0)),
                "baimu_count_change": int(info.get("baimu_count_change", 0)),
                "baimu_area_change_ha": float(info.get("baimu_area_change_ha", 0.0)),
                "total_reward": float(info.get("total_reward", 0.0)),
                "steps_run": int(info.get("steps_run", 0)),
                "mean_step_time_s": float(info.get("mean_step_time", 0.0)),
                "total_time_s": float(ep_time),
            })
            _say(f"[MPC] ep {ep}: slope={results[-1]['slope_change_pct']:+.4f}% "
                 f"cont={results[-1]['cont_change']:+.4f} "
                 f"baimu_ha={results[-1]['baimu_area_change_ha']:+.2f} "
                 f"time={ep_time:.1f}s")

            np.save(out_dir / "mpc_land_use.npy", env.land_use.astype(np.int8))

        # Aggregate
        slopes = [r["slope_change_pct"] for r in results]
        conts  = [r["cont_change"] for r in results]
        baimu  = [r["baimu_area_change_ha"] for r in results]
        summary = {
            "config": {
                "horizon": horizon, "top_k": top_k, "gamma": gamma,
                "continuation": continuation, "scoring": scoring,
                "n_episodes": n_episodes, "threads": threads,
                "max_steps": env.max_steps, "n_blocks": int(env.n_blocks),
                "n_parcels": int(env.n_parcels),
                "prepared_dir": getattr(env, "_prepared_dir", None),
                "proj_crs": proj_crs,
                "reward_overrides": reward_overrides,
            },
            "ensemble": {
                "n_members": ensemble.n_members,
                "paths": [os.path.basename(p) for p in ensemble.paths],
            },
            "results": results,
            "aggregate": {
                "slope_pct_mean": float(np.mean(slopes)),
                "slope_pct_std":  float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
                "cont_mean":      float(np.mean(conts)),
                "baimu_ha_mean":  float(np.mean(baimu)),
            },
        }

        # Optional: write optimized DLTB feature class
        if output_fc:
            _say(f"\n[MPC] Writing optimized DLTB to {output_fc} ...")
            try:
                from farmland_mpc.shapefile_io import write_optimized_dltb
            except ImportError:
                from core.shapefile_io import write_optimized_dltb
            shp_stats = write_optimized_dltb(
                input_fc=input_dltb_fc, output_fc=output_fc, env=env,
                farm_dlbm=farm_dlbm, forest_dlbm=forest_dlbm,
                messages=messages,
            )
            summary["shapefile_output"] = shp_stats

        with open(out_dir / "mpc_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        _say("")
        _say("[MPC] ==== Summary ====")
        _say(f"  slope: {summary['aggregate']['slope_pct_mean']:+.4f}% "
             f"+- {summary['aggregate']['slope_pct_std']:.4f}")
        _say(f"  cont : {summary['aggregate']['cont_mean']:+.4f}")
        _say(f"  baimu: {summary['aggregate']['baimu_ha_mean']:+.2f} ha")
        _say(f"  outputs written to {out_dir}")
        if output_fc:
            _say(f"  optimized feature class: {output_fc}")

        if arcpy_available:
            arcpy.ResetProgressor()

        return summary
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
