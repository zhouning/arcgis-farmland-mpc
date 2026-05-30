"""Natural-resources restoration environment, structurally parallel to CountyLevelEnv.

Generalises the farmland-consolidation MPC pipeline to a non-farmland decision
problem: prioritised selection of spatial planning units for ecological
restoration / abandoned-mine reclamation under a budget and multi-objective
reward. The action space is a finite categorical of size ``n_units``;
``action_masks()`` excludes already-selected units and (optionally)
budget-violating units. State is per-unit attribute features plus a global
state vector. Snapshot/restore is supported for pairwise sampling.

This env is consumed by the same ``farmland_mpc.mpc_plan`` and
``farmland_mpc.sample`` pipelines as ``CountyLevelEnv``; see
``farmland_mpc.restoration_blocks_env.make_restoration_env`` for the factory
that the package's ``--env`` CLI flag selects.

Differences from CountyLevelEnv:
  - Action is a one-time selection (selected units cannot be re-selected) —
    monotonic, single-assignment, vs CountyLevelEnv's reversible swaps.
  - Reward is the sum of per-unit risk-reduction / habitat-connectivity /
    water / cost terms read from a per-unit attribute table.
  - State features are read from an attribute CSV rather than recomputed
    from cadastral geometry; no parcel-level geometry is involved.

State conventions (so mpc_plan can consume the env unchanged):
  - block_features:  (n_units, K_UNIT=17) float32, padded with zeros if the
                     attribute table is narrower.
  - global_features: (K_GLOBAL=12) float32. First slot is fraction of budget
                     remaining; second is fraction of units already selected;
                     remaining slots are running cumulative reward components.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


K_UNIT = 17  # match CountyLevelEnv K_BLOCK so ensembles can be cross-loaded
K_GLOBAL = 12


@dataclass
class _Snapshot:
    selected: np.ndarray
    cumulative_reward_components: np.ndarray
    budget_used: float
    step_count: int


class RestorationEnv(gym.Env):
    """Region-agnostic ecological-restoration env.

    Parameters
    ----------
    units : pd.DataFrame
        Per-unit attribute table. Must contain ``unit_id`` and a
        ``candidate`` column (1 = eligible action, 0 = always masked) plus
        whichever feature/reward columns the scenario uses.
    adjacency : pd.DataFrame
        Edge list with ``source``, ``target`` columns; used for
        connectivity-bonus rewards.
    feature_cols : list[str]
        Attribute columns to use as per-unit numeric features. Excess columns
        are truncated; insufficient columns are zero-padded to ``K_UNIT``.
    reward_terms : dict[str, float]
        Linear combination weights, e.g.
        ``{'risk_reduction': 0.45, 'water_priority': 0.25,
            'connectivity': 0.20, 'cost_penalty': -0.10}``.
        Each key must be a column of ``units`` except ``connectivity``,
        which is computed dynamically as ``num adjacent already-selected
        units / max_degree``.
    budget : float
        Hard budget on cumulative ``cost_col`` of selected units.
    cost_col : str
        Column name that the budget tracks.
    max_steps : int
        Episode length. Truncates if all candidates are masked earlier.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 units: pd.DataFrame,
                 adjacency: pd.DataFrame,
                 feature_cols: list[str],
                 reward_terms: dict[str, float],
                 budget: float,
                 cost_col: str,
                 max_steps: int = 50):
        super().__init__()
        if "unit_id" not in units.columns:
            raise ValueError("units DataFrame must have a 'unit_id' column")
        units = units.sort_values("unit_id").reset_index(drop=True)
        self.units = units
        self.n_units = len(units)
        self.n_parcels = self.n_units  # alias for mpc_plan compatibility
        self.n_blocks = self.n_units   # alias for ensemble batch_predict
        self.max_steps = int(max_steps)
        self.budget = float(budget)
        self.cost_col = cost_col

        # Build per-unit feature matrix once (constant across an episode --
        # only the action mask and global state evolve).
        self.feature_cols = feature_cols
        feat = np.zeros((self.n_units, K_UNIT), dtype=np.float32)
        for j, c in enumerate(feature_cols[:K_UNIT]):
            if c not in units.columns:
                continue
            v = units[c].astype(float).to_numpy()
            # one-hot expand if dtype is object / categorical -- fallback to
            # numeric coercion that NaN-fills bad values
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            feat[:, j] = v.astype(np.float32)
        # Standardise per column to zero mean unit variance for numerical
        # stability under the same MSE training as farmland.
        for j in range(K_UNIT):
            col = feat[:, j]
            mu, sd = float(col.mean()), float(col.std())
            if sd > 1e-6:
                feat[:, j] = (col - mu) / sd
        self._unit_features = feat
        self._candidate_mask = (units.get("candidate", pd.Series([1] * self.n_units))
                                .astype(int).to_numpy().astype(bool))

        # Adjacency as scipy-style csr would be cleaner but keep numpy for
        # zero extra dependency.
        self._adj_neighbours = [[] for _ in range(self.n_units)]
        if "source" in adjacency.columns and "target" in adjacency.columns:
            for s, t in zip(adjacency["source"].astype(int),
                            adjacency["target"].astype(int)):
                if 0 <= s < self.n_units and 0 <= t < self.n_units:
                    self._adj_neighbours[s].append(int(t))
                    self._adj_neighbours[t].append(int(s))
        self._max_degree = max((len(nb) for nb in self._adj_neighbours), default=1)

        # Reward weights and column references
        self.reward_terms = dict(reward_terms)
        self._cost_arr = (units[cost_col].astype(float).to_numpy()
                          if cost_col in units.columns
                          else np.zeros(self.n_units, dtype=float))
        self._reward_cols: dict[str, np.ndarray] = {}
        for k in reward_terms:
            if k == "connectivity" or k == "cost_penalty":
                continue
            if k not in units.columns:
                # silently zero; lets a scenario specify a term that the user's
                # data does not yet expose
                self._reward_cols[k] = np.zeros(self.n_units, dtype=float)
            else:
                self._reward_cols[k] = units[k].astype(float).to_numpy()

        # Gym spaces
        obs_dim = self.n_units * K_UNIT + K_GLOBAL
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_units)

        # Episode state
        self.selected = np.zeros(self.n_units, dtype=bool)
        self.budget_used = 0.0
        self.step_count = 0
        self._cum_reward_components = np.zeros(8, dtype=np.float64)
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):  # noqa: D401
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.selected[:] = False
        self.budget_used = 0.0
        self.step_count = 0
        self._cum_reward_components[:] = 0.0
        return self._get_obs(), {
            "n_units": self.n_units,
            "n_candidates": int(self._candidate_mask.sum()),
            "budget": self.budget,
        }

    def step(self, action):
        a = int(action)
        if a < 0 or a >= self.n_units:
            raise ValueError(f"action {a} out of range [0, {self.n_units})")
        # If the action is already masked, fall back to a small negative reward
        # rather than raising, to mirror CountyLevelEnv's "wasted action" handling.
        mask = self.action_masks()
        if not mask[a]:
            self.step_count += 1
            terminated = (not mask.any()) or self.step_count >= self.max_steps
            return self._get_obs(), -1.0, terminated, False, {"wasted": True}

        cost = float(self._cost_arr[a])
        # commit selection
        self.selected[a] = True
        self.budget_used += cost
        self.step_count += 1

        # compute per-step reward as sum over weighted terms
        components = {}
        for term, w in self.reward_terms.items():
            if term == "cost_penalty":
                components[term] = w * (cost / max(self.budget, 1e-6))
            elif term == "connectivity":
                # bonus = w * fraction of neighbours that are already selected
                nbrs = self._adj_neighbours[a]
                if nbrs:
                    frac_sel = sum(1.0 for n in nbrs if self.selected[n]) / len(nbrs)
                else:
                    frac_sel = 0.0
                components[term] = w * frac_sel
            else:
                components[term] = w * float(self._reward_cols[term][a])
        # accumulate (first 8 components only -- enough for global state)
        keys = list(components.keys())
        for j, k in enumerate(keys[:len(self._cum_reward_components)]):
            self._cum_reward_components[j] += components[k]
        reward = float(sum(components.values()))

        info = {
            "components": components,
            "selected_id": a,
            "budget_used": self.budget_used,
        }
        # If post-action no candidate is selectable, terminate.
        post_mask = self.action_masks()
        terminated = (not post_mask.any()) or self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, False, info

    def action_masks(self):
        m = self._candidate_mask.copy()
        m &= ~self.selected
        if self.budget_used >= self.budget - 1e-9:
            return np.zeros(self.n_units, dtype=bool)
        # also mask units that would individually exceed remaining budget
        remaining = self.budget - self.budget_used
        m &= (self._cost_arr <= remaining + 1e-9)
        return m

    # ------------------------------------------------------------------
    # observations (mpc_plan / sample compatibility)
    # ------------------------------------------------------------------
    def _get_block_features(self):
        return self._unit_features.copy()

    def _get_global_features(self):
        gf = np.zeros(K_GLOBAL, dtype=np.float32)
        gf[0] = float(1.0 - self.budget_used / max(self.budget, 1e-6))
        gf[1] = float(self.selected.sum() / max(self.n_units, 1))
        gf[2] = float(self.step_count / max(self.max_steps, 1))
        n_components = min(len(self._cum_reward_components), K_GLOBAL - 3)
        gf[3:3 + n_components] = self._cum_reward_components[:n_components]
        return gf

    def _get_obs(self):
        return np.concatenate([
            self._unit_features.ravel(),
            self._get_global_features(),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # snapshot / restore (sample.py uses this for pairwise generation)
    # ------------------------------------------------------------------
    @property
    def land_use(self) -> np.ndarray:
        """Alias for ``selected`` so ``mpc_plan`` can dump trajectory tensors uniformly."""
        return self.selected.astype(np.int8)

    def snapshot(self):
        return _Snapshot(
            selected=self.selected.copy(),
            cumulative_reward_components=self._cum_reward_components.copy(),
            budget_used=float(self.budget_used),
            step_count=int(self.step_count),
        )

    def restore(self, snap: _Snapshot):
        self.selected = snap.selected.copy()
        self._cum_reward_components = snap.cumulative_reward_components.copy()
        self.budget_used = float(snap.budget_used)
        self.step_count = int(snap.step_count)


def make_restoration_env(prepared_dir: str | Path,
                         **kwargs) -> RestorationEnv:
    """Factory loading scenario_config.json + attributes/adjacency from a prepared/ dir.

    Same shape contract as ``farmland_mpc.blocks_env.make_env``: takes a
    prepared directory path and returns a ready-to-run env, all reward weights
    and feature columns read from on-disk JSON so the env is fully data-driven.

    The prepared/ dir is expected to contain:
      - scenario_config.json (output of restoration_prepare; reward_terms,
        budget, cost_col, max_steps, feature_cols)
      - attributes.csv      (one row per unit; numeric features + reward cols)
      - adjacency.csv       (source,target,...)
    """
    prepared_dir = Path(prepared_dir)
    cfg = json.loads((prepared_dir / "scenario_config.json").read_text())
    units = pd.read_csv(prepared_dir / "attributes.csv")
    adjacency = pd.read_csv(prepared_dir / "adjacency.csv")
    return RestorationEnv(
        units=units,
        adjacency=adjacency,
        feature_cols=cfg["feature_cols"],
        reward_terms=cfg["reward_terms"],
        budget=cfg.get("budget", cfg.get("budget_proxy", 1e9)),
        cost_col=cfg.get("cost_col", "restoration_cost_proxy"),
        max_steps=cfg.get("max_steps", 50),
    )
