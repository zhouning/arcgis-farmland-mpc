"""Benchmark-only PPO policy with logits stabilization.

Subclasses Paper 1-4 production `ParcelScoringPolicy` (D:/test/parcel_scoring_policy.py)
to add:
1. nan_to_num on obs (guards against env producing NaN features)
2. nan_to_num + clamp on logits (guards against NaN from weight corruption)
3. Disables validate_args on the underlying Categorical distribution so that
   transient float32 precision issues in softmax don't crash training

Without these guards, training on synthetic environments with 2600+ block
action spaces produces NaN logits within the first 1000 gradient steps,
which crash in MaskableCategorical's Simplex() check.

We don't ship this fix back into D:/test/parcel_scoring_policy.py because
Paper 4 production trains on real Bishan data and has never hit this failure
mode at 500k timesteps.
"""
from __future__ import annotations

import sys

import torch as th

sys.path.insert(0, "D:/test")
from parcel_scoring_policy import ParcelScoringPolicy


LOGIT_CLAMP = 50.0


class StableParcelScoringPolicy(ParcelScoringPolicy):
    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        th.distributions.Categorical.set_default_validate_args(False)

    def _compute_logits(self, obs: th.Tensor) -> th.Tensor:
        obs = th.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        logits = super()._compute_logits(obs)
        return th.nan_to_num(logits, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
