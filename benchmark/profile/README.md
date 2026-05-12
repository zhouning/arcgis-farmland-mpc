# Profile measurements

These are **evidence**, not estimates, for Plan B Task 4 (Decision Gate).

## Files
- `cpu_profile.json` — local 12-thread CPU wall time for generator + 3 baselines on plain_small_cons
- `gpu_mpc_profile.json` — T4 utilisation + speedup vs CPU for the contrastive-MPC eval inner loop
- `gpu_ppo_profile.json` — T4 utilisation + steps/sec for PPO 25k-timestep training

## How to run

```bash
# Local CPU (~30 min)
python -m profile.profile_cpu

# Colab T4 (open profile/profile_gpu_mpc.py and profile/profile_gpu_ppo.py
# in a Colab notebook; the headers explain Drive mounting + path patching).
```

## How to read

A run is **GPU-justified** when:
- Throughput (steps/sec or candidates/sec) ≥ 5× CPU baseline, AND
- GPU utilisation per `nvidia-smi dmon` averaged over a 60-sec window is ≥ 50%

Otherwise stay on CPU.
