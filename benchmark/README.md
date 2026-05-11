# Farmland Consolidation Synthetic Benchmark

Synthetic GeoPackage + block JSON data for reproducing the farmland-consolidation method from Paper 9 without access to the sensitive Third National Land Survey (DLTB) data.

## Quick start

```bash
pip install -r requirements.txt
python -m generator.generate --preset presets/plain_small_cons.yaml --seed 0 --out data/plain_small_cons_seed0
python -c "from synthetic_env_loader import make_synthetic_env; env = make_synthetic_env('data/plain_small_cons_seed0'); env.reset(seed=0); print(env.n_blocks)"
```

## Presets

Seven presets span terrain × size × fragmentation axes; `bishan_clone` and `neijiang_clone` are anchored to match real-data baseline MPC profiles within ±50% magnitude.

## Licensing

Code: MIT (repo root). Data: CC-BY 4.0 (`LICENSE-DATA`).
