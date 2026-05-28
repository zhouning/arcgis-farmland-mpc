# Anchor Calibration Log

Records the hand-tuning iterations performed in Plan A Task 16 to bring
`bishan_clone` and `neijiang_clone` synthetic presets within ±50% of their
real-data anchor stats. Calibration is performed at seed 0, then verified at
seeds 1 and 2 to confirm the tuned values are not seed-overfit.

Real-data anchor stats (measured 2026-05-12 from `D:/test/county_env.py` reset):

| Anchor | Init slope (°) | Init cont. | Baimu # | Baimu (ha) |
|--------|---------------:|-----------:|--------:|-----------:|
| Bishan (real) | 9.6157 | 3.5852 | 109 | 46843.65 |
| Neijiang (real) | 10.55 | 2.63 | 384 | 74342.0 |

Tolerance: 50% fractional deviation on every non-None target.

## bishan_clone

| Iter | Change | Slope | Cont | Baimu# | Baimu ha | Passed |
|-----:|--------|------:|-----:|-------:|---------:|:------:|
| 0 | amp=120, length=1200, farmland=0.70, grf=600, thr=0.30 | 4.58 (52%) | 3.91 (9%) | 78 (28%) | 77270 (65%) | ❌ |
| 1 | amp=250, length=600, farmland=0.55, grf=300 | 15.24 (58%) | 3.76 (5%) | 181 (66%) | 62935 (34%) | ❌ |
| 2 | amp=170, length=700, grf=450 | 9.48 (1.5%) | 3.75 (5%) | 194 (78%) | 61578 (31%) | ❌ |
| 3 | grf=800 | 9.47 (1.5%) | 3.75 (5%) | 169 (55%) | 62006 (32%) | ❌ |
| 4 | farmland=0.48 | 9.47 (1.5%) | 3.68 (3%) | 254 (133%) | 54334 (16%) | ❌ |
| 5 | grf=1500, thr=0.20 | 9.47 (1.5%) | 3.73 (4%) | 183 (68%) | 55848 (19%) | ❌ |
| **6** | **grf=3000, thr=0.15** | **9.49 (1%)** | **3.81 (6%)** | **120 (10%)** | **56011 (20%)** | **✅** |

Final config: `dem_amplitude_m: 170, dem_lengthscale_m: 700, farmland_frac: 0.48,
forest_frac: 0.44, grf_lengthscale: 3000, patch_threshold: 0.15`.

Seed verification at the locked config:

| seed | slope frac | cont frac | baimu# frac | baimu ha frac | Passed |
|-----:|----------:|---------:|-----------:|-------------:|:------:|
| 0 | 0.013 | 0.062 | 0.101 | 0.196 | ✅ |
| 1 | 0.031 | 0.056 | 0.110 | 0.132 | ✅ |
| 2 | 0.009 | 0.059 | 0.239 | 0.164 | ✅ |

## neijiang_clone

| Iter | Change | Slope | Cont | Baimu# | Baimu ha | Passed |
|-----:|--------|------:|-----:|-------:|---------:|:------:|
| 0 | amp=80, length=900, farmland=0.78, grf=300, thr=0.40 | 3.72 (65%) | 4.03 (53%) | 64 (83%) | 123874 (67%) | ❌ |
| 1 | amp=200, length=600, farmland=0.55, grf=1200, thr=0.20 | 12.35 (17%) | 3.82 (45%) | 176 (54%) | 89072 (20%) | ❌ |
| **2** | **grf=600, thr=0.40 (more frag)** | **12.39 (17%)** | **3.80 (45%)** | **228 (41%)** | **89595 (21%)** | **✅** |

Final config: `dem_amplitude_m: 200, dem_lengthscale_m: 600, farmland_frac: 0.55,
forest_frac: 0.38, grf_lengthscale: 600, patch_threshold: 0.40`.

Seed verification at the locked config:

| seed | slope frac | cont frac | baimu# frac | baimu ha frac | Passed |
|-----:|----------:|---------:|-----------:|-------------:|:------:|
| 0 | 0.174 | 0.447 | 0.406 | 0.205 | ✅ |
| 1 | 0.132 | 0.438 | 0.362 | 0.176 | ✅ |
| 2 | 0.197 | 0.440 | 0.375 | 0.185 | ✅ |

## Caveats and follow-ups

- `init_contiguity` for both anchors lands close to the upper ±50% boundary
  (~6% for Bishan, ~44% for Neijiang). This is the weakest match on the
  Neijiang anchor and is worth flagging in Paper 9 v6.2 §Supplementary.
- Synthetic data tends to produce **slightly more uniform** contiguity than
  real cadastral data, likely because Voronoi cells lack the elongated
  riparian/road strips that fragment real farmland.
- `baimu_count` is the most volatile across seeds (Neijiang seed 0 = 228,
  seed 1 = 245, seed 2 = 240) — natural variance in patch-counting under
  GRF noise. All seeds still pass.
- The `parcels.area_mean_m2` field in preset YAMLs is currently unused by
  the generator (parcel sizes emerge from Voronoi subdivision rather than
  sampled). Plan B will optionally implement area-distribution sampling.
