# Reproducing the paper from raw cadastral data + Copernicus DEM

This guide walks through the full pipeline used by an external user starting
from a Third National Land Survey (DLTB) shapefile and a public Copernicus
DEM tile, ending at the paper's 5-seed cross-ensemble result. It covers
both real counties (Bishan + Neijiang Dongxing) and is the canonical
end-to-end reproduction path validated on macOS Apple Silicon (2026-05-29).

If you only want to consume an already-prepared dataset, skip to
"Re-running training and evaluation" below.

## Wall-clock budget

On a desktop M-series Mac (14 physical cores, 128 GB RAM):

| Stage | Bishan | Neijiang Dongxing |
|---|---|---|
| `prepare` (slope + blocks + sanity) | ~3 min | ~5 min |
| `sample` (transitions + pairwise) | ~16 min | ~16 min |
| `train` (one 3-member ensemble) | ~40-45 min | ~58-60 min |
| `train` × 5 ensembles, 3-way parallel | ~80 min | ~120 min |
| 5-seed greedy MPC eval (Paper §5 protocol) | ~10 min | ~15 min |
| **Total per county (5-seed reproduction)** | **~2 h** | **~2.5 h** |

A bishan-scale county (~50k parcels / 2.6k blocks) fits in a 16 GB Docker
limit; a Neijiang-scale county (~76k / 3.7k blocks) needs ≥48 GB on greedy
MPC paths. Native macOS handles either at any scale.

## Prerequisites

```bash
# Clone the repo
git clone git@github.com:zhouning/arcgis-farmland-mpc.git
cd arcgis-farmland-mpc

# Conda env with pinned dependencies (Python 3.11)
conda env create -f environment.yml
conda activate farmland-mpc

# Verify the package is importable + CLI is on PATH
farmland-mpc version
```

## Inputs

You supply two things per county:

1. **DLTB cadastral shapefile** — Third National Land Survey export,
   restricted under PRC data-governance rules. The schema fields the
   pipeline reads are `BSM` (parcel ID), `DLBM` (land-use code),
   `QSDWDM` (township ownership code), and `geometry`. CRS can be
   anything pyproj recognises (Bishan example: EPSG:4610; Dongxing:
   EPSG:2359; both work).

2. **Copernicus DEM tile(s)** — public, downloadable from
   `https://copernicus-dem-30m.s3.amazonaws.com/`. The pipeline expects
   a single GeoTIFF in geographic CRS (EPSG:4326, 1-arcsecond ~30 m
   resolution). Multi-tile counties: merge tiles first with
   `rasterio.merge.merge` (see Bishan single-tile vs Dongxing two-tile
   examples in `prepare/dem_slope/README.md`).

The pipeline now keeps the DEM in its native geographic CRS (since
commit `dc3a01c`) and uses true east-west / north-south metres-per-pixel
from `pyproj.Geod` — this preserves the fact that 1-arcsecond pixels are
non-square at non-equatorial latitudes (~27 m E-W × ~31 m N-S at lat 29.5°).
Reprojecting the DEM to UTM beforehand systematically under-estimates
slope by ~1.25° at our latitude. If you must feed a UTM DEM, pass
`--slope-method horn_projected` to keep the legacy behaviour; the
default `--slope-method auto` picks the right algorithm for the input.

## Step 1 — `prepare` (per-parcel slope + block construction)

Bishan example:

```bash
farmland-mpc prepare \
    --dltb /path/to/bishan_DLTB.shp \
    --dem  /path/to/bishan_dem_4326.tif \
    --out  /path/to/bishan_run/prepared \
    --crs  EPSG:32648 \
    --slope-method auto
```

This emits, under `prepared/`:

- `dem_slope_analysis/output/DLTB_with_slope.shp` — per-parcel slope
- `results_real/blocks/township_<code>/` — block features per township
- `townships.json` — code → label mapping
- `prepare_data_summary.json` — provenance: input paths, CRS, parcel
  counts, **resolved `slope_method`**, initial slope/contiguity/baimu
  fang count + area (these go into Paper Table 1)

### Verifying alignment with Paper Table 1

After `prepare`, check `phase_c_sanity` in `prepare_data_summary.json`:

| County | Initial slope expected (paper) | Initial cont expected | Baimu count expected |
|---|---|---|---|
| Bishan | 9.62° | 3.59 | 109 |
| Dongxing | 10.55° | 2.63 | 384 |

Tolerance: ±0.1° on slope, ±0.02 on contiguity, ±5 on baimu count
(these absorb tiny differences in DLTB row ordering and block
construction). If your numbers are off by >1°, something is wrong with
the DEM (most commonly: DEM was reprojected to UTM before being passed
in, in which case use `--slope-method horn_projected` knowingly OR
re-run with the original geographic-CRS DEM).

## Step 2 — `sample` (transitions + pairwise)

```bash
farmland-mpc sample \
    --prepared-dir /path/to/bishan_run/prepared \
    --n-episodes 60 \
    --n-states 1000 \
    --n-actions 50 \
    --seed 0
```

Outputs:
- `prepared/tool2/transitions.npz` — 6,000 (s, a, s', r) tuples for
  the MSE backbone training
- `prepared/tool2/pairwise.npz` — 50,000 (s, a, r) ground-truth
  evaluations (1,000 states × 50 actions) for the contrastive loss
- `prepared/tool2/sample_transitions_summary.json` — provenance,
  including pairwise reward statistics

Verifying: `pairwise.reward_std_median` should fall around 0.6 for Bishan
and 0.2 for Dongxing (the values reflect each county's reward landscape;
they're a sanity check, not a tunable target).

## Step 3 — `train` (5 independent contrastive ensembles)

The paper §5 protocol is **5 independently trained ensembles, 3 members
each, with `lambda_rank=5.0`, `margin=0.1`, 30 epochs**. Each ensemble
gets its own subdir under `prepared/`:

```bash
# Run 5 ensembles in parallel (3 at a time on a 14-core machine)
for SEED in 0 1 2 3 4; do
  farmland-mpc train \
      --prepared-dir /path/to/bishan_run/prepared \
      --n-members 3 --epochs 30 --patience 8 \
      --lambda-rank 5.0 --margin 0.1 \
      --seed-base $SEED \
      --torch-threads 4 \
      --out-subdir ensemble_seed$SEED &

  # cap at 3 in flight at once (14-core machine)
  if (( SEED % 3 == 2 )); then wait; fi
done
wait
```

Each ensemble takes 40-60 minutes on a 14-core M-series Mac when 3 are
running concurrently with `--torch-threads 4` per process. Outputs:
`prepared/ensemble_seedN/` with `ensemble_member{0,1,2}.{onnx,pt}` and
`train_summary.json`.

Verifying: each `train_summary.json` should report
`final_ranking_acc` ≥ 0.85 (paper target 0.855) for all three members.
Mean across 30 members on the validated reference run: 0.882 ± 0.019.

If a single ensemble takes much longer than 60 minutes, you're running
out of RAM and PyTorch is swapping; reduce `--torch-threads` from 4 to 2
or drop concurrency to 2 ensembles in flight.

## Step 4 — Paper §5 5-seed cross-ensemble evaluation

```bash
python -m farmland_mpc.tests.eval_5seed_paper \
    --prepared-dir /path/to/bishan_run/prepared \
    --out-json /path/to/bishan_run/5seed_paper_baseline.json \
    --region "Bishan" \
    --n-seeds 5 \
    --n-episodes-per-seed 1 \
    --continuation greedy
```

This runs each of the 5 ensembles for one deterministic 100-step MPC
episode (greedy continuation, H=5, K=50, scoring=reward) and aggregates
into a JSON whose `cross_seed` block matches the research-side
`5seed_multiobj_results_baseline.json` schema verbatim, so you can diff
your reproduction against the public artifacts in
`neijiang_cross_region/`.

## Reference numbers from the validated 2026-05-29 macOS reproduction

These are the cross-seed (n=5 ensembles × 1 ep each) numbers obtained
from the package end-to-end, with `--slope-method auto` (the new
geographic-CRS algorithm). They are the right comparison target for
fresh reproductions; the paper §5 numbers (next column) sit slightly
differently because the paper used a different prepare path
(2,600-block research lab env vs the package's 2,640-block prepare),
the same ensemble training data was generated in 2026-05 from raw 三调
data, etc. — the **scientific findings reproduce; the bit-level
numerics diverge for documented reasons**.

### Bishan

| Metric | This package (n=5) | Paper §5 Table 4 | Paper §Methods-validate |
|---|---|---|---|
| slope %     | −1.7384 ± 0.0243 | −1.289 ± 0.079 | −2.0006 |
| Δcont       | +0.01442 ± 0.00126 | +0.0160 ± 0.0016 | +0.01275 |
| Δbaimu #    | +3.40 ± 1.36 | +3.4 ± 1.0 | +8 |
| Δbaimu (ha) | −478.64 ± 35.00 | −312 ± 34 | −473.30 |

Per-seed slope (from this reference run): −1.753 / −1.764 / −1.726 /
−1.696 / −1.752. Cross-seed std (0.024 pp) is an order of magnitude
smaller than paper §5 (0.079 pp), reflecting the fact that under our
prepare pipeline ensembles converge more tightly. Bishan's slope
improvement comes in **larger than paper §5** because the corrected
prepare exposes more accurate (and steeper) initial slopes, giving the
agent more headroom; baimu_ha matches **§Methods-validate** rather than
§5 Table 4 essentially exactly (−478 vs −473 ha).

### Neijiang Dongxing

| Metric | This package (n=5) | Research-side artefacts | Paper §5 |
|---|---|---|---|
| slope %     | −0.5741 ± 0.0233 | −0.5013 ± 0.0242 | −0.50 ± 0.02 |
| Δcont       | +0.03652 ± 0.00219 | +0.03370 ± 0.00114 | +0.034 ± 0.001 |
| Δbaimu #    | +0.40 ± 2.06 | −11.12 ± 3.36 | (not reported) |
| Δbaimu (ha) | +30.17 ± 118.13 | +267.09 ± 37.96 | +267 ± 38 |

Per-seed Δbaimu (ha): +71.92 / −23.60 / +223.70 / +14.28 / −135.44.
**The qualitative cross-county finding reproduces**: Bishan loses
qualifying area (−478 ha) while Dongxing gains it (+30 ha mean,
3 of 5 seeds positive), opposite signs. The magnitude (+30 vs paper
+267) is smaller because our 5 ensembles distribute differently than
the research-side 5 ensembles — both pipelines learn paper-consistent
policies, ours converges on a "preserve count, mildly grow area" mode,
the research side's on a "shed count to grow area" mode.

## Re-running training and evaluation from a pre-prepared `prepared/`

If you already have a `prepared/` directory (for example from someone
else's data-prep stage, or from a previous run), skip Step 1. Steps 2-4
are unchanged.

If you have an externally produced `DLTB_with_slope.gpkg` with a
`slope_mean` column already populated (e.g. from an ArcGIS-Pro slope
job, or from the research-side `prepare/dem_slope/*.py` scripts), feed
it as the DLTB input and use `--slope-method from_field` to skip the
DEM step entirely:

```bash
farmland-mpc prepare \
    --dltb /path/to/DLTB_with_slope.gpkg \
    --dem  /path/to/anything.tif \  # ignored in from_field mode
    --out  /path/to/run/prepared \
    --slope-method from_field \
    --slope-field slope_mean
```

## Troubleshooting

**"All parcel slopes are NaN"**: DEM doesn't cover the DLTB extent.
Re-check tile coverage; for Bishan you need N29-E106, for Dongxing you
need N29-E104 + N29-E105 merged.

**Training runs out of memory**: drop `--torch-threads` from 4 to 2,
or lower in-flight concurrency to 2 ensembles. A single ensemble with
`--torch-threads 14` peaks at ~16 GB on Bishan, ~22 GB on Dongxing.

**`farmland-mpc plan` exits with code 137 mid-episode (no per-step log)**:
OOM kill. Either run native (no Docker memory limit) or raise
`docker/docker-compose.yml` `batch.memory` to ≥48 GB. The greedy
continuation path allocates `(top_k × greedy_sample, n_blocks, K_BLOCK)`
≈ 632 MB per call × 3 ensemble members × 2 (in/out), peaking ~4 GB.

**slope numbers off by ~1.25°**: you fed a UTM-projected DEM to a
prepare run that resolved `slope_method` to `horn_projected`. Either
re-run with the original geographic-CRS DEM (now-default
`gradient_geographic`), or accept the legacy numbers if you're trying
to reproduce a pre-`dc3a01c` run bit-for-bit.

**bibtex on supplementary reports "2 errors"**: cosmetic; the SI
document is intentionally citation-free (all citations live in the main
draft). The PDF still builds correctly.
