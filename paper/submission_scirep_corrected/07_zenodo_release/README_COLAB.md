# Paper 9 revision experiments (Google Colab Pro+)

This folder contains the reproducible experiments needed for the first revision.
The supplied files next to this README are the ones to upload:

- `prepared_neijiang_for_colab.zip` (~107 MB): prepared Neijiang inputs;
- `arcgis-farmland-mpc_minimal_for_colab.zip` (~0.6 MB): training code;
- `paper9_revision_experiments_scripts.zip`: notebook and runner scripts.

Upload these three ZIP files to the private Drive folder `MyDrive/paper9/`.
The notebook extracts the data and code into the temporary Colab VM and writes
results under `MyDrive/paper9/revision_runs/`.

The local source directories used to create them are:

```text
D:\test\_publish\arcgis-farmland-mpc\runs\scirep_extra\prepared_neijiang
D:\test\_publish\arcgis-farmland-mpc
```

Do not upload the full repository `runs` directory.

## Experiments

1. **Neijiang margin ablation**: `m={0.05,0.10,0.20}`; each margin follows
   the same state-level sampling protocol and has a fresh three-member
   ensemble. The JSON records the exact sampled-state counts and seed.
2. **p_A recalibration**: `baimu_area_penalty={1000,2000,4000,8000}`; Tool 2
   is re-sampled and Tool 3 is re-trained for every value.
3. **Reward-weight sensitivity**: uses the existing
   `scripts/scirep_reward_weight_sensitivity.py` runner.
4. **DEM/CRS audit**: `audit_dem_crs.py` records the available slope source,
   CRS and parcel-level summary. A true projected-CRS comparison must be run
   after the DEM raster is uploaded; the script refuses to label an
   interpolated raster as a higher-resolution DEM.

## Colab quick start

Use `paper9_revision_colab_v3_resume_only.ipynb`. Select an A100 or L4 GPU
runtime, open the notebook, and run cells from top to bottom. The first cells
mount Drive and extract these files from `MyDrive/paper9/`:

```text
prepared_neijiang_for_colab.zip
arcgis-farmland-mpc_minimal_for_colab.zip
paper9_revision_experiments_scripts_v3.zip
```

The notebook installs `onnxscript`, checks `revision_runs/neijiang/full/`, and
skips every directory that already contains `experiment_summary.json`. It
removes partial Tool 2/Tool 3 files only from an incomplete `pA_8000`
directory, then resumes the missing run. It never uses `--force` and never
deletes completed margin or pA results.

Expected Drive output:

```text
MyDrive/paper9/revision_runs/neijiang/full/
  margin_0.05/experiment_summary.json
  margin_0.10/experiment_summary.json
  margin_0.20/experiment_summary.json
  pA_1000/experiment_summary.json
  pA_2000/experiment_summary.json
  pA_4000/experiment_summary.json
  pA_8000/experiment_summary.json
```

After the final status cell prints `COMPLETE` for all seven directories,
download the seven `experiment_summary.json` files. Do not copy values into
the manuscript until those JSON files have been inspected.
