"""Bridge synthetic datasets to D:/test/county_env.py via constant patching.

Mirrors county_env_neijiang.py's monkey-patch pattern. The synthetic dataset
directory must contain DLTB_with_slope.gpkg + township_<code>/ dirs.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

D_TEST_ROOT = Path(os.environ.get("TEST_SRC_ROOT", "D:/test"))
if str(D_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(D_TEST_ROOT))


def make_synthetic_env(dataset_dir: str | Path, **env_kwargs):
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing in {dataset_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_townships = int(manifest["n_townships"])
    county_code = "999999"
    township_codes = {
        f"{county_code}{i + 1:03d}": f"S{i + 1:02d}"
        for i in range(n_townships)
    }

    import county_env
    county_env.DLTB_PATH = str(dataset_dir / "DLTB_with_slope.gpkg")
    county_env.BLOCK_DIR = str(dataset_dir)
    county_env.ALL_TOWNSHIPS = dict(township_codes)
    county_env.TOWNSHIP_CODES = sorted(township_codes.keys())
    # Synthetic data is written in EPSG:4523 with origin near (0,0), which does
    # not round-trip through EPSG:32648 (Bishan's UTM zone). Keep area
    # computation in the dataset's own projected meter CRS (no-op reprojection).
    county_env.PROJ_CRS = "EPSG:4523"

    from county_env import CountyLevelEnv
    return CountyLevelEnv(**env_kwargs)
