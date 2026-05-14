"""Build Colab bundles for benchmark tasks.

Usage:
    python build_decision_gate_bundle.py           # Decision Gate bundle (60 KB)
    python build_decision_gate_bundle.py --ppo     # PPO sweep bundle (same sources + notebook)

Output: ./decision_gate_bundle.zip or ./ppo_sweep_bundle.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent
TEST_ROOT = Path("D:/test")
ADK_ROOT = Path("D:/adk")

BENCHMARK_INCLUDE = [
    "baselines",
    "eval",
    "generator",
    "presets",
    "profiling",
    "sweep",
    "synthetic_env_loader.py",
    "requirements.txt",
    "sweep_manifest.csv",
]
TEST_FILES = ["county_env.py", "mpc_planner.py", "parcel_scoring_policy.py"]
PAPER9_FILES = ["contrastive_trainer.py"]
ADK_DATA_AGENT = ["transition_model.py"]
EXCLUDE_PARTS = ("__pycache__", ".pytest_cache", "_tmp_cpu_profile")

COLAB_REQUIREMENTS = """# Installed on top of Colab default image (torch + numpy included).
# No version pins: let pip resolve against whatever torch Colab ships that week.
# sb3-contrib pulls stable-baselines3 + gymnasium as transitive deps.
sb3-contrib
geopandas
libpysal
opensimplex
rasterio
pyyaml
shapely
"""

README_COLAB = """# Decision Gate runbook

1. In Colab: Runtime -> Change runtime type -> T4 GPU.
2. Upload BOTH `decision_gate_bundle.zip` AND extract `decision_gate.ipynb`
   from the zip (or open it after Cell 1 unzips).
3. Run cells 1-8 top to bottom. Total wall ~10-15 min.
4. Cell 8 writes `/content/decision_artifacts.zip`. Download it.
5. Extract its 3 files into `benchmark/profiling/` and commit.
"""


def _skip(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts)


def _add_tree(zf: zipfile.ZipFile, src_root: Path, rel_items: list[str], prefix: str) -> int:
    count = 0
    for rel in rel_items:
        src = src_root / rel
        if not src.exists():
            raise FileNotFoundError(f"missing: {src}")
        if src.is_file():
            zf.write(src, f"{prefix}/{rel}")
            count += 1
        else:
            for f in src.rglob("*"):
                if f.is_dir() or _skip(f):
                    continue
                arc = f"{prefix}/{rel}/{f.relative_to(src)}".replace("\\", "/")
                zf.write(f, arc)
                count += 1
    return count


def main() -> None:
    import sys
    ppo_mode = "--ppo" in sys.argv

    if ppo_mode:
        out_zip = BENCH / "ppo_sweep_bundle.zip"
        notebook_path = BENCH / "sweep" / "ppo_sweep_colab.ipynb"
    else:
        out_zip = BENCH / "decision_gate_bundle.zip"
        notebook_path = BENCH / "profiling" / "decision_gate.ipynb"

    if out_zip.exists():
        out_zip.unlink()
    if not notebook_path.is_file():
        raise FileNotFoundError(f"notebook not found: {notebook_path}")

    total = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        total += _add_tree(zf, BENCH, BENCHMARK_INCLUDE, "src/benchmark")
        total += _add_tree(zf, TEST_ROOT, TEST_FILES, "src/test")
        zf.writestr("src/test/paper9_contrastive/__init__.py", "")
        total += _add_tree(
            zf, TEST_ROOT / "paper9_contrastive", PAPER9_FILES,
            "src/test/paper9_contrastive",
        )
        zf.writestr("src/adk/data_agent/__init__.py", "")
        total += _add_tree(
            zf, ADK_ROOT / "data_agent", ADK_DATA_AGENT, "src/adk/data_agent",
        )
        zf.writestr("requirements_colab.txt", COLAB_REQUIREMENTS)
        zf.write(notebook_path, notebook_path.name)
        if not ppo_mode:
            zf.writestr("README_COLAB.md", README_COLAB)

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_zip}")
    print(f"  files: {total + 3}")
    print(f"  size : {size_mb:.2f} MB")


if __name__ == "__main__":
    main()