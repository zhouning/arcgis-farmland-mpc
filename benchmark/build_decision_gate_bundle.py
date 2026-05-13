"""Build decision_gate_bundle.zip for Colab Task 4 Decision Gate.

Run: python build_decision_gate_bundle.py
Output: ./decision_gate_bundle.zip

Layout inside zip:
    src/benchmark/...                         (code subset)
    src/test/{county_env,mpc_planner,parcel_scoring_policy}.py
    src/test/paper9_contrastive/contrastive_trainer.py
    src/adk/data_agent/transition_model.py
    decision_gate.ipynb                        (driver notebook)
    requirements_colab.txt                     (extra pip deps)
    README_COLAB.md                            (runbook)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent
TEST_ROOT = Path("D:/test")
ADK_ROOT = Path("D:/adk")
OUT_ZIP = BENCH / "decision_gate_bundle.zip"

BENCHMARK_INCLUDE = [
    "baselines",
    "eval",
    "generator",
    "presets",
    "profiling",
    "synthetic_env_loader.py",
    "requirements.txt",
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
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    notebook_path = BENCH / "profiling" / "decision_gate.ipynb"
    if not notebook_path.is_file():
        raise FileNotFoundError(f"notebook not found: {notebook_path}")

    total = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
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
        zf.write(notebook_path, "decision_gate.ipynb")
        zf.writestr("README_COLAB.md", README_COLAB)

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_ZIP}")
    print(f"  files: {total + 4}")
    print(f"  size : {size_mb:.2f} MB")


if __name__ == "__main__":
    main()