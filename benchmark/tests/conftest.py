import sys
from pathlib import Path
import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))


@pytest.fixture
def rng_seed():
    return 42


@pytest.fixture
def tmp_out_dir(tmp_path):
    out = tmp_path / "synthetic"
    out.mkdir()
    return out
