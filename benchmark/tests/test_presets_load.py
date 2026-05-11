from pathlib import Path
import pytest
from generator.schema import load_preset

PRESETS = [
    "bishan_clone", "neijiang_clone",
    "plain_small_cons", "plain_large_cons", "plain_medium_frag",
    "mixed_medium_frag", "hilly_small_cons",
]


@pytest.mark.parametrize("preset_id", PRESETS)
def test_preset_loads(preset_id):
    path = Path(__file__).resolve().parents[1] / "presets" / f"{preset_id}.yaml"
    cfg = load_preset(path)
    assert cfg.preset_id == preset_id
    assert cfg.n_blocks_target > 0
