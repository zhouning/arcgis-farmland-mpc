import json
import sys
from pathlib import Path
import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from sweep.manifest import (
    build_manifest, load_state, save_state,
    mark_running, mark_done, mark_failed, list_queued, SweepCell,
)

PRESETS = ["bishan_clone", "neijiang_clone", "plain_small_cons"]
METHODS = ["Random", "Greedy"]
SEEDS = [0, 1]


def test_build_manifest_full_product(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    cells = build_manifest(PRESETS, METHODS, SEEDS, out_csv=csv_path)
    assert len(cells) == len(PRESETS) * len(METHODS) * len(SEEDS)
    assert csv_path.exists()
    text = csv_path.read_text()
    assert "bishan_clone" in text and "Random" in text


def test_state_status_transitions(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    state_path = tmp_path / "state.json"
    cells = build_manifest(PRESETS, METHODS, SEEDS, out_csv=csv_path)
    save_state(state_path, cells)
    state = load_state(state_path)
    assert all(c["status"] == "queued" for c in state["cells"])
    assert len(list_queued(state)) == len(cells)

    target = state["cells"][0]
    cell_id = target["cell_id"]
    mark_running(state, cell_id)
    save_state(state_path, state["cells"])
    state = load_state(state_path)
    assert next(c for c in state["cells"] if c["cell_id"] == cell_id)["status"] == "running"

    mark_done(state, cell_id, result_path="results/foo.json")
    assert next(c for c in state["cells"] if c["cell_id"] == cell_id)["status"] == "done"

    cell2 = state["cells"][1]["cell_id"]
    mark_failed(state, cell2, error="boom")
    assert next(c for c in state["cells"] if c["cell_id"] == cell2)["status"] == "failed"
    assert next(c for c in state["cells"] if c["cell_id"] == cell2)["error"] == "boom"


def test_list_queued_excludes_done(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    cells = build_manifest(PRESETS, METHODS, SEEDS, out_csv=csv_path)
    state = {"cells": cells}
    n = len(cells)
    mark_done(state, cells[0]["cell_id"], result_path="r/0.json")
    mark_done(state, cells[1]["cell_id"], result_path="r/1.json")
    mark_failed(state, cells[2]["cell_id"], error="x")
    queued = list_queued(state, include_failed=False)
    assert len(queued) == n - 3
    queued_with_failed = list_queued(state, include_failed=True)
    assert len(queued_with_failed) == n - 2
