"""Sweep manifest CRUD: CSV definition + JSON state."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import TypedDict, Optional


class SweepCell(TypedDict):
    cell_id: str
    preset_id: str
    method: str
    seed: int
    status: str            # queued | running | done | failed
    result_path: Optional[str]
    error: Optional[str]


def _cell_id(preset: str, method: str, seed: int) -> str:
    return f"{preset}__{method}__seed{seed}"


def build_manifest(
    presets: list[str],
    methods: list[str],
    seeds: list[int],
    out_csv: str | Path,
) -> list[SweepCell]:
    cells: list[SweepCell] = []
    for p in presets:
        for m in methods:
            for s in seeds:
                cells.append(SweepCell(
                    cell_id=_cell_id(p, m, s),
                    preset_id=p, method=m, seed=int(s),
                    status="queued", result_path=None, error=None,
                ))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SweepCell.__annotations__.keys()))
        writer.writeheader()
        for c in cells:
            writer.writerow(c)
    return cells


def load_state(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"cells": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: str | Path, cells: list[SweepCell]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells}, indent=2), encoding="utf-8")


def _find(state: dict, cell_id: str) -> SweepCell:
    for c in state["cells"]:
        if c["cell_id"] == cell_id:
            return c
    raise KeyError(cell_id)


def mark_running(state: dict, cell_id: str) -> None:
    _find(state, cell_id)["status"] = "running"


def mark_done(state: dict, cell_id: str, result_path: str) -> None:
    c = _find(state, cell_id)
    c["status"] = "done"
    c["result_path"] = result_path
    c["error"] = None


def mark_failed(state: dict, cell_id: str, error: str) -> None:
    c = _find(state, cell_id)
    c["status"] = "failed"
    c["error"] = error


def list_queued(state: dict, include_failed: bool = False) -> list[SweepCell]:
    pending = {"queued", "running"}
    if include_failed:
        pending.add("failed")
    return [c for c in state["cells"] if c["status"] in pending]
