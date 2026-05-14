"""Sweep runner: iterate manifest cells, checkpoint after each, honour --limit-time."""
from __future__ import annotations
import argparse
import csv
import time
import traceback
from pathlib import Path

from .manifest import (
    load_state, save_state, mark_running, mark_done, mark_failed, SweepCell,
)


METHOD_DISPATCH = {"Random", "Greedy", "GA", "PPO", "MPC"}


def _load_or_create_state(manifest_csv: Path, state_path: Path) -> dict:
    if state_path.exists():
        return load_state(state_path)
    cells: list[SweepCell] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cells.append(SweepCell(
                cell_id=row["cell_id"],
                preset_id=row["preset_id"],
                method=row["method"],
                seed=int(row["seed"]),
                status=row.get("status") or "queued",
                result_path=row.get("result_path") or None,
                error=row.get("error") or None,
            ))
    save_state(state_path, cells)
    return {"cells": cells}


def _dispatch(method: str):
    """Lazy-import each runner so missing GPU deps don't block CPU-only methods."""
    if method == "Random":
        from baselines.run_random import run_random
        return run_random
    if method == "Greedy":
        from baselines.run_greedy import run_greedy
        return run_greedy
    if method == "GA":
        from baselines.run_ga import run_ga_baseline
        return run_ga_baseline
    if method == "PPO":
        from baselines.run_ppo import run_ppo
        return run_ppo
    if method == "MPC":
        from baselines.run_mpc import run_mpc
        return run_mpc
    raise ValueError(f"unknown method {method!r}")


def run_sweep(
    manifest_csv,
    state_path,
    data_root,
    results_root,
    method_kwargs: dict | None = None,
    resume: bool = False,
    limit_time_s: float | None = None,
    only_method: str | None = None,
    only_preset: str | None = None,
    shard: tuple[int, int] | None = None,
) -> None:
    manifest_csv = Path(manifest_csv)
    state_path = Path(state_path)
    data_root = Path(data_root)
    results_root = Path(results_root)
    method_kwargs = method_kwargs or {}

    state = _load_or_create_state(manifest_csv, state_path)
    t0 = time.time()
    completed = 0

    filtered_cells = state["cells"]
    if only_method:
        filtered_cells = [c for c in filtered_cells if c["method"] == only_method]
    if only_preset:
        filtered_cells = [c for c in filtered_cells if c["preset_id"] == only_preset]
    if shard is not None:
        n, k = shard
        filtered_cells = [c for i, c in enumerate(filtered_cells) if i % k == n]
        print(f"[sweep] shard {n}/{k}: {len(filtered_cells)} cells")

    for cell in filtered_cells:
        if cell["status"] == "done":
            continue
        # Budget check: stop only after at least one cell completes this session
        # (otherwise limit_time_s=0 would let the sweep make zero progress).
        if (limit_time_s is not None and completed > 0
                and (time.time() - t0) > limit_time_s):
            print(f"[sweep] limit-time exceeded ({limit_time_s}s); exiting early.")
            break

        cell_id = cell["cell_id"]
        preset = cell["preset_id"]
        method = cell["method"]
        seed = cell["seed"]
        runner = _dispatch(method)
        out_path = results_root / preset / method / f"seed{seed}.json"
        dataset_dir = data_root / f"{preset}_seed{seed}"

        mark_running(state, cell_id)
        save_state(state_path, state["cells"])
        cell_t0 = time.time()
        try:
            runner(dataset_dir=dataset_dir,
                   preset_id=preset, seed=seed,
                   out_path=out_path,
                   **method_kwargs.get(method, {}))
            mark_done(state, cell_id, result_path=str(out_path))
            print(f"[sweep] OK   {cell_id} in {time.time() - cell_t0:.1f}s")
            completed += 1
        except Exception as e:  # noqa: BLE001
            mark_failed(state, cell_id, error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"[sweep] FAIL {cell_id}")
        save_state(state_path, state["cells"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="sweep_manifest.csv")
    p.add_argument("--state", default="sweep_state.json")
    p.add_argument("--data-root", default="data_dev")
    p.add_argument("--results-root", default="results")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit-time", type=float, default=None,
                   help="exit cleanly after this many wall-seconds")
    p.add_argument("--only-method", default=None)
    p.add_argument("--only-preset", default=None)
    p.add_argument("--shard", default=None,
                   help="N/K: this session takes cells where index%%K==N. "
                        "Apply AFTER --only-method/--only-preset filtering.")
    args = p.parse_args()
    shard = None
    if args.shard:
        n_str, k_str = args.shard.split("/")
        shard = (int(n_str), int(k_str))
        if not (0 <= shard[0] < shard[1]):
            raise ValueError(f"--shard {args.shard} out of range")
    run_sweep(
        manifest_csv=args.manifest, state_path=args.state,
        data_root=args.data_root, results_root=args.results_root,
        resume=args.resume, limit_time_s=args.limit_time,
        only_method=args.only_method, only_preset=args.only_preset,
        shard=shard,
    )


if __name__ == "__main__":
    main()
