"""Chain runner: baseline train -> partial train -> MPC eval (both modes).

Runs the full Neijiang cross-region experiment end-to-end in one background
process. Writes a progress marker file after each phase so we know where
we are after reboots / interruptions.
"""
import subprocess
import os, sys
import time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
MARKER = ROOT / "chain_progress.txt"


def mark(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(MARKER, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def run(cmd, log_path: Path):
    mark(f"START: {' '.join(cmd)}  log={log_path.name}")
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    mark(f"END  : rc={rc}  {' '.join(cmd)}")
    if rc != 0:
        mark(f"FATAL: rc={rc} — aborting chain")
        raise SystemExit(rc)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    mark("=== Neijiang cross-region chain starting ===")

    # Phase 1: baseline train (5 seeds x 3 members x 15 epochs, ~2h)
    run([PYTHON, "train_5seed_neijiang.py", "--mode", "baseline", "--n_seeds", "5"],
        ROOT / "train_baseline.log")

    # Phase 2: partial transfer train (5 seeds x 3 members x 5 epochs, ~40min)
    run([PYTHON, "train_5seed_neijiang.py", "--mode", "partial", "--n_seeds", "5"],
        ROOT / "train_partial.log")

    # Phase 3: MPC eval baseline (5 seeds x 5 eps, ~1.5h)
    run([PYTHON, "eval_mpc_neijiang.py", "--mode", "baseline", "--n_seeds", "5",
         "--eval_episodes", "5"],
        ROOT / "eval_baseline.log")

    # Phase 4: MPC eval partial (~1.5h)
    run([PYTHON, "eval_mpc_neijiang.py", "--mode", "partial", "--n_seeds", "5",
         "--eval_episodes", "5"],
        ROOT / "eval_partial.log")

    mark("=== All phases complete ===")


if __name__ == "__main__":
    main()
