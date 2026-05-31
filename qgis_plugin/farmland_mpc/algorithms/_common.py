"""Common helpers for Farmland MPC processing algorithms.

Locates the ``farmland-mpc`` executable (settings-configurable) and runs
it as a subprocess with stdout streaming into the QGIS feedback channel.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from qgis.core import (
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsSettings,
)

SETTINGS_KEY = "farmland_mpc/executable_path"
DEFAULT_FALLBACK_PATHS = (
    "~/miniconda3/envs/farmland-mpc/bin/farmland-mpc",
    "~/anaconda3/envs/farmland-mpc/bin/farmland-mpc",
    "~/.miniconda3/envs/farmland-mpc/bin/farmland-mpc",
    "/opt/homebrew/Caskroom/miniconda/base/envs/farmland-mpc/bin/farmland-mpc",
    "/opt/miniconda3/envs/farmland-mpc/bin/farmland-mpc",
)


def resolve_executable() -> str:
    """Return the absolute path to the ``farmland-mpc`` executable.

    Resolution order:
      1. ``QgsSettings`` value at ``farmland_mpc/executable_path``.
      2. ``FARMLAND_MPC_EXECUTABLE`` environment variable.
      3. ``shutil.which("farmland-mpc")`` on the user's ``PATH``.
      4. A short list of common conda-env locations.

    Raises ``QgsProcessingException`` if nothing is found, with a message
    instructing the user to set the path in QGIS settings.
    """
    settings = QgsSettings()
    configured = settings.value(SETTINGS_KEY, "", type=str)
    if configured:
        candidate = Path(os.path.expanduser(configured))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    env_var = os.environ.get("FARMLAND_MPC_EXECUTABLE", "").strip()
    if env_var:
        candidate = Path(os.path.expanduser(env_var))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    on_path = shutil.which("farmland-mpc")
    if on_path:
        return on_path

    for guess in DEFAULT_FALLBACK_PATHS:
        candidate = Path(os.path.expanduser(guess))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    raise QgsProcessingException(
        "Could not locate the farmland-mpc executable. "
        "Install farmland-mpc in a conda environment "
        "(see https://github.com/zhouning/arcgis-farmland-mpc "
        "→ docs/REPRODUCE.md), then set its path in "
        "QGIS Settings → Options → Advanced → Variable "
        "'farmland_mpc/executable_path' (typical value: "
        "~/miniconda3/envs/farmland-mpc/bin/farmland-mpc), "
        "or export FARMLAND_MPC_EXECUTABLE in the shell that "
        "started QGIS."
    )


def run_cli(
    subcommand: str,
    args: Sequence[str],
    feedback: QgsProcessingFeedback,
    work_dir: str | None = None,
) -> int:
    """Run ``farmland-mpc <subcommand> [args...]`` and stream output.

    Returns the process exit code. Raises ``QgsProcessingException`` if
    the user cancels mid-run or if the subprocess exits non-zero.
    """
    executable = resolve_executable()
    cmd = [executable, subcommand, *args]
    feedback.pushCommandInfo(" ".join(shlex.quote(c) for c in cmd))

    # Inherit env so the conda env's Python finds its site-packages.
    env = os.environ.copy()
    # Ensure the conda env's bin is at the front of PATH (so torch/etc. resolve).
    env["PATH"] = str(Path(executable).parent) + os.pathsep + env.get("PATH", "")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=work_dir,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )
    except OSError as e:
        raise QgsProcessingException(f"Failed to start farmland-mpc: {e}") from e

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if feedback.isCanceled():
                proc.terminate()
                raise QgsProcessingException("Cancelled by user.")
            line = line.rstrip()
            if not line:
                continue
            # Map farmland-mpc log levels to QGIS feedback channels.
            if " ERROR " in line or line.startswith("ERROR"):
                feedback.reportError(line)
            elif " WARNING " in line or line.startswith("WARNING"):
                feedback.pushWarning(line)
            else:
                feedback.pushInfo(line)
    finally:
        # Close the subprocess pipe explicitly to silence ResourceWarning
        # that QGIS surfaces when the algorithm finishes.
        proc.stdout.close()

    code = proc.wait()
    if code != 0:
        raise QgsProcessingException(
            f"farmland-mpc {subcommand} exited with code {code}. "
            "See the log above for the failure reason."
        )
    return code
