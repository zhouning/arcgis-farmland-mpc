#!/usr/bin/env bash
# Three modes for the same image:
#   (default)         → JupyterLab on :8888 with the colab notebook pre-mounted
#   webui             → FastAPI form UI on :8000
#   smoke             → run the smoke_end_to_end test and exit
#   farmland-mpc ...  → forward straight to the CLI
#   bash / sh / ...   → exec whatever the user gave us (escape hatch)
set -euo pipefail

CMD="${1:-jupyter}"

case "$CMD" in
  jupyter)
    shift || true
    # Seed bundled notebook into /work on first run so the user lands on it.
    # Don't overwrite if the user already has their own copy.
    mkdir -p /work/notebooks
    if [ ! -e /work/notebooks/farmland_mpc_local.ipynb ]; then
      cp /repo/docker/notebooks/farmland_mpc_local.ipynb /work/notebooks/
    fi
    exec jupyter lab \
      --ip=0.0.0.0 --port=8888 \
      --no-browser \
      --ServerApp.root_dir=/work \
      --ServerApp.default_url=/lab/tree/notebooks/farmland_mpc_local.ipynb \
      --ServerApp.allow_origin='*' \
      --ServerApp.token="${JUPYTER_TOKEN:-}" \
      --ServerApp.password='' \
      "$@"
    ;;

  webui)
    shift || true
    exec uvicorn farmland_mpc_webui.app:app \
      --host 0.0.0.0 --port 8000 \
      --app-dir /repo/docker/webui \
      "$@"
    ;;

  smoke)
    cd /tmp
    exec python -m farmland_mpc.tests.smoke_end_to_end
    ;;

  farmland-mpc)
    exec "$@"
    ;;

  bash|sh)
    exec "$@"
    ;;

  *)
    # Unknown first arg → treat the whole argv as a command. Lets users do e.g.
    #   docker run --rm IMAGE python scripts/fetch_dem.py --dltb /work/in.shp ...
    exec "$@"
    ;;
esac
