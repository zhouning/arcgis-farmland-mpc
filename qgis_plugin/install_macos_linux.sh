#!/usr/bin/env bash
# Install the Farmland MPC QGIS Processing plugin into the user's
# QGIS profile. QGIS 4 reads from the QGIS3/ profile path (the directory
# name has not been bumped), so the same install works for QGIS 3.34 LTR
# and QGIS 4.0+. Symlinks rather than copies so subsequent `git pull`s
# update the plugin in place.
set -euo pipefail

PLUGIN_NAME="farmland_mpc"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/$PLUGIN_NAME"

if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC not found." >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin)
        BASE="$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"
        ;;
    Linux)
        BASE="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "On Windows, run install_windows.ps1 instead (or copy the folder manually)." >&2
        exit 1
        ;;
    *)
        echo "Unsupported OS." >&2
        exit 1
        ;;
esac

mkdir -p "$BASE"
DEST="$BASE/$PLUGIN_NAME"
if [ -L "$DEST" ] || [ -d "$DEST" ]; then
    echo "Removing existing $DEST"
    rm -rf "$DEST"
fi
ln -s "$SRC" "$DEST"
echo "Installed (symlink): $DEST -> $SRC"

cat <<EOF

Done. Next steps:

  Option A — from a terminal (no QGIS GUI required):

    qgis_process plugins enable farmland_mpc
    qgis_process list | grep -A4 "Farmland MPC"
    # then run an algorithm:
    qgis_process run farmland_mpc:prepare \\
        --DLTB=/path/to/dltb.shp \\
        --DEM=/path/to/dem.tif \\
        --OUT=/path/to/run/prepared \\
        --CRS=EPSG:32648

  Option B — from inside QGIS GUI:

    1. Plugins → Manage and Install Plugins → check "Farmland MPC".
    2. Settings → Options → Advanced → set
         farmland_mpc/executable_path
       to the absolute path of \`farmland-mpc\` in your conda env, e.g.
         \$HOME/miniconda3/envs/farmland-mpc/bin/farmland-mpc
       (or export FARMLAND_MPC_EXECUTABLE in the shell that starts QGIS).
    3. Open the Processing Toolbox; look for the "Farmland MPC" provider.

EOF
