#!/usr/bin/env bash
# Build the R1 training asset from Unitree's URDF.
#   URDF  --(Isaac Lab URDF converter, fixed joints merged, no drives)-->  USD
#         --(patch_contact_report.py)-->  USD with contact reporting on every link
#
# Requires the Isaac Lab venv to be active and $ISAACLAB_PATH to point at the Isaac Lab clone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
URDF="${R1_URDF:-$HOME/projects/unitree-r1/unitree_ros/robots/r1_description/R1.urdf}"
OUT_DIR="${R1_USD_DIR:-$HOME/projects/unitree-r1/r1_usd}/R1_body_merged"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
export OMNI_KIT_ACCEPT_EULA=YES

TMP="$(mktemp -d)"
python "$ISAACLAB_PATH/scripts/tools/convert_urdf.py" "$URDF" "$TMP/R1.usd" \
    --merge-joints --joint-stiffness 0 --joint-damping 0 --joint-target-type position

rm -rf "$OUT_DIR"
mkdir -p "$(dirname "$OUT_DIR")"
cp -r "$TMP/R1.usd/R1" "$OUT_DIR"
cp "$TMP/R1.usd/config.yaml" "$OUT_DIR/converter_config.yaml"
rm -rf "$TMP"

python "$ROOT/scripts/tools/patch_contact_report.py" "$OUT_DIR/R1.usda"
echo "R1 asset ready at $OUT_DIR/R1.usda"
