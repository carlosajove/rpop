#!/usr/bin/env bash
# MuJoCo R1 stand-in for xr_teleoperate.  Usage: ./run_sim.sh [--rake-length 0.8] [--headless] [--scene out/x.xml]
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate tv
cd "$(dirname "$0")"
exec python -u r1_mujoco_sim.py --cam-width 800 --cam-height 600 "$@"
