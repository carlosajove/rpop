#!/usr/bin/env bash
# xr_teleoperate (R1 5-DoF arms + BrainCo hands) against the MuJoCo sim.  Start ./run_sim.sh first.
# Pico 4 (same Wi-Fi): PICO Browser -> https://192.168.1.84:8012/?ws=wss://192.168.1.84:8012  (accept the
# self-signed cert; if the websocket fails use https://vuer.ai?ws=wss://192.168.1.84:8012), then "Virtual Reality".
# Keyboard in this terminal: r = start tracking, q = quit.   Add --record to record episodes (s = start/stop).
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate tv
# bigger head-camera panel in the headset (~80 deg vertical, matches the sim camera fovy=80) and steadier R1 IK
export XR_TELEOP_PANEL_HEIGHT=1.7
export R1_IK_SMOOTH=1
cd "$(dirname "$0")/../xr_teleoperate/teleop"
exec python teleop_hand_and_arm.py --arm R1_A5 --ee brainco --sim --img-server-ip 127.0.0.1 --display-mode pass-through "$@"
