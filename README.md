# r1_mujoco — Unitree R1 (5-DoF arms, BrainCo hands) in MuJoCo

Standing R1 with the legs frozen, BrainCo Revo2 hands, floor and props. Used for (a) the reach / rake study for the
set designer and (b) practising xr_teleoperate rake strokes in VR against a simulator (no robot).

Everything runs from the `tv` conda env (`conda activate tv`, Python 3.10, pinocchio 3.1, mujoco 3.8, xr_teleoperate
and unitree_sdk2_python installed editable) except the analysis scripts, which also run in the Isaac Lab venv.

## Files
| file | purpose |
|---|---|
| `build_r1_scene.py` | URDF -> MJCF. Freezes the 12 leg joints at the standing pose, lifts the pelvis so the soles touch z=0, attaches the BrainCo hand URDFs (from `xr_teleoperate/assets/brainco_hand`) on the wrist flanges, adds sites `{L,R}_wrist/flange/ee/grasp`, head camera, torque actuators for the 14 upper-body joints, optional welded rake (`--rake-length`, `--rake-tilt`). Output `out/r1_scene_<pose>.xml`, `out/r1_rake_right_<L>[_tilt<T>].xml`. |
| `reach_analysis.py` | Monte-Carlo workspace of each hand (joint-space sampling + MuJoCo FK + collision filter), voxel volume, reach at heights, two-hand overlap, analytic rake-head-on-floor check for several handle lengths and grip angles. `--rake-scenes` runs the collision-aware check with the real rake body. Writes `out/reach_*.json`, `out/cloud_*.npz`. |
| `two_hand_handle.py` | Closed-chain test: both hands on one straight handle (position + handle-axis constraint per hand, bounded least-squares IK seeded from the cloud, then collision check) over a grid of handle poses, separations and orientations. `out/two_hand_handle*.json`. |
| `plot_reach.py` | Figures `out/reach_workspace.png`, `out/rake_footprints.png`. |
| `r1_mujoco_sim.py` | The simulator that replaces the robot for xr_teleoperate `--sim`: DDS domain 1, `rt/lowcmd` in (PD from kp/kd/q/dq/tau), `rt/lowstate` out at 250 Hz, `rt/brainco/*/{cmd,state}`, `rt/reset_pose/cmd`, plus a teleimager-compatible head camera server (ZMQ config :60000, JPEG :55555, stereo 480x1280). |
| `test_r1_ik_offline.py` | Feeds a scripted rake-stroke through xr_teleoperate's `R1_A5_ArmIK` and evaluates the poses in MuJoCo (errors, joint limits, renders); `--dds` streams it to the running sim. |
| `run_sim.sh`, `run_teleop.sh` | Launchers (sim first, then teleop). |
| `assets/R1_fixed.urdf` | Unitree `R1.urdf` with the `<mujoco meshdir>` bug fixed (it produced `meshes/meshes/...`). |

## Model facts (standing, bent-knee pose = the r1_lab / G1-style init pose)
- pelvis 0.728 m, shoulder pitch axis 0.974 m, top of head 1.217 m (straight legs: +1.3 cm).
- Hand frame: BrainCo `<side>_base_link` mounted 0.140 m along the wrist-roll axis with the G1 convention
  (fingers along the forearm, palms facing the midline). `*_ee` = xr_teleoperate's IK target (0.20 m along the wrist axis, mid palm);
  `*_grasp` = handle centre of a power grasp (hand-frame x 0.025, z 0.055 from the hand base).
- Shoulder roll limits are asymmetric: 142 deg outward, only 13 deg inward, so an arm cannot cross the body midline.
- The wrist has only a roll axis: a handle held across the palm is always perpendicular to the forearm.

## Teleop in VR
```
./run_sim.sh --rake-length 0.8          # MuJoCo viewer opens; --headless for no window
./run_teleop.sh                         # xr_teleoperate R1_A5 + brainco, --sim, image server 127.0.0.1
```
Pico 4: PICO Browser -> `https://192.168.1.84:8012/?ws=wss://192.168.1.84:8012`, accept the self-signed certificate
(`~/.config/xr_teleoperate/{cert,key}.pem`, SAN 192.168.1.84 / 100.68.73.8), Virtual Reality, then press `r` in the
teleop terminal. `--input-mode controller` for controllers, `--record` to write episodes to `teleop/utils/data`.
