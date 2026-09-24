# R1 Isaac Lab — handoff notes (2026-09-09/10)

## Installation (done, verified)
- Display moved from AMD iGPU to RTX 5090, `prime-select nvidia`, X11. Driver 595.84 (Ubuntu open modules) kept; Isaac Sim 6.0.1 requires ≥595.58.03.
- Isaac Sim 6.0.1 zip → `~/isaacsim` (compat checker PASSED; CPU governor warning → use `powerprofilesctl set performance`; IOMMU warning irrelevant with one GPU).
- venv `~/venvs/isaaclab30-sim601-py312`: `isaacsim[all,extscache]==6.0.1.0`, torch 2.10.0+cu128 (Isaac Lab pin), Isaac Lab `release/3.0.0-beta2` editable (`./isaaclab.sh --install`), `r1_lab` editable.
- Hub asset cache installed. Big NVIDIA scenes (Rivermark) freeze the GUI on first load: load one small environment at a time.

## Asset
- Source: `unitree_ros/robots/r1_description/R1.urdf` (full body 26 DoF; `r1_a5` = upper body only; `r1_a7` = 7-DoF arms, not EDU; `r1_air` = R1 AIR).
- Build: `scripts/tools/build_r1_asset.sh` = Isaac Lab `convert_urdf.py --merge-joints` (ankle rod links merged, no drives, free root) + `patch_contact_report.py`.
- Geometry facts: standing pelvis height 0.727 m; sole 0.053 m below ankle; footprint heel −0.05 .. toe +0.13 m from ankle; CoM 3.6 cm ahead of pelvis; mass 28.84 kg.
- `R1_CFG` in `source/r1_lab/r1_lab/assets/r1.py`: init z 0.76, G1-style bent pose, gains legs 100–150/4–5, feet 20/2, waist 150/5, arms 40/10, head 40/5; effort limits from URDF (60/50/33 N·m). `R1_USD_DIR` env var overrides asset dir.

## Task (`source/r1_lab/r1_lab/tasks/manager_based/locomotion_velocity/`)
Copied from Isaac Lab G1 velocity task. Differences: R1 asset; height scanner + external forces on `pelvis_link`; base_contact on `pelvis_link`+`waist_yaw_link`; joint-deviation terms for waist/head instead of torso/fingers; terminations add `pelvis_too_low` (0.25 m); run 5 adds `feet_air_time_too_long` (−2, >0.5 s), `feet_contact_time_too_long` (−1, >0.8 s while commanded), `feet_slide` −0.25.

## Runs (`logs/rsl_rl/r1_flat/`)
| run | dir | change | outcome |
|---|---|---|---|
| 1 | 17-04-51 | baseline (contact-only termination) | falls every 1.3 s, lies down undetected; training metrics misleading |
| 2 | 17-29-08 | + pelvis termination, `--video` | aborted: video renders 4096 envs, 16 s/iter |
| 3 | 17-35-20 | + pelvis termination | walks, 0 falls in eval, tracking err 0.08 m/s, but left/right foot contact 68/32 % (hop) |
| 4 | 18-02-30_continued | run 3 resumed to 5499 it | reward 31.5, err 0.07, still 62/37 % hop |
| 5 | 18-51-59_gait_v1 | anti-hop rewards, from scratch, 3000 it | see `logs/run5_summary.txt` |

Diagnosis of the hop: `feet_air_time_positive_biped` = min(stance contact time, swing air time) clamped at threshold → a permanently raised foot maxes it.

## Tools written
`scripts/tools/patch_contact_report.py`, `build_r1_asset.sh`, `check_standing.py`, `record_videos.sh`, `post_run.sh`; `scripts/rsl_rl/eval_policy.py`.

## Open items / next
1. Judge run 5 (foot contact split, videos). If still asymmetric: add explicit symmetry reward or gait-phase reward; consider ankle stiffness 40.
2. Rough terrain: `Isaac-Velocity-Rough-R1-v0` (3000 it).
3. Teleop: WORKS end-to-end (2026-09-10 23:00, PICO drove GR1T2 hands in sim). Launch (headless XR is the documented, working profile): `cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-PickPlace-GR1T2-Abs-v0 --xr` (NO `--teleop_device`, NO `--enable_pinocchio`, NO `--viz kit`: windowed XR profile hides the AR panel so the XR session never starts). isaacteleop 1.3.131 ships with Isaac Lab 3.0 (pin ~=1.3.0; do NOT install 1.6.x); it launches CloudXR as a native process in `~/.cloudxr` (not Docker). Headset recipe: PICO Interaction setting = Hands only (task uses HandsSource only; controllers only drive the Start/Stop/Reset panel), controllers on the desk, PICO Browser -> `https://nvidia.github.io/IsaacTeleop/client/main/`, IP 192.168.1.84, port empty, accept cert, Connect; if connected but black/no panel, close and restart the browser (immersive session not entered). Start via pinch on the panel; user origin = robot origin (anchor 0,0,0). Ports TCP 49100/48322, UDP 47998; ufw inactive; PC on Wi-Fi. Recording: PICO screen recorder (start before Connect). Data: `scripts/tools/record_demos.py` (HDF5) + `replay_demos.py` / `hdf5_to_mp4.py`; training via isaaclab_mimic + robomimic.
4. Hands (U6 = BrainCo Revo 2; model only inside `unitree_ros/robots/g1_with_brainco_hand/`), R1 upper-body teleop task using `r1_a5` or the full body with fixed root.
5. Real robot: Unitree `xr_teleoperate` in a separate Python 3.10 conda env.
6. Commit `r1_lab` (currently uncommitted beyond the generator's initial commit).


## MuJoCo + xr_teleoperate (2026-09-17)
- New project dir `~/projects/rpop/r1_mujoco/` (README there). Conda env `tv` (`~/miniforge3`, Python 3.10, pinocchio 3.1.0 + casadi, mujoco 3.8.0, xr_teleoperate + teleimager + televuer + dex-retargeting + unitree_sdk2_python editable, `params-proto<3` needed for vuer 0.0.60).
- `xr_teleoperate/` (with submodules), `unitree_sdk2_python/`, `unitree_sdk2/` (C++, built + installed to `~/.local`), `brainco_hand_service/` cloned. BrainCo service needs `sudo apt install libspdlog-dev libfmt-dev libyaml-cpp-dev libboost-program-options-dev`, then `bash setup_brainco_service.sh`.
- MuJoCo model: `r1_mujoco/build_r1_scene.py` builds the standing R1 (legs frozen at hip -0.20/knee 0.42/ankle -0.23, pelvis 0.728 m, head 1.217 m) with BrainCo hands from `xr_teleoperate/assets/brainco_hand`. Unitree's R1.urdf `<mujoco meshdir="meshes">` + `filename="meshes/..."` doubles the path: fixed copy in `r1_mujoco/assets/R1_fixed.urdf`.
- Sim for teleop: `r1_mujoco/run_sim.sh` then `run_teleop.sh` (xr_teleoperate `--arm R1_A5 --ee brainco --sim --img-server-ip 127.0.0.1`). Verified end-to-end without headset: arm + hand controllers initialise, `rt/lowstate` 250 Hz, camera served. Pico URL `https://192.168.1.84:8012/?ws=wss://192.168.1.84:8012`; certs in `~/.config/xr_teleoperate/`.
- Results: `r1_mujoco/out/*.json|png`, summary in the design report (artifact) and in the README. Key numbers: hand workspace ~0.5 m^3 per hand (0.97 m^3 with waist yaw), hands never below 0.46 m, forward reach 0.55 m at 0.9-1.1 m height; rake head reaches the floor in front of the toes only with handle >= 0.8 m and an angled (>=45 deg) grip; plain across-the-palm grip needs >= 1.0-1.2 m; both hands rigid on one plain handle feasible in ~4-7 % of poses (diagonal handle only) -> one rigid hand + loose guide for the second.
