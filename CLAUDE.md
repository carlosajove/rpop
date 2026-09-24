# Unitree R1 EDU — Isaac Sim / Isaac Lab project (context for Claude sessions)

Owner: Carlos. Machine `carlos-rpop`: Ubuntu 24.04, RTX 5090 (driver 595.84, open kernel modules, Secure Boot on),
Ryzen 9 9950X, 64 GB. Full history and findings: `r1_lab/docs/HANDOFF.md`. Install report: `~/Desktop/isaac-sim-install-report.md`.

## Goal
Simulate and RL-train a Unitree R1 EDU humanoid; teleoperate it in sim with a **PICO 4 Ultra** (Carlos has the headset)
via NVIDIA Isaac Teleop / CloudXR; later drive the real robot with Unitree `xr_teleoperate`. R1 EDU variant (U1..U6) still unknown;
all EDU variants share the 26-DoF body (5-DoF arms); U6 adds BrainCo Revo 2 hands.

## Layout (tools in $HOME, work here)
- `~/isaacsim/`                     Isaac Sim 6.0.1 workstation zip (GUI). Launch `./isaac-sim.sh`.
- `~/IsaacLab/`                     Isaac Lab clone, branch `release/3.0.0-beta2` (Isaac Sim 6.0.1 line). NEVER edit.
- `~/venvs/isaaclab30-sim601-py312` venv: isaacsim 6.0.1.0 pip, torch 2.10.0+cu128 (Isaac Lab pin; do NOT re-run the isaacsim pip line, it drags torch to 2.11), Isaac Lab editable, r1_lab editable.
- `unitree_ros/`                    sparse clone of Unitree URDFs (R1 variants, dexterous hands, G1+BrainCo).
- `r1_usd/R1_body_merged/`          canonical R1 USD (27 bodies, 26 joints). Built by `r1_lab/scripts/tools/build_r1_asset.sh`. `R1_body/` = GUI import, reference only.
- `r1_mujoco/`                      MuJoCo standing R1 + BrainCo hands: reach/rake study and the xr_teleoperate simulator bridge (see its README). Runs in conda env `tv` (`conda activate tv`).
- `xr_teleoperate/`, `unitree_sdk2_python/`, `unitree_sdk2/`, `brainco_hand_service/`  Unitree teleop stack (installed in `tv`; C++ sdk installed to `~/.local`).
- `r1_lab/`                         Isaac Lab external project (git repo, uncommitted changes as of 2026-09-10). Tasks `Isaac-Velocity-{Flat,Rough}-R1-v0` (+ `-Play-v0`).

## Everyday commands (always: `source ~/venvs/isaaclab30-sim601-py312/bin/activate; export OMNI_KIT_ACCEPT_EULA=YES; cd ~/projects/unitree-r1/r1_lab`)
- Train:   `python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-R1-v0 --num_envs 4096 --max_iterations 3000 --run_name <name>`  (~0.7 s/iter; headless by default; do NOT use `--video` in training, it renders all envs and is 20x slower)
- Watch:   `python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-R1-Play-v0 --num_envs 50 --viz kit --real-time [--checkpoint <run>/model_N.pt]`
- Eval:    `python scripts/rsl_rl/eval_policy.py --task Isaac-Velocity-Flat-R1-Play-v0 --num_envs 200 --checkpoint <run>/model_N.pt --eval_steps 500`  (height, tilt, tracking error, terminations by cause, foot contact split)
- Videos:  `bash scripts/tools/record_videos.sh <run_dir> Isaac-Velocity-Flat-R1-Play-v0 model_1000.pt model_2999.pt`
- Post-run (eval + videos + summary): `bash scripts/tools/post_run.sh <run_dir> <label>`
- Standing/asset check: `python scripts/tools/check_standing.py --zero_pose --z 0.74 --gain_scale 10 --no_reset_noise --diag`
- Power profile: `powerprofilesctl set performance` while simulating, `balanced` otherwise.

## Gotchas (all verified)
- Isaac Sim 6.0 URDF converter nests rigid bodies; Isaac Lab 3.0 `activate_contact_sensors` only tags the pelvis. Fixed by `patch_contact_report.py` (bakes PhysxContactReportAPI into `payloads/Physics/physx.usda`). Any new asset must be patched.
- Importing `isaaclab.envs.mdp` boots Kit → never import it in a package that loads at task registration (see `locomotion_velocity/mdp/__init__.py`). Config checks outside Kit need `OMNI_KIT_ACCEPT_EULA=YES` or they hang on an EULA prompt.
- `feet_air_time_positive_biped` saturates for a foot held in the air → one-legged hop is optimal. Countered by `r1_mdp.feet_air_time_too_long` / `feet_contact_time_too_long` (run 5).
- Passive zero-action standing fails (ankle lever ~170 N·m/rad vs 20 N·m/rad gains): expected, not a bug.
- Termination: pelvis below 0.25 m (Carlos's choice; 0.40 would also catch kneeling). No orientation termination (his choice).
- Don't `pkill -f` with a pattern that appears in your own command line.

## Status 2026-09-10 evening
Runs in `r1_lab/logs/rsl_rl/r1_flat/`: run3 `17-35-20` walks but hops on left leg; run4 `18-02-30_continued` same; run5 `18-51-59_gait_v1` stalled at iteration 985 (process hung 2.5 h, GPU idle; cause unknown) and was resumed from model_950 as `2026-09-10_21-44-31_gait_v1b` to ~3000 it. `post_run.sh` runs after it: see `logs/run5_summary.txt` and `logs/rsl_rl/r1_flat/2026-09-10_21-44-31_gait_v1b/videos/play/*.mp4`.
Next: judge run 5 gait (foot contact L/R should be ~equal) → rough terrain → PICO 4 Ultra teleop of GR1T2 in sim WORKS (recipe in HANDOFF item 3) → record demos to HDF5 → R1 upper-body teleop task → R1 hands (U6: BrainCo) → xr_teleoperate on the real robot.

## Status 2026-09-17
MuJoCo path done (HANDOFF "MuJoCo + xr_teleoperate"): `r1_mujoco/run_sim.sh` + `run_teleop.sh` drive the R1 A5 model from xr_teleoperate in `--sim` mode; Pico 4 not yet tried against it (Isaac/CloudXR route was). BrainCo C++ service awaits the apt packages (needs sudo).
