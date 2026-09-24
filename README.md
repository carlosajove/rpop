# unitree-r1 workspace (reference copy)

This folder is the team's REFERENCE checkout of the Unitree R1 work. The shared venv in /opt/lab imports `r1_lab` from here.

Rules
- Nobody edits code here directly. Code changes go through GitHub: clone into your home, work on a branch, open a PR.
- This copy stays on `main` and is only updated with `git pull` after a PR is merged.
- Generated data (logs, checkpoints, point clouds, videos) stays in /data and is git-ignored. Never commit it.

Run your own branch against the shared venv (no second venv needed):

    source /opt/lab/venvs/isaaclab30-sim601-py312/bin/activate
    export PYTHONPATH=~/work/r1_lab/source/r1_lab:$PYTHONPATH   # your clone first
    python -c "import r1_lab; print(r1_lab.__file__)"           # should print your clone

Contents
| Folder | What | Origin |
|---|---|---|
| r1_lab | IsaacLab extension: R1 asset config, locomotion_velocity task, training/eval scripts. `logs/` (ignored) holds training runs. | ours, git |
| r1_mujoco | MuJoCo scene builder, sim, teleop bridge, reach analysis. `out/` (ignored) holds point clouds, plots, logs. | ours, git |
| r1_usd | R1 USD assets (body, merged body, test scene). Data, not code. | ours, no git |
| xr_teleoperate | Unitree VR teleop stack, with local modifications to the arm IK and BrainCo hand control (to be pushed to a fork). | unitreerobotics fork |
| unitree_ros, unitree_sdk2, unitree_sdk2_python, brainco_hand_service | Unitree SDKs and services, plain clones. | unitreerobotics |
| setup_brainco_service.sh, CLAUDE.md | BrainCo hand service installer and workspace notes. | ours |

GitHub: r1_lab -> https://github.com/carlosajove/r1_lab, r1_mujoco -> https://github.com/carlosajove/r1_mujoco (private; ask carlos for collaborator access).
