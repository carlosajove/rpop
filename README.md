# rpop

Unitree R1 humanoid: Isaac Lab training, MuJoCo simulation, VR teleoperation.
Private repo: https://github.com/carlosajove/rpop

## Layout

| Folder | What |
|---|---|
| r1_lab/ | Isaac Lab extension: R1 asset config, `locomotion_velocity` task, train/eval scripts, tools. `logs/` (ignored) holds training runs. |
| r1_mujoco/ | MuJoCo scene builder, simulator, teleop bridge, reach analysis. `out/` (ignored) holds point clouds, plots, logs. |
| r1_usd/ | R1 USD assets (`R1_body_merged/` is the canonical one, built by `r1_lab/scripts/tools/build_r1_asset.sh`). |
| setup_brainco_service.sh | Installs the BrainCo hand service (systemd). |
| CLAUDE.md | Working notes and everyday commands. |

Third-party checkouts are expected next to these folders and are git-ignored. Clone them yourself:
`xr_teleoperate`, `unitree_ros`, `unitree_sdk2`, `unitree_sdk2_python`, `brainco_hand_service` from https://github.com/unitreerobotics.

## Environment

Isaac Lab work uses the shared venv on the workstation:

    source /opt/lab/venvs/isaaclab30-sim601-py312/bin/activate   # also accepts the Omniverse EULA
    python -c "import r1_lab; print(r1_lab.__file__)"

MuJoCo/teleop work uses the conda env `tv` (see `r1_mujoco/README.md`).

## Workflow

- Clone into your home and work on branches; open a pull request for `main`.
- `/data/projects/rpop` on the workstation is the reference copy: it stays on `main`, is updated only with `git pull`, and is what the shared venv imports. Do not edit it directly.
- To run your own branch against the shared venv, put your clone first on PYTHONPATH:

      export PYTHONPATH=~/rpop/r1_lab/source/r1_lab:$PYTHONPATH

- Generated data (logs, checkpoints, point clouds, videos) stays in `/data`, never in git.
- `R1_USD_DIR` defaults to `<repo>/r1_usd`; override with the env var if your assets live elsewhere.
