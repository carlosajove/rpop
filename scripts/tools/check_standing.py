# Copyright (c) 2026, r1_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the R1 in a task with zero actions and report whether it stands.

Zero actions mean "hold the default joint pose" for the position-controlled actuators, so this
checks the asset, the joint-name patterns, the actuator gains and the initial height together.

Example::

    python scripts/tools/check_standing.py --task Isaac-Velocity-Flat-R1-v0 --num_envs 16 --steps 300
    python scripts/tools/check_standing.py --zero_pose --z 0.74 --gain_scale 10 --verbose
"""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Standing check with zero actions.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-R1-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=300, help="Number of environment steps to run.")
parser.add_argument("--zero_pose", action="store_true", help="Override the default joint pose with all zeros.")
parser.add_argument("--z", type=float, default=None, help="Override the initial pelvis height (m).")
parser.add_argument("--usd", type=str, default=None, help="Override the robot USD path.")
parser.add_argument("--gain_scale", type=float, default=1.0, help="Multiply all actuator stiffness/damping.")
parser.add_argument("--no_reset_noise", action="store_true", help="Disable random root velocity/pose on reset.")
parser.add_argument("--diag", action="store_true", help="Print joint tracking, torques and contact forces.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import r1_lab.tasks  # noqa: F401


def T(x):
    """Return a torch tensor from Isaac Lab data (warp arrays expose ``.torch``)."""
    return x.torch if hasattr(x, "torch") else x


def main():
    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        # keep the robot undisturbed
        env_cfg.events.push_robot = None
        env_cfg.events.base_external_force_torque = None
        if args_cli.no_reset_noise:
            env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
            env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        if args_cli.usd is not None:
            env_cfg.scene.robot.spawn.usd_path = args_cli.usd
        if args_cli.zero_pose:
            env_cfg.scene.robot.init_state.joint_pos = {".*": 0.0}
        if args_cli.z is not None:
            pos = env_cfg.scene.robot.init_state.pos
            env_cfg.scene.robot.init_state.pos = (pos[0], pos[1], args_cli.z)
        if args_cli.gain_scale != 1.0:
            for act in env_cfg.scene.robot.actuators.values():
                for attr in ("stiffness", "damping"):
                    v = getattr(act, attr)
                    scaled = {k: x * args_cli.gain_scale for k, x in v.items()} if isinstance(v, dict) else v * args_cli.gain_scale
                    setattr(act, attr, scaled)

        env = gym.make(args_cli.task, cfg=env_cfg)
        scene = env.unwrapped.scene
        robot = scene["robot"]
        sensor = scene["contact_forces"]
        names = robot.body_names

        print("\n=== resolved names ===")
        print(f"joints ({robot.num_joints}): {robot.joint_names}")
        print(f"bodies ({robot.num_bodies}): {names}")
        print(f"contact sensor bodies ({len(sensor.body_names)}): {sensor.body_names}")
        for name, act in robot.actuators.items():
            print(f"actuator '{name}': {len(act.joint_names)} joints -> {act.joint_names}")
        print(f"default root pos: {T(robot.data.default_root_state)[0, :3].tolist()}")
        default_pos = T(robot.data.default_joint_pos)[0]
        nonzero = {n: round(float(v), 3) for n, v in zip(robot.joint_names, default_pos) if abs(float(v)) > 1e-6}
        print(f"default joint pose (non-zero): {nonzero}")

        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        dt = env.unwrapped.step_dt
        i_lf, i_rf = names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")
        s_lf, s_rf = sensor.body_names.index("left_ankle_roll_link"), sensor.body_names.index("right_ankle_roll_link")

        def diag(step: int):
            jp, jt = T(robot.data.joint_pos)[0], T(robot.data.joint_pos_target)[0]
            dev = (jp - jt).abs()
            k = int(dev.argmax())
            g0 = T(robot.data.projected_gravity_b)[0]
            forces = T(sensor.data.net_forces_w)[0]
            tau = T(robot.data.applied_torque)[0]
            kt = int(tau.abs().argmax())
            bp = T(robot.data.body_pos_w)[0]
            print(
                f"        env0 g_b=({g0[0]:+.2f},{g0[1]:+.2f},{g0[2]:+.2f})"
                f"  worst joint {robot.joint_names[k]} pos={float(jp[k]):+.3f} tgt={float(jt[k]):+.3f}"
                f"  max|tau| {robot.joint_names[kt]}={float(tau[kt]):+.1f}Nm"
                f"  foot z L/R={float(bp[i_lf, 2]):.3f}/{float(bp[i_rf, 2]):.3f}"
                f"  foot Fz L/R={float(forces[s_lf, 2]):.0f}/{float(forces[s_rf, 2]):.0f}N"
            )

        term_count = 0
        print("\n=== stepping with zero actions ===")
        print(f"{'step':>5} {'t[s]':>6} {'pelvis z mean':>14} {'min':>7} {'max':>7} {'tilt[deg] max':>14} {'terminated':>11}")
        for i in range(args_cli.steps):
            with torch.inference_mode():
                _, _, terminated, _, _ = env.step(actions)
            term_count += int(terminated.sum())
            periodic = i % 25 == 0 or i == args_cli.steps - 1
            early = args_cli.diag and i < 60 and i % 5 == 0
            if periodic or early:
                z = T(robot.data.root_pos_w)[:, 2]
                grav = T(robot.data.projected_gravity_b)
                tilt = torch.rad2deg(torch.acos((-grav[:, 2]).clamp(-1, 1)))
                print(f"{i:>5} {i * dt:>6.2f} {z.mean():>14.3f} {z.min():>7.3f} {z.max():>7.3f} {tilt.max():>14.1f} {term_count:>11}")
                if args_cli.diag:
                    diag(i)
        z = T(robot.data.root_pos_w)[:, 2]
        verdict = "STANDING" if term_count == 0 and z.min() > 0.55 else "FALLEN or unstable"
        print(f"\n=== verdict: {verdict}  (terminations={term_count}, final pelvis z min={z.min():.3f}) ===")
        env.close()


if __name__ == "__main__":
    main()
