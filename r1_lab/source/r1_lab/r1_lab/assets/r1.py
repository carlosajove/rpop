# Copyright (c) 2026, r1_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for the Unitree R1 EDU humanoid (26-DoF body, no hands).

The USD asset is produced by ``scripts/tools/build_r1_asset.sh`` (Isaac Lab URDF converter with fixed
joints merged, then ``patch_contact_report.py``).  Its location is taken from the ``R1_USD_DIR`` environment variable so
the same config works on any machine; the default is ``<repo>/r1_usd`` next to this package.

Joint naming (from the URDF, identical left/right):
    legs:  {side}_hip_pitch_joint, {side}_hip_roll_joint, {side}_hip_yaw_joint, {side}_knee_joint,
           {side}_ankle_pitch_joint, {side}_ankle_roll_joint
    waist: waist_roll_joint, waist_yaw_joint
    arms:  {side}_shoulder_pitch_joint, {side}_shoulder_roll_joint, {side}_shoulder_yaw_joint,
           {side}_elbow_joint, {side}_wrist_roll_joint
    head:  head_pitch_joint, head_yaw_joint

Effort limits from the URDF: hips/knee/waist/shoulder pitch+roll 60 N·m, ankles 50 N·m,
shoulder yaw/elbow/wrist/head 33 N·m.  Total mass 28.8 kg.  Pelvis height standing 0.727 m.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))  # <repo>/r1_lab/source/r1_lab/r1_lab/assets
R1_USD_DIR = os.environ.get("R1_USD_DIR", os.path.join(_REPO_ROOT, "r1_usd"))
"""Directory holding the imported R1 USD assets. Override with the ``R1_USD_DIR`` env var."""

R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{R1_USD_DIR}/R1_body_merged/R1.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # standing pelvis height is 0.727 m with straight legs; the bent-knee pose below is shorter,
        # so start a little high and let it settle on the ground.
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_elbow_joint": 0.87,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim=60.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_yaw_joint"],
            effort_limit_sim=60.0,
            stiffness=150.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 33.0,
                ".*_elbow_joint": 33.0,
                ".*_wrist_roll_joint": 33.0,
            },
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pitch_joint", "head_yaw_joint"],
            effort_limit_sim=33.0,
            stiffness=40.0,
            damping=5.0,
            armature=0.005,
        ),
    },
)
"""Configuration for the Unitree R1 EDU humanoid body (no hands)."""
