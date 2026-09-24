# Copyright (c) 2026, r1_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Gait-shaping rewards for a biped.

``feet_air_time_positive_biped`` in Isaac Lab saturates at its threshold for a foot that never comes
down, so a one-legged hop is an optimal policy under it.  The terms here close that loophole.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time_too_long(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_air_time: float) -> torch.Tensor:
    """Penalize any foot that has been in the air longer than ``max_air_time`` seconds.

    Returns the summed excess air time over the feet (>= 0), to be used with a negative weight.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.clamp(air_time - max_air_time, min=0.0).sum(dim=1)


def feet_contact_time_too_long(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_contact_time: float, command_name: str
) -> torch.Tensor:
    """Penalize a foot that stays planted longer than ``max_contact_time`` while a velocity is commanded.

    Discourages dragging one foot along the ground while the other does all the work.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    contact_time = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    excess = torch.clamp(contact_time - max_contact_time, min=0.0).sum(dim=1)
    moving = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return excess * moving
