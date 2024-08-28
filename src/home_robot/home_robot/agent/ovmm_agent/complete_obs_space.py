# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import gym.spaces as spaces
import numpy as np


def get_complete_obs_space(skill_config, baseline_config):
    """
    Get superset of observation space needed for any policy.
    This avoids needing to use the habitat configs to import the observation space on hardware.
    TODO: Find way to import observation space from regular YAML configs to avoid this hardcoding.
    """
    val= 512
    if "recep_embedding" in skill_config.gym_obs_keys:
        val = 1024
    return spaces.dict.Dict(
        {
            "is_holding": spaces.Box(0.0, 1.0, (1,), np.float32),
            "head_depth": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.float32,
            ),
            "joint": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (10,),
                np.float32,
            ),
            "object_embedding": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (val,),
                np.float32,
            ),
            "relative_resting_position": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (3,),
                np.float32,
            ),
            "object_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "goal_recep_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "ovmm_nav_goal_segmentation": spaces.Box(
                0.0,
                1.0,
                (
                    skill_config.sensor_height,
                    skill_config.sensor_width,
                    skill_config.nav_goal_seg_channels,
                ),
                np.int32,
            ),
            "receptacle_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "robot_start_gps": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (2,),
                np.float32,
            ),
            "robot_start_compass": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (1,),
                np.float32,
            ),
            "start_recep_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "start_receptacle": spaces.Box(
                0,
                baseline_config.ENVIRONMENT.num_receptacles - 1,
                (1,),
                np.int64,
            ),
            "goal_receptacle": spaces.Box(
                0,
                baseline_config.ENVIRONMENT.num_receptacles - 1,
                (1,),
                np.int64,
            ),
            "rgb": spaces.Box(
                0.0,
                255,
                (skill_config.sensor_height, skill_config.sensor_width, 3),
                np.float32,
            ),
            "yolo_object_sensor": spaces.Box(
                0,
                150,
                (1,),
                np.int64,
            ),
            "yolo_start_receptacle_sensor": spaces.Box(
                0,
                150,
                (1,),
                np.int64,
            ),
            "yolo_goal_receptacle_sensor": spaces.Box(
                0,
                150,
                (1,),
                np.int64,
            ),
            "head_rgb": spaces.Box(
                0.0,
                255,
                (skill_config.sensor_height, skill_config.sensor_width, 3),
                np.float32,
            ),
            "recep_embedding": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (val,),
                np.float32,
            ),
            "new_object_embedding_sensor": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (1024,),
                np.float32,
            ),
        }
    )
