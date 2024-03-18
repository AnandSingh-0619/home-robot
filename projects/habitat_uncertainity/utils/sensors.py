#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



from typing import Any, Optional

import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes

from utils.YOLO_pred import YOLOPerception as YOLO_pred 

@registry.register_sensor
class YOLO_ObjectSegmentationSensor(Sensor):
    cls_uuid: str = "yolo_object_segmentation"
    panoptic_uuid: str = "head_panoptic"

    def __init__(
        self,
        sim,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._blank_out_prob = self._config.blank_out_prob
        self._sim = sim
        self._object_ids_start = self._sim.habitat_config.object_ids_start
        self._resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(
                self._resolution[0],
                self._resolution[1],
                1,
            ),
            low=0,
            high=1,
            dtype=np.uint8,
        )

    def get_observation(self, observations, *args, episode, task, **kwargs):
        classes = []
        for g in episode.candidate_objects_hard:
            category = self._sim.scene_obj_ids[g.object_category]
            classes.append(category)
            
        segmentation_sensor = YOLO_pred.predict(
            obs=observations,
            vocab = classes,
            depth_threshold=None,
            draw_instance_predictions=False,
        )

        return segmentation_sensor


@registry.register_sensor
class YOLO_RecepSegmentationSensor(YOLO_ObjectSegmentationSensor):
    cls_uuid: str = "yolo_recep_segmentation"

    def _get_recep_goals(self, episode):
        raise NotImplementedError

    def get_observation(self, observations, *args, episode, task, **kwargs):
        recep_goals = self._get_recep_goals(episode)
        classes = []
        for g in recep_goals:
            category = g.object_category
            classes.append(category)

        segmentation_sensor = YOLO_pred.predict(
            obs=observations,
            vocab = classes,
            depth_threshold=None,
            draw_instance_predictions=False,
        )
        return segmentation_sensor


    def get_observation(self, observations, *args, episode, task, **kwargs):

        segmentation_sensor = YOLO_pred.predict(
            obs=observations,
            depth_threshold=None,
            draw_instance_predictions=False,
        )

        return segmentation_sensor

@registry.register_sensor
class StartYOLORecepSegmentationSensor(YOLO_RecepSegmentationSensor):
    cls_uuid: str = "start_yolo_recep_segmentation"

    def _get_recep_goals(self, episode):
        return episode.candidate_start_receps

@registry.register_sensor
class GoalYOLORecepSegmentationSensor(YOLO_RecepSegmentationSensor):
    cls_uuid: str = "goal_yolo_recep_segmentation"

    def _get_recep_goals(self, episode):
        return episode.candidate_goal_receps