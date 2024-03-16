#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pickle
from typing import Any, Optional

import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.datasets.ovmm.ovmm_dataset import OVMMDatasetV0, OVMMEpisode
from habitat.tasks.ovmm.sub_tasks.nav_to_obj_sensors import (
    OVMMNavToObjSucc,
    OVMMRotDistToGoal,
    TargetIoUCoverage,
)
from habitat.tasks.ovmm.sub_tasks.place_sensors import OVMMPlaceSuccess
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToGoal,
    NavToPosSucc,
)
from habitat.tasks.rearrange.sub_tasks.pick_sensors import RearrangePickSuccess

from YOLOPerception import predict as YOLO_pred

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

        segmentation_sensor = YOLO_pred(
            obs=observations,
            depth_threshold=None,
            draw_instance_predictions=False,
        )

        return segmentation_sensor
