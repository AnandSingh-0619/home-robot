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
from habitat.tasks.ovmm.sub_tasks.nav_to_obj_task import OVMMDynNavRLEnv
from habitat_uncertainity.utils.YOLO_pred import (
    YOLOPerception as YOLO_pred, 
)
from habitat.datasets.ovmm.ovmm_dataset import OVMMDatasetV0, OVMMEpisode

CLASSES = [
    "action_figure", "android_figure", "apple", "backpack", "baseballbat",
    "basket", "basketball", "bath_towel", "battery_charger", "board_game",
    "book", "bottle", "bowl", "box", "bread", "bundt_pan", "butter_dish",
    "c-clamp", "cake_pan", "can", "can_opener", "candle", "candle_holder",
    "candy_bar", "canister", "carrying_case", "casserole", "cellphone", "clock",
    "cloth", "credit_card", "cup", "cushion", "dish", "doll", "dumbbell", "egg",
    "electric_kettle", "electronic_cable", "file_sorter", "folder", "fork",
    "gaming_console", "glass", "hammer", "hand_towel", "handbag", "hard_drive",
    "hat", "helmet", "jar", "jug", "kettle", "keychain", "knife", "ladle", "lamp",
    "laptop", "laptop_cover", "laptop_stand", "lettuce", "lunch_box",
    "milk_frother_cup", "monitor_stand", "mouse_pad", "multiport_hub",
    "newspaper", "pan", "pen", "pencil_case", "phone_stand", "picture_frame",
    "pitcher", "plant_container", "plant_saucer", "plate", "plunger", "pot",
    "potato", "ramekin", "remote", "salt_and_pepper_shaker", "scissors",
    "screwdriver", "shoe", "soap", "soap_dish", "soap_dispenser", "spatula",
    "spectacles", "spicemill", "sponge", "spoon", "spray_bottle", "squeezer",
    "statue", "stuffed_toy", "sushi_mat", "tape", "teapot", "tennis_racquet",
    "tissue_box", "toiletry", "tomato", "toy_airplane", "toy_animal", "toy_bee",
    "toy_cactus", "toy_construction_set", "toy_fire_truck", "toy_food",
    "toy_fruits", "toy_lamp", "toy_pineapple", "toy_rattle", "toy_refrigerator",
    "toy_sink", "toy_sofa", "toy_swing", "toy_table", "toy_vehicle", "tray",
    "utensil_holder_cup", "vase", "video_game_cartridge", "watch", "watering_can",
    "wine_bottle", "bathtub", "bed", "bench", "cabinet", "chair", "chest_of_drawers",
    "couch", "counter", "filing_cabinet", "hamper", "serving_cart", "shelves",
    "shoe_rack", "sink", "stand", "stool", "table", "toilet", "trunk", "wardrobe",
    "washer_dryer"
]

@registry.register_sensor
class YOLOSensor(Sensor):
    cls_uuid: str = "yolo_segmentation_sensor"
    panoptic_uuid: str = "head_panoptic"
    yolo_perception_instance = None
    def __init__(
        self,
        sim,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._sim = sim
        self._object_ids_start = self._sim.habitat_config.object_ids_start
        self._resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )
        self.classes =149
        super().__init__(config=config)
        self.segmentation = YOLO_pred()
        # if YOLOSensor.yolo_perception_instance is None:
        #     YOLOSensor.yolo_perception_instance = YOLO_pred()
        

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

            
        segmentation_sensor = self.segmentation.predict(
            obs=observations,
            depth_threshold=None,
            draw_instance_predictions=False,
        )
        observations["head_depth"] = segmentation_sensor
        return segmentation_sensor


@registry.register_sensor
class YOLOObjectSegmentationSensor(YOLOSensor):
    cls_uuid: str = "yolo_object_segmentation_sensor"
    panoptic_uuid: str = "head_panoptic"

    def __init__(
        self,
        sim,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._sim = sim
        self._object_ids_start = self._sim.habitat_config.object_ids_start
        self._resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )

        super().__init__(sim, config=config)

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
        
        category = episode.candidate_objects_hard[0].object_category
        classes = CLASSES       
        class_id = classes.index(category)
        yolo_segmentation_sensor = observations["head_depth"] # observations["yolo_segmentation"]
        filtered_mask = np.where(yolo_segmentation_sensor == class_id, 1, 0)  
        
        return filtered_mask


@registry.register_sensor
class YOLORecepSegmentationSensor(YOLOObjectSegmentationSensor):
    cls_uuid: str = "yolo_recep_segmentation_sensor"

    def _get_recep_goals(self, episode):
        raise NotImplementedError

    def get_observation(self, observations, *args, episode, task, **kwargs):
        recep_goals = self._get_recep_goals(episode)
        category = recep_goals[0].object_category
        classes = CLASSES       
        class_id = classes.index(category)
        yolo_segmentation_sensor = observations["head_depth"] 
        filtered_mask = np.where(yolo_segmentation_sensor == class_id, 1, 0)  
        
        return filtered_mask



@registry.register_sensor
class StartYOLORecepSegmentationSensor(YOLORecepSegmentationSensor):
    cls_uuid: str = "start_yolo_recep_segmentation_sensor"

    def _get_recep_goals(self, episode):
        return episode.candidate_start_receps

@registry.register_sensor
class GoalYOLORecepSegmentationSensor(YOLORecepSegmentationSensor):
    cls_uuid: str = "goal_yolo_recep_segmentation_sensor"

    def _get_recep_goals(self, episode):
        return episode.candidate_goal_receps

@registry.register_sensor
class YOLOObjectSensor(Sensor):
    cls_uuid: str = "yolo_object_sensor"

    def __init__(
        self,
        sim,
        config,
        dataset: "OVMMDatasetV0",
        category_attribute="object_category",
        name_to_id_mapping="obj_category_to_obj_category_id",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        self._category_attribute = category_attribute
        self._name_to_id_mapping = name_to_id_mapping

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = max(
            getattr(self._dataset, self._name_to_id_mapping).values()
        )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OVMMEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        category_name = getattr(episode, self._category_attribute)
        classes = CLASSES       
        class_id = classes.index(category_name)

        return np.array(class_id, dtype=np.int64,)
    

@registry.register_sensor
class YOLOStartReceptacleSensor(YOLOObjectSensor):
    cls_uuid: str = "yolo_start_receptacle_sensor"

    def __init__(
        self,
        sim,
        config,
        dataset: "OVMMDatasetV0",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            sim=sim,
            config=config,
            dataset=dataset,
            category_attribute="start_recep_category",
            name_to_id_mapping="recep_category_to_recep_category_id",
        )
@registry.register_sensor
class YOLOGoalReceptacleSensor(YOLOObjectSensor):
    cls_uuid: str = "yolo_goal_receptacle_sensor"

    def __init__(
        self,
        sim,
        config,
        dataset: "OVMMDatasetV0",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            sim=sim,
            config=config,
            dataset=dataset,
            category_attribute="goal_recep_category",
            name_to_id_mapping="recep_category_to_recep_category_id",
        )


# @registry.register_sensor
# class OVMMNavGoalYOLOSegmentationSensor(Sensor):
#     cls_uuid: str = "ovmm_nav_goal_yolo_segmentation"
#     panoptic_uuid: str = "head_panoptic"

#     def __init__(
#         self,
#         sim,
#         config,
#         dataset,
#         task,
#         *args: Any,
#         **kwargs: Any,
#     ):
#         self._config = config
#         self._sim = sim
#         self._object_ids_start = self._sim.habitat_config.object_ids_start
#         self._is_nav_to_obj = task.is_nav_to_obj
#         self._blank_out_prob = self._config.blank_out_prob
#         self.resolution = (
#             sim.agents[0]
#             ._sensors[self.panoptic_uuid]
#             .specification()
#             .resolution
#         )
#         self._num_channels = 2 if self._is_nav_to_obj else 1
#         self._resolution = (
#             sim.agents[0]
#             ._sensors[self.panoptic_uuid]
#             .specification()
#             .resolution
#         )
#         super().__init__(config=config)

#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid

#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return SensorTypes.TENSOR

#     def _get_observation_space(self, *args, **kwargs):
#         return spaces.Box(
#             shape=(
#                 self._resolution[0],
#                 self._resolution[1],
#                 self._num_channels,
#             ),
#             low=0,
#             high=1,
#             dtype=np.int32,
#         )

#     def _get_obs_channel(self, pan_obs, max_obs_val, goals, goals_type):
#         pan_obs = pan_obs.squeeze(axis=-1)
#         obs = np.zeros_like(pan_obs)
#         for goal in goals:
#             if goals_type == "obj":
#                 obj_category = self._sim.scene_obj_ids[goal.object_category]
#             elif goals_type == "rec":
#                 rom = self._sim.get_rigid_object_manager()
#                 handle = self._sim.receptacles[
#                     goal.object_name
#                 ].parent_object_handle
#                 obj_id = rom.get_object_id_by_handle(handle)
#             instance_id = obj_id + self._object_ids_start


#             obs[pan_obs == instance_id] = 1
#         return obs

#     def get_observation(
#         self, observations, *args, episode, task: OVMMDynNavRLEnv, **kwargs
#     ):
#         pan_obs = observations[self.panoptic_uuid]
#         max_obs_val = np.max(pan_obs)
#         obs = np.zeros(
#             (self.resolution[0], self.resolution[1], self._num_channels),
#             dtype=np.int32,
#         )
#         if self._is_nav_to_obj:
#             obs[..., 0] = self._get_obs_channel(
#                 pan_obs,
#                 max_obs_val,
#                 episode.candidate_objects_hard,
#                 "obj",
#             )
#             obs[..., 1] = self._get_obs_channel(
#                 pan_obs,
#                 max_obs_val,
#                 episode.candidate_start_receps,
#                 "rec",
#             )
#         else:
#             obs[..., 0] = self._get_obs_channel(
#                 pan_obs,
#                 max_obs_val,
#                 episode.candidate_goal_receps,
#                 "rec",
#             )

#         return obs


# @registry.register_sensor
# class ReceptacleSegmentationSensor(Sensor):
#     cls_uuid: str = "receptacle_segmentation"
#     panoptic_uuid: str = "head_panoptic"

#     def __init__(
#         self,
#         sim,
#         config,
#         *args: Any,
#         **kwargs: Any,
#     ):
#         self._config = config
#         self._sim = sim
#         self._object_ids_start = self._sim.habitat_config.object_ids_start
#         self._blank_out_prob = self._config.blank_out_prob
#         self.resolution = (
#             sim.agents[0]
#             ._sensors[self.panoptic_uuid]
#             .specification()
#             .resolution
#         )
#         super().__init__(config=config)

#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid

#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return SensorTypes.SEMANTIC

#     def _get_observation_space(self, *args, **kwargs):
#         return spaces.Box(
#             shape=(
#                 self.resolution[0],
#                 self.resolution[1],
#                 1,
#             ),
#             low=np.iinfo(np.uint32).min,
#             high=np.iinfo(np.uint32).max,
#             dtype=np.int32,
#         )

#     def get_observation(
#         self, observations, *args, episode, task: OVMMDynNavRLEnv, **kwargs
#     ):
#         obs = np.copy(observations[self.panoptic_uuid])
#         obj_id_map = np.zeros(np.max(obs) + 1, dtype=np.int32)
#         assert (
#             task.loaded_receptacle_categories
#         ), "Empty receptacle semantic IDs, task didn't cache them."
#         for obj_id, semantic_id in task.receptacle_semantic_ids.items():
#             instance_id = obj_id + self._object_ids_start
#             # Skip if receptacle is not in the agent's viewport or if the instance
#             # is selected to be blanked out randomly
#             if (
#                 instance_id >= obj_id_map.shape[0]
#                 or np.random.random() < self._blank_out_prob
#             ):
#                 continue
#             obj_id_map[instance_id] = semantic_id
#         obs = obj_id_map[obs]
#         return obs
