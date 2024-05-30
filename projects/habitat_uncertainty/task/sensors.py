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

        return np.array([class_id], dtype=np.int64,)
    

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


