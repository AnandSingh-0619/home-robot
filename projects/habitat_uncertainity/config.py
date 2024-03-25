from dataclasses import dataclass, field
from typing import Optional

from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    HabitatConfig,
    LabSensorConfig,
    MeasurementConfig,
    SimulatorConfig,
)

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

cs = ConfigStore.instance()


##########################################################################
# Sensors
##########################################################################
# @dataclass
# class YOLOSensorConfig(LabSensorConfig):
#     type: str = "YOLOSensor"

# @dataclass
# class YOLOObjectSegmentationSensorConfig(LabSensorConfig):
#     type: str = "YOLOObjectSegmentationSensor"

# @dataclass
# class StartYOLORecepSegmentationSensorConfig(YOLOObjectSegmentationSensorConfig):
#     type: str = "StartYOLORecepSegmentationSensor"

# @dataclass
# class GoalYOLORecepSegmentationSensorConfig(YOLOObjectSegmentationSensorConfig):
#     type: str = "GoalYOLORecepSegmentationSensor"

@dataclass
class YOLOObjectSensorConfig(LabSensorConfig):
    type: str = "YOLOObjectSensor"

@dataclass
class YOLOStartReceptacleSensorConfig(YOLOObjectSensorConfig):
    type: str = "YOLOStartReceptacleSensor"

@dataclass
class YOLOGoalReceptacleSensorConfig(YOLOObjectSensorConfig):
    type: str = "YOLOGoalReceptacleSensor"
 # -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
# cs.store(
#     package="habitat.task.lab_sensors.yolo_segmentation_sensor",
#     group="habitat/task/lab_sensors",
#     name="yolo_segmentation_sensor",
#     node=YOLOSensorConfig,
# )

# cs.store(
#     package="habitat.task.lab_sensors.yolo_object_segmentation_sensor",
#     group="habitat/task/lab_sensors",
#     name="yolo_object_segmentation_sensor",
#     node=YOLOObjectSegmentationSensorConfig,
# )

# cs.store(
#     package="habitat.task.lab_sensors.start_yolo_recep_segmentation_sensor",
#     group="habitat/task/lab_sensors",
#     name="start_yolo_recep_segmentation_sensor",
#     node=StartYOLORecepSegmentationSensorConfig,
# )
# cs.store(
#     package="habitat.task.lab_sensors.goal_yolo_recep_segmentation_sensor",
#     group="habitat/task/lab_sensors",
#     name="goal_yolo_recep_segmentation_sensor",
#     node=GoalYOLORecepSegmentationSensorConfig,
# )

cs.store(
    package="habitat.task.lab_sensors.yolo_object_sensor",
    group="habitat/task/lab_sensors",
    name="yolo_object_sensor",
    node=YOLOObjectSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.yolo_start_receptacle_sensor",
    group="habitat/task/lab_sensors",
    name="yolo_start_receptacle_sensor",
    node=YOLOStartReceptacleSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.yolo_goal_receptacle_sensor",
    group="habitat/task/lab_sensors",
    name="yolo_goal_receptacle_sensor",
    node=YOLOGoalReceptacleSensorConfig,
)
class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://config/tasks/",
        )
        search_path.append(
            provider="habitat_baselines",
            path="pkg://habitat_baselines/config/",
        )
        