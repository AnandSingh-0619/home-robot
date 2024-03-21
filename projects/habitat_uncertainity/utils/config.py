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
@dataclass
class YOLO_SensorConfig(LabSensorConfig):
    type: str = "yolo_segmentation"

@dataclass
class YOLO_ObjectSegmentationSensorConfig(LabSensorConfig):
    type: str = "yolo_object_segmentation"

@dataclass
class StartYOLORecepSegmentationSensorConfig(YOLO_ObjectSegmentationSensorConfig):
    type: str = "start_yolo_recep_segmentation"

@dataclass
class GoalYOLORecepSegmentationSensorConfig(YOLO_ObjectSegmentationSensorConfig):
    type: str = "goal_yolo_recep_segmentation"
 # -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
cs.store(
    package="habitat.task.lab_sensors.yolo_segmentation",
    group="habitat/task/lab_sensors",
    name="yolo_segmentation",
    node=YOLO_SensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.yolo_object_segmentation",
    group="habitat/task/lab_sensors",
    name="yolo_object_segmentation",
    node=YOLO_ObjectSegmentationSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.start_yolo_recep_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="start_yolo_recep_segmentation_sensor",
    node=StartYOLORecepSegmentationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.goal_yolo_recep_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="goal_yolo_recep_segmentation_sensor",
    node=GoalYOLORecepSegmentationSensorConfig,
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
        