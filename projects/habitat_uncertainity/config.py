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
        