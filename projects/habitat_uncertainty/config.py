from dataclasses import dataclass, field
from typing import Dict

from habitat.config.default_structured_configs import LabSensorConfig
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin
from habitat_baselines.config.default_structured_configs import (
    AgentAccessMgrConfig,
    DDPPOConfig,
    HabitatBaselinesRLConfig,
    PolicyConfig,
    PPOConfig,
    PreemptionConfig,
    RLConfig,
    VERConfig,
    AuxLossConfig,
    ActionDistributionConfig,
    ObsTransformConfig,
    HierarchicalPolicyConfig,  
    MISSING,  
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

@dataclass
class CustomPolicyConfig(PolicyConfig):
    name: str = "PointNavResNetPolicy"
    action_distribution_type: str = "categorical"  # or 'gaussian'
    # If the list is empty, all keys will be included.
    # For gaussian action distribution:
    action_dist: ActionDistributionConfig = ActionDistributionConfig()
    obs_transforms: Dict[str, ObsTransformConfig] = field(default_factory=dict)
    hierarchical_policy: HierarchicalPolicyConfig = MISSING
    ovrl: bool = False
    no_downscaling: bool = False
    use_augmentations: bool = False
    deterministic_actions: bool = False
    use_yolo: bool = False
    yolo_model_id: str="yolov8s-world.pt"
    verbose: bool=True
    save_images: bool=False
    save_path: str = "image_dump/19"
    mask_rgb: bool=False
    num_env: int=32
    use_visual_encoder: bool=True
    use_depth_encoder: bool=True
    combine_encoders: bool=True
    
        
@dataclass
class customDDPPOConfig(DDPPOConfig):
    """Decentralized distributed proximal policy optimization config"""

    sync_frac: float = 0.6
    distrib_backend: str = "GLOO"
    rnn_type: str = "GRU"
    num_recurrent_layers: int = 1
    backbone: str = "resnet18"
    pretrained_weights: str = "data/ddppo-models/gibson-2plus-resnet50.pth"
    pretrained: bool = False
    pretrained_encoder: bool = False
    train_encoder: bool = True
    reset_critic: bool = True
    force_distributed: bool = False
    normalize_visual_inputs: bool = False
    use_detector: bool = False

@dataclass
class customAgentAccessMgrConfig(AgentAccessMgrConfig):
    type: str = "SingleAgentAccessManager"

@dataclass
class customRLConfig(RLConfig):
    """Reinforcement learning config"""

    agent: customAgentAccessMgrConfig = customAgentAccessMgrConfig()
    preemption: PreemptionConfig = PreemptionConfig()
    policy: CustomPolicyConfig = CustomPolicyConfig()
    ppo: PPOConfig = PPOConfig()
    ddppo: customDDPPOConfig = customDDPPOConfig()
    ver: VERConfig = VERConfig()
    auxiliary_losses: Dict[str, AuxLossConfig] = field(default_factory=dict)

@dataclass
class customHabitatBaselinesRLConfig(HabitatBaselinesRLConfig):
    rl: customRLConfig = customRLConfig()

@dataclass
class YOLOObjectSensorConfig(LabSensorConfig):
    type: str = "YOLOObjectSensor"

@dataclass
class YOLOStartReceptacleSensorConfig(YOLOObjectSensorConfig):
    type: str = "YOLOStartReceptacleSensor"

@dataclass
class YOLOGoalReceptacleSensorConfig(YOLOObjectSensorConfig):
    type: str = "YOLOGoalReceptacleSensor"

@dataclass
class RecepEmbeddingSensorConfig(LabSensorConfig):
    type: str = "RecepEmbeddingSensor"
    embeddings_file: str = "data/objects/clip_recep_embeddings.pickle"
    dimensionality: int = 1024

    
@dataclass
class NewObjectEmbeddingSensorConfig(LabSensorConfig):
    type: str = "NewObjectEmbeddingSensor"
    embeddings_file: str = "data/objects/clip_embeddings.pickle"
    dimensionality: int = 1024

# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=customHabitatBaselinesRLConfig(),
)

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

cs.store(
    package="habitat.task.lab_sensors.recep_embedding_sensor",
    group="habitat/task/lab_sensors",
    name="recep_embedding_sensor",
    node=RecepEmbeddingSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.new_object_embedding_sensor",
    group="habitat/task/lab_sensors",
    name="new_object_embedding_sensor",
    node=NewObjectEmbeddingSensorConfig,
)
# cs.store(
#     package="habitat.task.lab_sensors.yolo_segmentation_sensor",
#     group="habitat/task/lab_sensors",
#     name="yolo_segmentation_sensor",
#     node=YOLOSensorConfig,
# )

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
