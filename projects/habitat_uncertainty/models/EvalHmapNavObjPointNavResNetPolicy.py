#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from re import search
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat.tasks.ovmm.ovmm_sensors import (
    GoalReceptacleSensor,
    ObjectCategorySensor,
    StartReceptacleSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet, resnet_gn
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.ddppo.transforms import ShiftAndJitterTransform
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from habitat_uncertainty.task.sensors import (
    YOLOObjectSensor,
    YOLOStartReceptacleSensor,
    YOLOGoalReceptacleSensor,
)
import supervision as sv
from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO
from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
from habitat.core.logging import logger
from habitat_baselines.utils.timing import g_timer
import cv2
from ultralytics import SAM
from nvitop import Device
import time
from torchvision import transforms

import gc
PARENT_DIR = Path(__file__).resolve().parent
MOBILE_SAM_CHECKPOINT_PATH = str(PARENT_DIR / "pretrained_wt" / "mobile_sam.pt")
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


if TYPE_CHECKING:
    from omegaconf import DictConfig

try:
    import clip
except ImportError:
    clip = None


@baseline_registry.register_policy
class EvalHmapNavObjPointNavResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone; one of ["resnet18", "resnet50", "resneXt50", "se_resnet50", "se_resneXt50", "se_resneXt101", "resnet50_clip_avgpool", "resnet50_clip_attnpool"]
        """

        assert backbone in [
            "resnet18",
            "resnet50",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
            "resnet50_clip_avgpool",
            "resnet50_clip_attnpool",
        ], f"{backbone} backbone is not recognized."

        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            GazePointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                no_downscaling=hasattr(policy_config, "no_downscaling")
                and policy_config.no_downscaling,
                ovrl=hasattr(policy_config, "ovrl") and policy_config.ovrl,
                use_augmentations=getattr(
                    policy_config, "use_augmentations", False
                ),
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs=config.habitat_baselines.rl.ddppo.normalize_visual_inputs,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
        )

class YOLOPerception(PerceptionModule):
    def __init__(
        self,
        checkpoint_file: Optional[str] = MOBILE_SAM_CHECKPOINT_PATH,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = 0.05,
        
    ):
        """Loads a YOLO model for object detection and instance segmentation

        Arguments:
            yolo_model_id: one of "yolo_world/l" or "yolo_world/s" for large or small
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        yolo_model_id="yolov8x-worldv2.pt",
        self.verbose = verbose
        if checkpoint_file is None:
            checkpoint_file = str(
                Path(__file__).resolve().parent
                / "pretrained_wt/mobile_sam.pt"
            )
        if self.verbose:
            print(
                f"Loading YOLO model from {yolo_model_id} and MobileSAM with checkpoint={checkpoint_file}"   
            )
        with torch.no_grad():
            self.model = YOLO(model='yolov8x-worldv2.pt')
        vocab = CLASSES
        self.model.set_classes(vocab)
        # Freeze the YOLO model's parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.confidence_threshold = confidence_threshold
        with torch.no_grad():
            self.sam_model = SAM(checkpoint_file)
            # Freeze the SAM model's parameters
        for param in self.sam_model.parameters():
            param.requires_grad = False
        self.model.cuda()
        self.sam_model.cuda()
        torch.cuda.empty_cache()

    def create_gaussian_mask(self, height, width, boxes, max_sigma=30):
        if len(boxes) > 0:
            # print("Detected")
            boxes=boxes[0]
            x_min, y_min, x_max, y_max = boxes
            center = ((boxes[0] + boxes[2]) // 2, (boxes[1] + boxes[3]) // 2)    

            size = (x_max - x_min, y_max - y_min)
            sigma_x = min(size[0] / 3, max_sigma)
            sigma_y = min(size[1] / 3, max_sigma)

            y, x = np.ogrid[0:height, 0:width]
            distance = np.sqrt((x - center[0])**2 / (2 * (sigma_x**2) + 1e-6) + (y - center[1])**2 / (2 * (sigma_y**2) + 1e-6))        
            gaussian = np.exp(-distance)
            return np.expand_dims(gaussian / gaussian.max(), axis=2) 
        else:
            # print("Nothing Detected")
            return np.zeros((160, 120, 1))
    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
            indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
            image of shape (H, W, 3)
        """

        torch.cuda.empty_cache()
        # start_time = time.time()  
        nms_threshold=0.1
        seg_tensor = obs["ovmm_nav_goal_segmentation"]
        images_tensor = obs["head_rgb"] 
        obj_class_ids = obs["yolo_object_sensor"].cpu().numpy().flatten()
        rec_class_ids = obs["yolo_start_receptacle_sensor"].cpu().numpy().flatten()
        batch_size = images_tensor.shape[0]
        images = [images_tensor[i].cpu().numpy() for i in range(images_tensor.size(0))] 

        start_receptacle_index = CLASSES.index("bathtub")
        receptacle_class_indices = list(range(start_receptacle_index, len(CLASSES)))  
        search_list = np.concatenate((obj_class_ids, receptacle_class_indices)).tolist()

        height, width, _ = images[0].shape
        results = list(self.model(images, classes=search_list, conf=self.confidence_threshold,  iou=nms_threshold, stream=True, verbose=False))
        obj_semantic_masks = []
        rec_semantic_masks = []
        all_rec_masks = []

        for idx, result in enumerate(results):
            class_ids = result.boxes.cls.cpu().numpy()
            input_boxes = result.boxes.xyxy.cpu().numpy()

            obj_mask_idx = np.isin(class_ids, obj_class_ids[idx])
            rec_mask_idx = np.isin(class_ids, rec_class_ids[idx])

            obj_boxes = input_boxes[obj_mask_idx]
            rec_boxes = input_boxes[rec_mask_idx]
            all_rec_boxes = input_boxes[np.isin(class_ids, receptacle_class_indices)]

            obj_semantic_mask = self.create_gaussian_mask(height, width, obj_boxes)
            rec_semantic_mask = self.create_gaussian_mask(height, width, rec_boxes)
            
            obj_semantic_mask = cv2.resize(obj_semantic_mask, (120, 160), interpolation=cv2.INTER_NEAREST)
            rec_semantic_mask = cv2.resize(rec_semantic_mask, (120, 160), interpolation=cv2.INTER_NEAREST)
            
            obj_semantic_mask = np.expand_dims(obj_semantic_mask, axis=-1)
            rec_semantic_mask = np.expand_dims(rec_semantic_mask, axis=-1)
            obj_semantic_masks.append(obj_semantic_mask)
            rec_semantic_masks.append(rec_semantic_mask)
            del obj_semantic_mask, rec_semantic_mask
            all_rec_mask = np.zeros((height, width, 1), dtype=np.float64)
            if all_rec_boxes.size != 0:
                try:
                    with torch.no_grad():
                        sam_outputs =list(self.sam_model.predict(stream=True,source=result.orig_img, bboxes=all_rec_boxes, points=None, labels=None, verbose=False))
                        
                    sam_output = sam_outputs[0]
                    result_masks = sam_output.masks
                    masks_tensor = result_masks.data

                    for mask, class_id in zip(masks_tensor, class_ids):
                        mask_np = mask.cpu().numpy()[:, :, None]
                        if class_id in receptacle_class_indices:
                            all_rec_mask[mask_np > 0] = class_id- start_receptacle_index              


                    del sam_outputs, result_masks
                except Exception as e:
                    devices = Device.all()
                    logger.info(f"After SAM GPU Memory - Used: {devices[0].memory_used_human()}, Free: {devices[0].memory_free_human()}")
                    logger.info(f"An error occurred at index {idx}:{e}")

                    continue
            all_rec_mask = cv2.resize(all_rec_mask, (120, 160), interpolation=cv2.INTER_NEAREST)
            all_rec_mask = np.expand_dims(all_rec_mask, axis=-1)
            all_rec_masks.append(all_rec_mask)
        torch.cuda.empty_cache()
        obj_semantic_masks = np.array(obj_semantic_masks)
        rec_semantic_masks = np.array(rec_semantic_masks)
        all_rec_masks = np.array(all_rec_masks)

        combined_masks = np.concatenate((obj_semantic_masks, rec_semantic_masks, all_rec_masks), axis=-1)
        del obj_semantic_masks, rec_semantic_masks, results
        gc.collect()
        combined_masks = torch.tensor(combined_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device()))).detach().requires_grad_(False)
        return combined_masks

class GazeResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        no_downscaling=False,
        use_augmentations=False,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        self.no_downscaling = no_downscaling
        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid and k!="head_rgb"
        ]
        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            spatial_size_h = observation_space.spaces[
                self.visual_keys[0]
            ].shape[0]
            spatial_size_w = observation_space.spaces[
                self.visual_keys[0]
            ].shape[1]
            if not no_downscaling:
                spatial_size_h = spatial_size_h // 2
                spatial_size_w = spatial_size_w // 2
            self.backbone = make_backbone(
                self._n_input_channels, baseplanes, ngroups
            )

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                )
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )
            rgb_keys = [k for k in observation_space.spaces if "rgb" in k]
            rgb_size = [
                observation_space.spaces[k].shape[:2] for k in rgb_keys
            ]

            self.visual_transform = None
            if use_augmentations:
                self.visual_transform = ShiftAndJitterTransform(
                    size=rgb_size[0]
                )
                self.visual_transform.randomize_environments = False
        # Initialize the segmentation attribute
        self.segmentation = YOLOPerception(sem_gpu_id=0, verbose=False, confidence_threshold=0.2)
        self.masks = torch.zeros((1, 160, 120, 2))  # Adjust to the batch size


    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        torch.cuda.empty_cache()
        if self.is_blind:
            return None
        
        if (  # noqa: SIM401
                GazePointNavResNetNet.SEG_MASKS
                in observations
            ):
                self.masks = observations[GazePointNavResNetNet.SEG_MASKS]
        else:
            # with g_timer.avg_time("trainer.yolo_detector_step"):
            masks = self.segmentation.predict(observations)
            self.masks = torch.tensor(masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device()))).detach().requires_grad_(False)
            logger.info(f"Calling YOLO inference from visual encoder")


        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            #Make changes to the sensors as required by the GAZE skill
            if(k == "ovmm_nav_goal_segmentation"):
                obs_k = self.masks[..., 0:2]

            # if(k == "receptacle_segmentation"):
            #     obs_k = self.masks[..., 1:2]
      
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = (
                    obs_k.float() * self.key_needs_rescaling[k]
                )  # normalize
            if self.visual_transform is not None:
                obs_k = self.visual_transform(obs_k)
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1).float()
        if not self.no_downscaling:
            x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        torch.cuda.empty_cache()
        return x


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        pooling="attnpool",
    ):
        super().__init__()

        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if not self.is_blind:
            if clip is None:
                raise ImportError(
                    "Need to install CLIP (run `pip install git+https://github.com/openai/CLIP.git@40f5484c1c74edd83cb9cf687c6ab92b28d8b656`)"
                )

            model, preprocess = clip.load("RN50")

            # expected input: C x H x W (np.uint8 in [0-255])
            self.preprocess = T.Compose(
                [
                    # resize and center crop to 224
                    preprocess.transforms[0],
                    preprocess.transforms[1],
                    # already tensor, but want float
                    T.ConvertImageDtype(torch.float),
                    # normalize with CLIP mean, std
                    preprocess.transforms[4],
                ]
            )
            # expected output: C x H x W (np.float32)

            self.backbone = model.visual

            if self.rgb and self.depth:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048,)  # type: Tuple
            elif pooling == "none":
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif pooling == "avgpool":
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048,)
            else:
                self.output_shape = (1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self.rgb:
            rgb_observations = observations["rgb"]
            rgb_observations = rgb_observations.permute(
                0, 3, 1, 2
            )  # BATCH x CHANNEL x HEIGHT X WIDTH
            rgb_observations = torch.stack(
                [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            rgb_x = self.backbone(rgb_observations).float()
            cnn_input.append(rgb_x)

        if self.depth:
            depth_observations = observations["depth"][
                ..., 0
            ]  # [BATCH x HEIGHT X WIDTH]
            ddd = torch.stack(
                [depth_observations] * 3, dim=1
            )  # [BATCH x 3 x HEIGHT X WIDTH]
            ddd = torch.stack(
                [
                    self.preprocess(
                        TF.convert_image_dtype(depth_map, torch.uint8)
                    )
                    for depth_map in ddd
                ]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            depth_x = self.backbone(ddd).float()
            cnn_input.append(depth_x)

        if self.rgb and self.depth:
            x = F.adaptive_avg_pool2d(cnn_input[0] + cnn_input[1], 1)
            x = x.flatten(1)
        else:
            x = torch.cat(cnn_input, dim=1)

        return x


class GazePointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    SEG_MASKS = "segmentation_masks"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        no_downscaling: bool = False,
        ovrl: bool = False,
        use_augmentations: bool = False,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test
        self.segmentation = None
        
        self.masks = torch.zeros((32, 160, 120, 2))  #to change to the batch size
        self.segmentation = YOLOPerception(            
            sem_gpu_id=0,
            verbose=False,
            confidence_threshold=0.05)
        
        logger.info(
            "YOLO number of parameters: {}".format(
                sum(param.numel() for param in self.segmentation.model.parameters())
            )
        )
        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                ObjectCategorySensor.cls_uuid,
                StartReceptacleSensor.cls_uuid,
                GoalReceptacleSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys and k!="head_rgb" and k!="yolo_object_sensor" and k!="yolo_start_receptacle_sensor"]

        self._fuse_keys_1d: List[str] = [
            k
            for k in fuse_keys
            if len(observation_space.spaces[k].shape) == 1 and "third" not in k
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )
        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if ObjectCategorySensor.cls_uuid in observation_space.spaces:
            self._n_rearrange_obj_categories = (
                int(
                    observation_space.spaces[
                        ObjectCategorySensor.cls_uuid
                    ].high[0]
                )
                + 1
            )
            self.rearrange_obj_categories_embedding = nn.Embedding(
                self._n_rearrange_obj_categories, 32
            )
            rnn_input_size += 32

        if StartReceptacleSensor.cls_uuid in observation_space.spaces:
            self._n_start_receptacles = (
                int(
                    observation_space.spaces[
                        StartReceptacleSensor.cls_uuid
                    ].high[0]
                )
                + 1
            )
            self.start_receptacles_embedding = nn.Embedding(
                self._n_start_receptacles, 32
            )
            rnn_input_size += 32

        if GoalReceptacleSensor.cls_uuid in observation_space.spaces:
            self._n_goal_receptacles = (
                int(
                    observation_space.spaces[
                        GoalReceptacleSensor.cls_uuid
                    ].high[0]
                )
                + 1
            )
            self.goal_receptacles_embedding = nn.Embedding(
                self._n_goal_receptacles, 32
            )
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observation_space.spaces:
                goal_observation_space = spaces.Dict(
                    {"rgb": observation_space.spaces[uuid]}
                )
                goal_visual_encoder = GazeResNetEncoder(
                    goal_observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(
                        resnet_gn if ovrl else resnet, backbone
                    ),
                    no_downscaling=no_downscaling,
                    use_augmentations=use_augmentations,
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)

                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(goal_visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )
                setattr(self, f"{uuid}_fc", goal_visual_fc)

                rnn_input_size += hidden_size

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )

        if backbone.startswith("resnet50_clip"):
            self.visual_encoder = ResNetCLIPEncoder(
                observation_space
                if not force_blind_policy
                else spaces.Dict({}),
                pooling="avgpool" if "avgpool" in backbone else "attnpool",
            )
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Linear(
                        self.visual_encoder.output_shape[0], hidden_size
                    ),
                    nn.ReLU(True),
                )
        else:
            self.visual_encoder = GazeResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet_gn if ovrl else resnet, backbone),
                no_downscaling=no_downscaling,
                use_augmentations=use_augmentations,
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        torch.cuda.empty_cache()
        x = []
        aux_loss_state = {}
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                GazePointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                visual_feats = observations[
                    GazePointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if ObjectCategorySensor.cls_uuid in observations:
            object_goal = observations[ObjectCategorySensor.cls_uuid].long()
            x.append(
                self.rearrange_obj_categories_embedding(object_goal).squeeze(
                    dim=1
                )
            )

        if StartReceptacleSensor.cls_uuid in observations:
            start_receptacle = observations[
                StartReceptacleSensor.cls_uuid
            ].long()
            x.append(
                self.start_receptacles_embedding(start_receptacle).squeeze(
                    dim=1
                )
            )

        if GoalReceptacleSensor.cls_uuid in observations:
            goal_receptacle = observations[
                GoalReceptacleSensor.cls_uuid
            ].long()
            x.append(
                self.goal_receptacles_embedding(goal_receptacle).squeeze(dim=1)
            )

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observations:
                goal_image = observations[uuid]

                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})

                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out
        torch.cuda.empty_cache()
        return out, rnn_hidden_states, aux_loss_state
