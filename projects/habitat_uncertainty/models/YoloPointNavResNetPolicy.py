#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from gym import spaces
from habitat.core.logging import logger
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
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from habitat_baselines.rl.ddppo.transforms import ShiftAndJitterTransform
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from habitat_baselines.utils.timing import g_timer
from PIL import Image
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from ultralytics import YOLO

from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations

PARENT_DIR = Path(__file__).resolve().parent
MOBILE_SAM_CHECKPOINT_PATH = str(PARENT_DIR / "pretrained_wt" / "mobile_sam.pt")
CLASSES = [
    "action_figure",
    "android_figure",
    "apple",
    "backpack",
    "baseballbat",
    "basket",
    "basketball",
    "bath_towel",
    "battery_charger",
    "board_game",
    "book",
    "bottle",
    "bowl",
    "box",
    "bread",
    "bundt_pan",
    "butter_dish",
    "c-clamp",
    "cake_pan",
    "can",
    "can_opener",
    "candle",
    "candle_holder",
    "candy_bar",
    "canister",
    "carrying_case",
    "casserole",
    "cellphone",
    "clock",
    "cloth",
    "credit_card",
    "cup",
    "cushion",
    "dish",
    "doll",
    "dumbbell",
    "egg",
    "electric_kettle",
    "electronic_cable",
    "file_sorter",
    "folder",
    "fork",
    "gaming_console",
    "glass",
    "hammer",
    "hand_towel",
    "handbag",
    "hard_drive",
    "hat",
    "helmet",
    "jar",
    "jug",
    "kettle",
    "keychain",
    "knife",
    "ladle",
    "lamp",
    "laptop",
    "laptop_cover",
    "laptop_stand",
    "lettuce",
    "lunch_box",
    "milk_frother_cup",
    "monitor_stand",
    "mouse_pad",
    "multiport_hub",
    "newspaper",
    "pan",
    "pen",
    "pencil_case",
    "phone_stand",
    "picture_frame",
    "pitcher",
    "plant_container",
    "plant_saucer",
    "plate",
    "plunger",
    "pot",
    "potato",
    "ramekin",
    "remote",
    "salt_and_pepper_shaker",
    "scissors",
    "screwdriver",
    "shoe",
    "soap",
    "soap_dish",
    "soap_dispenser",
    "spatula",
    "spectacles",
    "spicemill",
    "sponge",
    "spoon",
    "spray_bottle",
    "squeezer",
    "statue",
    "stuffed_toy",
    "sushi_mat",
    "tape",
    "teapot",
    "tennis_racquet",
    "tissue_box",
    "toiletry",
    "tomato",
    "toy_airplane",
    "toy_animal",
    "toy_bee",
    "toy_cactus",
    "toy_construction_set",
    "toy_fire_truck",
    "toy_food",
    "toy_fruits",
    "toy_lamp",
    "toy_pineapple",
    "toy_rattle",
    "toy_refrigerator",
    "toy_sink",
    "toy_sofa",
    "toy_swing",
    "toy_table",
    "toy_vehicle",
    "tray",
    "utensil_holder_cup",
    "vase",
    "video_game_cartridge",
    "watch",
    "watering_can",
    "wine_bottle",
    "bathtub",
    "bed",
    "bench",
    "cabinet",
    "chair",
    "chest_of_drawers",
    "couch",
    "counter",
    "filing_cabinet",
    "hamper",
    "serving_cart",
    "shelves",
    "shoe_rack",
    "sink",
    "stand",
    "stool",
    "table",
    "toilet",
    "trunk",
    "wardrobe",
    "washer_dryer",
]

if TYPE_CHECKING:
    from omegaconf import DictConfig

try:
    import clip
except ImportError:
    clip = None


@baseline_registry.register_policy
class YoloPointNavResNetPolicy(NetPolicy):
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
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            PointNavResNetNet(
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
                use_augmentations=getattr(policy_config, "use_augmentations", False),
                policy_config=policy_config,
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
                ((k, v) for k, v in observation_space.items() if k not in ignore_names)
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
        yolo_model_id: Optional[str] = "yolov8s-world.pt",
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = 0.03,
        save_images: Optional[bool] = False,
        save_path: "str" = "",
    ):
        """Loads a YOLO model for object detection and instance segmentation

        Arguments:
            yolo_model_id: yolo model to be used for detection
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """

        vocab = CLASSES
        with torch.no_grad():
            self.model = YOLO(model=yolo_model_id)
            self.model.set_classes(vocab)
        # Freeze the YOLO model's parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.confidence_threshold = confidence_threshold
        self.model.cuda()
        self.save_images = save_images
        self.save_path = save_path
        self.image_counter = 0
        torch.cuda.empty_cache()
        self.output_shape = [160, 120, 3]
        if verbose:
            logger.info(f"Loading YOLO model: {yolo_model_id} ")
            total_yolo_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Total number of parameters in YOLO model: {total_yolo_params}"
            )

    def create_binary_mask(self, height, width, boxes):
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        if len(boxes) > 0:
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box)
                mask[y_min : y_max + 1, x_min : x_max + 1] = 1
        return mask

    def update_receptacle_mask(
        self, all_rec_mask, rec_boxes, sorted_class_ids, height, width
    ):

        for idx, class_id in enumerate(sorted_class_ids):
            rec_gaussian_mask_id = class_id - 128 + 1  # Adjusting for 1-based class IDs
            rec_label_mask = self.label_all_pixels(
                height, width, [rec_boxes[idx]], class_id=rec_gaussian_mask_id
            )
            mask_condition = all_rec_mask == 0
            all_rec_mask = np.where(mask_condition, rec_label_mask, all_rec_mask)

        return all_rec_mask

    def label_all_pixels(self, height, width, boxes, class_id):
        x_min, y_min, x_max, y_max = map(int, boxes[0])
        mask = np.zeros((height, width), dtype=np.float64)
        mask[y_min : y_max + 1, x_min : x_max + 1] = class_id
        return np.expand_dims(mask, axis=-1)

    def get_semantic_vis(self, semantic_map):
        semantic_map = semantic_map.squeeze(axis=-1)
        height, width = semantic_map.shape
        semantic_map_vis = np.zeros((height, width, 3), dtype=np.uint8)

        # Set the color for the detected objects (e.g., red)
        semantic_map_vis[semantic_map == 1] = [255, 0, 0]  # Red color
        return semantic_map_vis

    def overlay_semantics(
        self, rgb_image, object_semantic_vis, receptacle_semantic_vis
    ):
        object_semantic_vis_pil = Image.fromarray(object_semantic_vis)
        receptacle_semantic_vis_pil = Image.fromarray(receptacle_semantic_vis)

        object_mask = (np.array(object_semantic_vis_pil) != 0).astype(np.uint8) * 255
        receptacle_mask = (np.array(receptacle_semantic_vis_pil) != 0).astype(
            np.uint8
        ) * 255

        object_mask_pil = Image.fromarray(object_mask).convert("L")
        receptacle_mask_pil = Image.fromarray(receptacle_mask).convert("L")

        rgb_image.paste(object_semantic_vis_pil, mask=object_mask_pil)
        rgb_image.paste(receptacle_semantic_vis_pil, mask=receptacle_mask_pil)

        combined_vis = np.asarray(rgb_image)

        return combined_vis

    def get_combined_semantic_vis(self, rgb_image, object_mask, receptacle_mask):
        rgb_image_pil = Image.fromarray(rgb_image)

        object_semantic_vis = self.get_semantic_vis(object_mask)
        receptacle_semantic_vis = self.get_semantic_vis(receptacle_mask)

        combined_vis = self.overlay_semantics(
            rgb_image_pil, object_semantic_vis, receptacle_semantic_vis
        )

        return combined_vis

    def concatenate_images(self, *images):
        min_height = min(img.shape[0] for img in images)
        resized_images = [cv2.resize(img, (img.shape[1], min_height)) for img in images]
        concatenated_image = np.hstack(resized_images)
        return concatenated_image

    def save_image_sequence(self, image, base_path="image_dump/18"):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        file_path = os.path.join(base_path, f"image_{self.image_counter:05d}.png")
        cv2.imwrite(file_path, image)
        self.image_counter += 1

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

        """

        torch.cuda.empty_cache()
        nms_threshold = 0.8

        images_tensor = obs["head_rgb"]
        obj_class_ids = obs["yolo_object_sensor"].cpu().numpy().flatten()
        rec_class_ids = obs["yolo_start_receptacle_sensor"].cpu().numpy().flatten()
        depth = obs["head_depth"].cpu()

        batch_size = images_tensor.shape[0]
        images = [images_tensor[i].cpu().numpy() for i in range(images_tensor.size(0))]

        start_receptacle_index = CLASSES.index("bathtub")
        receptacle_class_indices = list(range(start_receptacle_index, len(CLASSES)))
        search_list = np.concatenate((obj_class_ids, receptacle_class_indices)).tolist()

        height, width, _ = images[0].shape
        results = list(
            self.model(
                images,
                classes=search_list,
                conf=self.confidence_threshold,
                iou=nms_threshold,
                stream=True,
                verbose=False,
            )
        )
        obj_semantic_masks = []
        rec_semantic_masks = []
        all_rec_masks = []

        for idx, result in enumerate(results):
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            input_boxes = result.boxes.xyxy.cpu().numpy()

            image = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
            obj_semantic_mask = np.zeros((640, 480, 1))
            rec_semantic_mask = np.zeros((640, 480, 1))
            all_rec_mask = np.zeros((640, 480, 1))

            if class_ids.size != 0:
                obj_mask_idx = np.isin(class_ids, obj_class_ids[idx])
                rec_mask_idx = np.isin(class_ids, rec_class_ids[idx])

                obj_boxes = input_boxes[obj_mask_idx]
                rec_boxes = input_boxes[rec_mask_idx]
                all_rec_class_ids = class_ids[
                    np.isin(class_ids, receptacle_class_indices)
                ]
                all_rec_boxes = input_boxes[
                    np.isin(class_ids, receptacle_class_indices)
                ]

                obj_semantic_mask = self.create_binary_mask(height, width, obj_boxes)
                rec_semantic_mask = self.create_binary_mask(height, width, rec_boxes)
                all_rec_mask = np.zeros((height, width, 1), dtype=np.float64)

                rec_boxes = []  # Collect all receptacle boxes
                depth_class_boxes = (
                    []
                )  # Collect depth values corresponding to each center pixel

                if all_rec_boxes.size != 0:

                    x_centers = (
                        (all_rec_boxes[:, 0] + all_rec_boxes[:, 2]) // 2
                    ).astype(int)
                    y_centers = (
                        (all_rec_boxes[:, 1] + all_rec_boxes[:, 3]) // 2
                    ).astype(int)
                    depth_values = depth[idx, y_centers, x_centers, 0]

                    depth_class_boxes = np.vstack(
                        (depth_values, all_rec_class_ids, all_rec_boxes.T)
                    ).T

                    sorted_indices = np.argsort(depth_class_boxes[:, 0])
                    sorted_depth_class_boxes = depth_class_boxes[sorted_indices]
                    sorted_depth_values, sorted_class_ids, sorted_rec_boxes = (
                        sorted_depth_class_boxes[:, 0],
                        sorted_depth_class_boxes[:, 1].astype(int),
                        sorted_depth_class_boxes[:, 2:].astype(int),
                    )

                    all_rec_mask = self.update_receptacle_mask(
                        all_rec_mask, sorted_rec_boxes, sorted_class_ids, height, width
                    )

                all_rec_mask = np.clip(all_rec_mask, 0, 255)
                if self.save_images:
                    normalized_mask = (all_rec_mask * 255 / all_rec_mask.max()).astype(
                        np.uint8
                    )
                    color_map = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)
                    alpha = 0.5  # Transparency factor
                    all_recep_image = cv2.addWeighted(
                        image, 1 - alpha, color_map, alpha, 0
                    )

                    combined_vis = self.get_combined_semantic_vis(
                        image, obj_semantic_mask, rec_semantic_mask
                    )
                    concatenated_image = self.concatenate_images(
                        combined_vis, color_map
                    )

                    cv2.putText(
                        concatenated_image,
                        f"Target object: {CLASSES[obj_class_ids[idx]]}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        concatenated_image,
                        f"Target Receptacle: {CLASSES[rec_class_ids[idx]]}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 0, 0),
                        2,
                    )
                    self.save_image_sequence(concatenated_image, self.save_path)

            obj_semantic_mask = cv2.resize(
                obj_semantic_mask, (120, 160), interpolation=cv2.INTER_NEAREST
            )
            rec_semantic_mask = cv2.resize(
                rec_semantic_mask, (120, 160), interpolation=cv2.INTER_NEAREST
            )

            obj_semantic_mask = np.expand_dims(obj_semantic_mask, axis=-1)
            rec_semantic_mask = np.expand_dims(rec_semantic_mask, axis=-1)

            all_rec_mask = cv2.resize(
                all_rec_mask, (120, 160), interpolation=cv2.INTER_NEAREST
            )
            all_rec_mask = np.expand_dims(all_rec_mask, axis=-1)

            obj_semantic_masks.append(obj_semantic_mask)
            rec_semantic_masks.append(rec_semantic_mask)
            all_rec_masks.append(np.array(all_rec_mask))

        torch.cuda.empty_cache()
        obj_semantic_masks = np.array(obj_semantic_masks)
        rec_semantic_masks = np.array(rec_semantic_masks)
        all_rec_masks = np.array(all_rec_masks)

        combined_masks = np.concatenate(
            (obj_semantic_masks, rec_semantic_masks, all_rec_masks), axis=-1
        )
        combined_masks = (
            torch.tensor(
                combined_masks,
                device=torch.device("cuda:{}".format(torch.cuda.current_device())),
            )
            .detach()
            .requires_grad_(False)
        )
        del obj_semantic_masks, rec_semantic_masks, results

        return combined_masks


class ResNetEncoder(nn.Module):
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
        policy_config: "DictConfig" = None,
    ):
        super().__init__()
        self.config = policy_config
        self.no_downscaling = no_downscaling
        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1
            and k != ImageGoalSensor.cls_uuid
            and k
            != "head_rgb"  # and k!= "ovmm_nav_goal_segmentation" and k!= "receptacle_segmentation"
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
            spatial_size_h = observation_space.spaces[self.visual_keys[1]].shape[0]
            spatial_size_w = observation_space.spaces[self.visual_keys[1]].shape[1]
            if not no_downscaling:
                spatial_size_h = spatial_size_h // 2
                spatial_size_w = spatial_size_w // 2
            self.backbone = make_backbone(self._n_input_channels, baseplanes, ngroups)

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial_h * final_spatial_w))
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

            self.output_shape = (
                self.backbone.final_channels,
                7,
                7,
            )
            rgb_keys = [k for k in observation_space.spaces if "rgb" in k]
            rgb_size = [observation_space.spaces[k].shape[:2] for k in rgb_keys]

            self.visual_transform = None
            if use_augmentations:
                self.visual_transform = ShiftAndJitterTransform(size=rgb_size[0])
                self.visual_transform.randomize_environments = False
            if policy_config.use_yolo:
                self.segmentation = YOLOPerception(
                    yolo_model_id=self.config.yolo_model_id,
                    sem_gpu_id=0,
                    verbose=self.config.verbose,
                    confidence_threshold=0.35,
                    save_images=self.config.save_images,
                    save_path=self.config.save_path,
                )
                self.masks = torch.zeros(
                    (self.config.num_env, 160, 120, 3)
                )  # Adjust to the batch size

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        if PointNavResNetNet.SEG_MASKS in observations:
            self.masks = observations[PointNavResNetNet.SEG_MASKS]
        else:
            with g_timer.avg_time("trainer.yolo_detector_step"):
                if self.config.use_yolo:
                    masks = self.segmentation.predict(observations)
                    self.masks = (
                        torch.tensor(
                            masks,
                            device=torch.device(
                                "cuda:{}".format(torch.cuda.current_device())
                            ),
                        )
                        .detach()
                        .requires_grad_(False)
                    )

        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # if np.random.random() < 0.5:
            if self.config.use_yolo:
                if k == "ovmm_nav_goal_segmentation":
                    obs_k = self.masks[..., 0:2]

                if k == "receptacle_segmentation":
                    obs_k = self.masks[..., 2:3]

                if k == "head_depth":
                    obs_k = (obs_k).cpu()
                    obs_k = obs_k.permute(0, 3, 1, 2)
                    depth_tensor_resized = F.interpolate(
                        obs_k, size=(160, 120), mode="nearest"
                    )
                    # depth_tensor_resized_np = depth_tensor_resized.cpu().numpy()
                    obs_k = depth_tensor_resized.permute(0, 2, 3, 1)
                    obs_k = torch.tensor(
                        obs_k,
                        device=torch.device(
                            "cuda:{}".format(torch.cuda.current_device())
                        ),
                    ).detach()

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2).float()
            obs_k = F.interpolate(
                obs_k, size=(120, 120), mode="bilinear", align_corners=False
            )
            if self.key_needs_rescaling[k] is not None:
                obs_k = obs_k.float() * self.key_needs_rescaling[k]  # normalize
            if self.visual_transform is not None:
                obs_k = self.visual_transform(obs_k)
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        if not self.no_downscaling:
            x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.adaptive_pool(x)

        return x


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        pooling="attnpool",
        policy_config: "DictConfig" = None,
    ):
        super().__init__()
        self.config = policy_config
        self.rgb = "head_rgb" in observation_space.spaces
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
            if not self.config.mask_rgb:
                rgb_observations = observations["head_rgb"]
            else:
                rgb_observations = observations["head_rgb"] * 0

            rgb_observations = rgb_observations.permute(
                0, 3, 1, 2
            )  # BATCH x CHANNEL x HEIGHT X WIDTH
            rgb_observations = torch.stack(
                [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            rgb_x = self.backbone(rgb_observations).float()
            cnn_input.append(rgb_x)

        if self.depth:
            depth_observations = observations["head_depth"][
                ..., 0
            ]  # [BATCH x HEIGHT X WIDTH]
            ddd = torch.stack(
                [depth_observations] * 3, dim=1
            )  # [BATCH x 3 x HEIGHT X WIDTH]
            ddd = torch.stack(
                [
                    self.preprocess(TF.convert_image_dtype(depth_map, torch.uint8))
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


class PointNavResNetNet(Net):
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
        policy_config: "DictConfig" = None,
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
            self.prev_action_embedding = nn.Linear(num_actions, self._n_prev_action)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test

        self.policy_config = policy_config

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
            fuse_keys = [
                k
                for k in fuse_keys
                if k not in goal_sensor_keys
                and k != "head_rgb"
                and k != "yolo_object_sensor"
                and k != "yolo_start_receptacle_sensor"
            ]

        self._fuse_keys_1d: List[str] = [
            k
            for k in fuse_keys
            if len(observation_space.spaces[k].shape) == 1 and "third" not in k
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0] for k in self._fuse_keys_1d
            )
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
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
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32

        if ObjectCategorySensor.cls_uuid in observation_space.spaces:
            self._n_rearrange_obj_categories = (
                int(observation_space.spaces[ObjectCategorySensor.cls_uuid].high[0]) + 1
            )
            self.rearrange_obj_categories_embedding = nn.Embedding(
                self._n_rearrange_obj_categories, 32
            )
            rnn_input_size += 32

        if StartReceptacleSensor.cls_uuid in observation_space.spaces:
            self._n_start_receptacles = (
                int(observation_space.spaces[StartReceptacleSensor.cls_uuid].high[0])
                + 1
            )
            self.start_receptacles_embedding = nn.Embedding(
                self._n_start_receptacles, 32
            )
            rnn_input_size += 32

        if GoalReceptacleSensor.cls_uuid in observation_space.spaces:
            self._n_goal_receptacles = (
                int(observation_space.spaces[GoalReceptacleSensor.cls_uuid].high[0]) + 1
            )
            self.goal_receptacles_embedding = nn.Embedding(self._n_goal_receptacles, 32)
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
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
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
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
                goal_visual_encoder = ResNetEncoder(
                    goal_observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(resnet_gn if ovrl else resnet, backbone),
                    no_downscaling=no_downscaling,
                    use_augmentations=use_augmentations,
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)

                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(goal_visual_encoder.output_shape), hidden_size),
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

        if self.policy_config.use_visual_encoder:
            self.visual_encoder = ResNetCLIPEncoder(
                observation_space if not force_blind_policy else spaces.Dict({}),
                pooling="none",
                policy_config=policy_config,
            )
        if self.policy_config.use_depth_encoder:
            self.depth_encoder = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet_gn if ovrl else resnet, backbone),
                no_downscaling=no_downscaling,
                use_augmentations=use_augmentations,
                normalize_visual_inputs=normalize_visual_inputs,
                policy_config=policy_config,
            )
        if self.policy_config.combine_encoders:
            final_channels = (
                self.visual_encoder.output_shape[0]
                if self.policy_config.use_visual_encoder
                else 0
            ) + (
                self.depth_encoder.output_shape[0]
                if self.policy_config.use_depth_encoder
                else 0
            )
        elif self.policy_config.use_visual_encoder:
            final_channels = self.visual_encoder.output_shape[0]
        elif self.policy_config.use_depth_encoder:
            final_channels = self.depth_encoder.output_shape[0]
        else:
            raise ValueError("At least one encoder (visual or depth) must be enabled.")

        spatial_dims = (
            self.visual_encoder.output_shape[1]
            if self.policy_config.use_visual_encoder
            else self.depth_encoder.output_shape[1]
        )
        # compression_channels = int(round(final_channels / (spatial_dims * spatial_dims)))
        compression_channels = 102
        self.compression = nn.Sequential(
            nn.Conv2d(
                final_channels,
                compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, compression_channels),
            nn.ReLU(True),
        )
        self.final_visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(compression_channels * spatial_dims * spatial_dims, hidden_size),
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
        return self.depth_encoder.is_blind

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
        x = []
        aux_loss_state = {}
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observations
            ):
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                if self.policy_config.use_visual_encoder:
                    visual_feats = self.visual_encoder(observations)

            if self.policy_config.use_depth_encoder:
                depth_feats = self.depth_encoder(observations)
            if self.policy_config.combine_encoders:
                concat_features = torch.cat((visual_feats, depth_feats), dim=1)
            elif self.policy_config.use_visual_encoder:
                concat_features = visual_feats
            elif self.policy_config.use_depth_encoder:
                concat_features = depth_feats

            concat_features = self.compression(concat_features)
            concat_features = self.final_visual_fc(concat_features)

            aux_loss_state["perception_embed"] = concat_features
            x.append(concat_features)

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
                assert goal_observations.shape[1] == 3, "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1]) * vertical_angle_sin,
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
                self.rearrange_obj_categories_embedding(object_goal).squeeze(dim=1)
            )

        if StartReceptacleSensor.cls_uuid in observations:
            start_receptacle = observations[StartReceptacleSensor.cls_uuid].long()
            x.append(self.start_receptacles_embedding(start_receptacle).squeeze(dim=1))

        if GoalReceptacleSensor.cls_uuid in observations:
            goal_receptacle = observations[GoalReceptacleSensor.cls_uuid].long()
            x.append(self.goal_receptacles_embedding(goal_receptacle).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

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
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state
