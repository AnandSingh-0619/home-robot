#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.gym.gym_definitions import make_gym_from_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_info,
    extract_scalars_from_infos,
)
from habitat_baselines.utils.timing import g_timer
from habitat_uncertainity.utils.YOLO_pred import YOLOPerception as YOLO_pred
from habitat_baselines import PPOTrainer
from habitat.core.logging import logger
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

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

@baseline_registry.register_trainer(name="ddppo_yolo")
@baseline_registry.register_trainer(name="ppo_yolo")
class PPOyoloTrainer(PPOTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    envs: VectorEnv
    _env_spec: Optional[EnvironmentSpec]

    def __init__(self, config=None):
        super().__init__(config)

    
    def _init_train(self, resume_state=None):
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        self._add_preemption_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique seeds
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # remove the non scalar measures from the measures since they can only be used in
        # evaluation
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(
                        non_scalar_metric_root
                    )
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                    )

        self._init_envs()

        if torch.cuda.is_available():
            self.device = torch.device(
                "cuda", self.config.habitat_baselines.torch_gpu_id
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        logger.add_filehandler(self.config.habitat_baselines.log_file)

        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.updater.init_distributed(find_unused_params=False)  # type: ignore
        self._agent.post_init()

        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            assert (
                self._encoder is not None
            ), "Visual encoder is not specified for this actor"
            with inference_mode():
                batch[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self._encoder(batch)

        self._agent.rollouts.insert_first_observations(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )
        self._segmentation = YOLO_pred()
        self.t_start = time.time()

    
    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=self._rank0_keys
            )
            for k, v_k in extracted_infos.items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )
            #YOLO Detection with Mobile SAM segmentations
            # profiling_wrapper.range_push("yolo_detector_step")
            with g_timer.avg_time("trainer.yolo_detector_step"):
                segment_masks = self._segmentation.predict(batch)
            # profiling_wrapper.range_pop() 
            if("object_segmentation" in batch):
                filtered_masks = []
                class_ids = batch["yolo_object_sensor"].cpu().numpy().flatten()
                class_ids_expanded = class_ids[:, np.newaxis, np.newaxis, np.newaxis]
                filtered_masks = np.where(segment_masks == class_ids_expanded, 1, 0)
                sigma = 8.0  # Sigma value for the Gaussian filter
                masks = []
                for mask in filtered_masks:
                    mask_binary = mask.astype(bool)
                    object_pixels = np.where(mask_binary)
                    if len(object_pixels[0]) == 0:
                        masks.append(mask)
                        continue
                    center_coordinate = (int(np.mean(object_pixels[1])), int(np.mean(object_pixels[0])))
                    mask = np.zeros_like(mask, dtype=np.float32)
                    x, y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
                    distance_matrix = np.sqrt((x - center_coordinate[1])**2 + (y - center_coordinate[0])**2)
                    heatmap = np.exp(-(distance_matrix**2) / (2 * sigma**2))
                    heatmap /= np.max(heatmap)
                    heatmap = np.expand_dims(heatmap, axis=2)
                    masks.append(heatmap)
                    mask = np.expand_dims(mask, axis=2)
                    heatmap = np.expand_dims(heatmap, axis=2)
                    # plt.imsave('mask3.png', mask, cmap='hot')
                    # plt.imsave('mask4.png', heatmap, cmap='hot')
                filtered_masks = np.array(masks)
                
                batch["object_segmentation"] = torch.tensor(filtered_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device())))

            if("start_recep_segmentation" in batch):
                filtered_masks = []
                class_ids = batch["yolo_start_receptacle_sensor"].cpu().numpy().flatten()
                class_ids_expanded = class_ids[:, np.newaxis, np.newaxis, np.newaxis]
                filtered_masks = np.where(segment_masks == class_ids_expanded, 1, 0)
                sigma = 8.0  # Sigma value for the Gaussian filter
                masks = []
                for mask in filtered_masks:
                    mask_binary = mask.astype(bool)
                    object_pixels = np.where(mask_binary)
                    if len(object_pixels[0]) == 0:
                        masks.append(mask)
                        continue
                    center_coordinate = (int(np.mean(object_pixels[1])), int(np.mean(object_pixels[0])))
                    mask = np.zeros_like(mask, dtype=np.float32)
                    x, y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
                    distance_matrix = np.sqrt((x - center_coordinate[1])**2 + (y - center_coordinate[0])**2)
                    heatmap = np.exp(-(distance_matrix**2) / (2 * sigma**2))
                    heatmap /= np.max(heatmap)
                    heatmap = np.expand_dims(heatmap, axis=2)
                    masks.append(heatmap)
                    mask = np.expand_dims(mask, axis=2)
                    heatmap = np.expand_dims(heatmap, axis=2)
                    # plt.imsave('mask5.png', mask, cmap='hot')
                    # plt.imsave('mask6.png', heatmap, cmap='hot')
                filtered_masks = np.array(masks)
                
                batch["start_recep_segmentation"] = torch.tensor(filtered_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device())))
            
            if("goal_recep_segmentation" in batch):
                filtered_masks = []
                class_ids = batch["yolo_goal_receptacle_sensor"].cpu().numpy().flatten()
                class_ids_expanded = class_ids[:, np.newaxis, np.newaxis, np.newaxis]
                filtered_masks = np.where(segment_masks == class_ids_expanded, 1, 0)
                # for class_id in class_ids:
                #     if (segment_masks == class_id).any():
                #         logger.info(f"Receptacle Detected: {CLASSES[class_id]}")
                #     else:
                #         logger.info(f"No receptacle Detected")
                batch["goal_recep_segmentation"] = torch.tensor(filtered_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device())))
            
            if("ovmm_nav_goal_segmentation" in batch):
                    
                if batch["ovmm_nav_goal_segmentation"].shape[3] == 2:
                    filtered_masks = []
                    class_ids = batch["yolo_object_sensor"].cpu().numpy().flatten()
                    class_ids_expanded = class_ids[:, np.newaxis, np.newaxis, np.newaxis]
                    filtered_masks = np.where(segment_masks == class_ids_expanded, 1, 0)
                    obs_k1 =torch.tensor(filtered_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device())))

                    filtered_masks = []
                    class_ids = batch["yolo_start_receptacle_sensor"].cpu().numpy().flatten()
                    class_ids_expanded = class_ids[:, np.newaxis, np.newaxis, np.newaxis]
                    filtered_masks = np.where(segment_masks == class_ids_expanded, 1, 0)
                    obs_k2 =torch.tensor(filtered_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device())))
                    batch["ovmm_nav_goal_segmentation"] = torch.cat((obs_k1, obs_k2), dim=3)

                else:
                    filtered_masks = []
                    class_ids = observations["yolo_goal_receptacle_sensor"].cpu().numpy().flatten()
                    class_ids_expanded = class_ids[:, np.newaxis, np.newaxis, np.newaxis]
                    filtered_masks = np.where(segment_masks == class_ids_expanded, 1, 0)
                    batch["ovmm_nav_goal_segmentation"] =torch.tensor(filtered_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device())))
            
            if("receptacle_segmentation_sensor" in batch):
                filtered_masks = []
                filtered_masks = np.where(segment_masks > 127, 1, 0)
                batch["receptacle_segmentation_sensor"] = torch.tensor(filtered_masks, device=torch.device('cuda:{}'.format(torch.cuda.current_device())))
            
        if self._is_static_encoder:
            with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                batch[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self._encoder(batch)

        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    