from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.rollout_storage import (  # noqa: F401.
    RolloutStorage,
)
from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.policy import NetPolicy, Policy
from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.ppo.updater import Updater
from habitat_baselines.rl.ppo.single_agent_access_mgr import SingleAgentAccessMgr
from habitat_uncertainty.models.GazePointNavResNetPolicy import GazePointNavResNetNet
if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_agent_access_mgr
class SingleAgentAccessManager(SingleAgentAccessMgr):
   
    def _create_storage(
        self,
        num_envs: int,
        env_spec: EnvironmentSpec,
        actor_critic: Policy,
        policy_action_space: spaces.Space,
        config: "DictConfig",
        device,
    ) -> Storage:
        """
        Default behavior for setting up and initializing the rollout storage.
        """

        obs_space = get_rollout_obs_space(
            env_spec.observation_space, actor_critic, config
        )
        ppo_cfg = config.habitat_baselines.rl.ppo
        rollouts = baseline_registry.get_storage(
            config.habitat_baselines.rollout_storage_name
        )(
            numsteps=ppo_cfg.num_steps,
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=policy_action_space,
            actor_critic=actor_critic,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
        )
        rollouts.to(device)
        return rollouts
    
    def load_state_dict(self, state: Dict) -> None:
        self._actor_critic.load_state_dict(state["state_dict"])
        if self._updater is not None:
            self._updater.load_state_dict(state)
            if "lr_sched_state" in state:
                # self._lr_scheduler.load_state_dict(state["lr_sched_state"])

                lr_sched_state = state["lr_sched_state"]
                if isinstance(lr_sched_state, tuple):
                    lr_sched_state = lr_sched_state[0]  # Assuming the relevant dictionary is at index 0
                self._lr_scheduler.load_state_dict(lr_sched_state)
                

   
def get_rollout_obs_space(obs_space, actor_critic, config):
    """
    Helper to get the observation space for the rollout storage when using a
    frozen visual encoder.
    """

    if not config.habitat_baselines.rl.ddppo.train_encoder:
        encoder = actor_critic.visual_encoder
        obs_space = spaces.Dict(
            {
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=encoder.output_shape,
                    dtype=np.float32,
                ),
                **obs_space.spaces,
            }
        )
    if not config.habitat_baselines.rl.ddppo.train_detector:
        obs_space = spaces.Dict(
            {
                GazePointNavResNetNet.SEG_MASKS: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=[160, 120, 2],
                    dtype=np.float32,
                ),
                **obs_space.spaces,
            }
        )
    return obs_space

