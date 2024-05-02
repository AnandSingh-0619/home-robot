
#!/usr/bin/env python3

import argparse
import glob
import os
import os.path as osp
import torch
from habitat import get_config
from habitat.config import read_write
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_uncertainity.config import HabitatConfigPlugin

from habitat_baselines.run import execute_exp
from habitat_uncertainity.task.sensors import YOLOObjectSensor, YOLOStartReceptacleSensor, YOLOGoalReceptacleSensor
from habitat_uncertainity.utils.YOLO_pred import YOLOPerception as YOLO_pred
from habitat_uncertainity.trainers.ppo_trainer_yolo import PPOyoloTrainer

def register_plugins():
    register_hydra_plugin(HabitatConfigPlugin)


def main():
    """Builds upon the habitat_baselines.run.main() function to add more flags
    for convenience."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run-type",
        "-r",
        default='train',
        choices=["train", "eval"],
        required=False,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        "-e",
        default='projects/habitat_uncertainity/config/yolo_rl_skill.yaml',
        type=str,
        required=False,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Saves files to $JUNK directory and ignores resume state.",
    )
    parser.add_argument(
        "--single-env",
        "-s",
        action="store_true",
        help="Sets num_environments=1.",
    )
    parser.add_argument(
        "--debug-datapath",
        "-p",
        action="store_true",
        help="Uses faster-to-load $OVON_DEBUG_DATAPATH episode dataset for debugging.",
    )

    parser.add_argument(
        "--checkpoint-config",
        "-c",
        action="store_true",
        help=(
            "If set, checkpoint's config will be used, but overrides WILL be "
            "applied. Does nothing when training; meant for using ckpt config + "
            "overrides for eval."
        ),
    )
    parser.add_argument(
        "--text-goals",
        "-t",
        action="store_true",
        help="If set, only CLIP text goals will be used for evaluation.",
    )
    parser.add_argument(
        "--eval-analysis",
        "-v",
        action="store_true",
        help="If set, add semantic sensor for evaluation.",
    )
    parser.add_argument(
        "opts",
        default="habitat_baselines.num_environments=4",
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # Register custom hydra plugin
    register_plugins()

    config = get_config(args.exp_config, args.opts)

    execute_exp(config, "train")


if __name__ == "__main__":
    main()