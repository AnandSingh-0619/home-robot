
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
from habitat_uncertainity.models.yoloPointNavResNetPolicy import yoloPointNavResNetPolicy, yoloResNetEncoder
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

    with read_write(config):
        edit_config(config, args)

    # YOLO_pred(
    #         sem_gpu_id =0,
    #         verbose= False
    #         )
    execute_exp(config, "train")

def edit_config(config, args):
    # config.habitat.task.lab_sensors.pop("object_segmentation_sensor")
    # config.habitat.task.lab_sensors.pop("start_recep_segmentation_sensor")
    # if not hasattr(config.habitat.task.lab_sensors, "yolo_object_segmentation"):
    #     config.habitat.task.lab_sensors.update(
    #         {"yolo_object_segmentation": YOLO_ObjectSegmentationSensorConfig()}
    #     )
    if args.debug:
        assert osp.isdir(os.environ["JUNK"]), (
            "Environment variable directory $JUNK does not exist "
            f"(Current value: {os.environ['JUNK']})"
        )

        # Remove resume state in junk if training, so we don't resume from it
        resume_state_path = osp.join(os.environ["JUNK"], ".habitat-resume-state.pth")
        if args.run_type == "train" and osp.isfile(resume_state_path):
            print(
                "Removing junk resume state file:",
                osp.abspath(resume_state_path),
            )
            os.remove(resume_state_path)

        config.habitat_baselines.tensorboard_dir = os.environ["JUNK"]
        config.habitat_baselines.video_dir = os.environ["JUNK"]
        config.habitat_baselines.checkpoint_folder = os.environ["JUNK"]
        config.habitat_baselines.log_file = osp.join(os.environ["JUNK"], "junk.log")
        config.habitat_baselines.load_resume_state_config = False

    if args.debug_datapath:
        # Only load one scene for faster debugging
        scenes = "1UnKg1rAb8A" if args.run_type == "train" else "4ok3usBNeis"
        config.habitat.dataset.content_scenes = [scenes]

    if args.single_env:
        config.habitat_baselines.num_environments = 1

    if args.eval_analysis:
        from habitat.config.default_structured_configs import (
            HabitatSimSemanticSensorConfig,
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig(height=640, width=360)}
        )


if __name__ == "__main__":
    main()