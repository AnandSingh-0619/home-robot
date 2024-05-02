# Habitat Uncertainity
This project extends the Home-Robot (Habitat-OVMM) framework by integrating the YOLO object detection system. This integration aims to improve the robot's ability to perform manipulation tasks in real environments.

[README](https://github.com/facebookresearch/home-robot/tree/main/projects/habitat_ovmm) 
The setup involves setting up the Home-Robot framework up to step 2 from the Habitat OVMM README.

## Table of contents
   1. [Change Log](#changes-for-integrating-model)
   2. [Training DD-PPO skills](#training-dd-ppo-skills)

## Change Log 

### Updated Head Panoptic Sensor with YOLO ([YOLO_pred](https://github.com/AnandSingh-0619/home-robot/blob/yolo-sam/projects/habitat_uncertainity/utils/YOLO_pred.py))

1. Replaced Ground Truth segmentation with a combination of Detector (YOLO-Open World) and Segmentation model (MobileSAM) for improved performance and functionality in real world.

2. The new setup provides pixel-wise semantic segmentation of the agent's observation.

### New Sensors ([sensors](https://github.com/AnandSingh-0619/home-robot/blob/yolo-sam/projects/habitat_uncertainity/task/sensors.py))
1. YOLOObjectSensor: This sensor returns the target object class id in the current episode of training environment as an observation. It enables the agent to focus on detecting and interacting with specific objects relevant to the task.

2. YOLOStartReceptacleSensor & YOLOGoalReceptacleSensor: These sensors specifically return the start and goal receptacle class ids, aiding in task-oriented perception. They help the agent identify and navigate to receptacles essential for task completion.

These sensors play a crucial role in improving segmentation in the environment by returning the target object/receptacle class ID for filtering masks, a functionality not provided by existing sensors. They allow the agent to focus on specific objects or receptacles relevant to the task by filtering the masks obtained from the YOLO perception system.  

### New Trainer: [ppo_trainer_yolo](https://github.com/AnandSingh-0619/home-robot/blob/yolo-sam/projects/habitat_uncertainity/trainers/ppo_trainer_yolo.py)
The _collect_environment_result method in the trainer gathers observations, rewards, and other data from the environment. It utilizes information from the new YOLOObjectSensor, YOLOStartReceptacleSensor, and YOLOGoalReceptacleSensor sensors to filter masks for various segmentation tasks, including object segmentation, start receptacle segmentation, goal receptacle segmentation, and navigation goal segmentation. Additionally, it caches predictions from the frozen YOLO model in the rollout storage to reduce compute costs during new policy action evaluation.

## Training DD-PPO skills

First setup data directory
```
cd /path/to/home-robot/src/third_party/habitat-lab/

# create soft link to data/ directory
ln -s /path/to/home-robot/data data
```
Base config file used for trainig is [`yolo_rl_skill`](https://github.com/AnandSingh-0619/home-robot/blob/yolo-sam/projects/habitat_uncertainity/config/yolo_rl_skill.yaml) 
To train on a single machine use the following script:
```
#/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH
conda deactivate
conda activate home-robot
cd ~/flash/home-robot 

srun python -um habitat_uncertainity.run \
    --exp-config=projects/habitat_uncertainity/config/yolo_rl_skill.yaml \
    --run-type=train \
    benchmark.ovmm=<skill_name> \
    habitat_baselines.num_environments=32 \
    habitat_baselines.load_resume_state_config=True \
```
Here `<skill_name>` should be one of `gaze`, `place`, `nav_to_obj` or `nav_to_rec`.
### Note: Currently the code has been setup and tested with the skill `gaze`
To run on a cluster with SLURM using distributed training run the following script. While this is not necessary, if you have access to a cluster, it can significantly speed up training. To run on multiple machines in a SLURM cluster run the following script: change `#SBATCH --ntasks-per-node $NUM_OF_GPUS` and `$SBATCH --gpus $NUM_OF_GPUS` to specify the number of GPUS to use per requested machine.

```
#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=logs.ddppo-ver-%j.out
#SBATCH --error=logs.ddppo-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --partition=overcap
#SBATCH --gpus a40:4
#SBATCH --qos="long"
#SBATCH --requeue
#SBATCH --signal=USR1@100

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH
conda deactivate
conda activate home-robot
cd ~/flash/home-robot 

srun python -um habitat_uncertainity.run \
    --exp-config=projects/habitat_uncertainity/config/yolo_rl_skill.yaml \
    --run-type=train \
    benchmark.ovmm=<skill_name> \
    habitat_baselines.num_environments=32 \
    habitat_baselines.load_resume_state_config=True \
```


Refer to the provided  [job.sh](https://github.com/AnandSingh-0619/home-robot/blob/yolo-sam/job.sh) for detailed cluster job submission instructions.

## Tensorboard Log
Sample performance comparison showing FPS.

![fps_comparison](https://drive.google.com/uc?export=view&id=1WayMUi1FTZWsDtTC5WXsGv3EtbDt6iUA)

## Disclaimer
This project is currently under development, and the code is primarily tested with the gaze skill.