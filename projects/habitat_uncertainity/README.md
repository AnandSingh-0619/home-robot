# Habitat Uncertainity
Setup from [README](https://github.com/facebookresearch/home-robot/tree/main/projects/habitat_ovmm) until step 2 from habitat ovmm

## Table of contents
   1. [Change Log](#changes-for-integrating-model)
   2. [Training DD-PPO skills](#training-dd-ppo-skills)

## Change Log 

### Updated Head Panoptic Sensor

1. Replaced Ground Truth segmentation with a combination of Detector (YOLO-Open World) and Segmentation model (MobileSAM) for improved performance and functionality.
[YOLO_pred](https://github.com/AnandSingh-0619/home-robot/blob/yolo-sam/projects/habitat_uncertainity/utils/YOLO_pred.py)
2. The new setup provides pixel-wise semantic segmentation of the agent's observation.

### New Trainer: ppo_trainer_yolo

1. In the DDPPO algorithm, the agent collects observations in the rollout stage and stores the observations, rewards, and other relevant data.
2. The [ppo_trainer_yolo](https://github.com/AnandSingh-0619/home-robot/blob/yolo-sam/projects/habitat_uncertainity/trainers/ppo_trainer_yolo.py) has been created to handle these input observations from various sensors.
3. This trainer calculates the semantic mask and stores them in the required output as object segmentation sensor or receptacle segmentation sensor.


## Training DD-PPO skills

First setup data directory
```
cd /path/to/home-robot/src/third_party/habitat-lab/

# create soft link to data/ directory
ln -s /path/to/home-robot/data data
```

To train on a single machine use the following script:
```
#/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH
set -x
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
set -x
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


