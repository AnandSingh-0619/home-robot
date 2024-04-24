#!/bin/bash
#SBATCH --job-name=home-robot-yolo
#SBATCH --output=slurm_logs/home_robo-ver-%j.out
#SBATCH --error=slurm_logs/home_robo-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --partition=overcap,kira-lab
#SBATCH --gpus a40:4
#SBATCH --qos="long"
#SBATCH --exclude=spd-13
#SBATCH --requeue
#SBATCH --signal=USR1@100

export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH
<<<<<<< Updated upstream
CHECKPOINT_DIR="data/new_checkpoints/ovmm/gaze_og_heatmap4"
TENSORBOARD_DIR="tb/yolo_gaze/nGPU4_nENV32_og_heatmap4"
=======
export CUDA_LAUNCH_BLOCKING=1
CHECKPOINT_DIR="data/new_checkpoints/ddppo/ovmm/gaze_yolo_heatmap1"
TENSORBOARD_DIR="tb/yolo_gaze/nGPU4_nENV32_yolo_heatmap1"
>>>>>>> Stashed changes
source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh

conda deactivate
conda activate home-robot
# export HABITAT_ENV_DEBUG=1
cd ~/flash/home-robot 

# srun python -um habitat_uncertainity.run \
#     --exp-config=projects/habitat_uncertainity/config/yolo_rl_skill.yaml \
#     --run-type=train \
#     habitat_baselines.num_environments=32 \
#     habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
#     habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
#     habitat_baselines.load_resume_state_config=True \


srun python -um habitat_baselines.run \
   --config-name=ovmm/rl_skill.yaml \
   habitat_baselines.evaluate=False \
   habitat_baselines.num_environments=32 \
   habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
   habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
   habitat_baselines.load_resume_state_config=False \