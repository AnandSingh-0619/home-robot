#!/bin/bash
#SBATCH --job-name=home-robot-yolo
#SBATCH --output=slurm_logs/home_robo-ver-%j.out
#SBATCH --error=slurm_logs/home_robo-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 3
#SBATCH --partition=overcap
#SBATCH --gpus a40:4
#SBATCH --qos="long"

CHECKPOINT_DIR="data/new_checkpoints/ovmm/gaze"
source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate home-robot
export HABITAT_ENV_DEBUG=1
cd ~/flash/home-robot 
srun python -um habitat_uncertainity.run \
    --exp-config=projects/habitat_uncertainity/config/yolo_rl_skill.yaml \
    --run-type=train \
    habitat_baselines.num_environments=20 \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR}


