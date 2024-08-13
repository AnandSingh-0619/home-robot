#!/bin/bash
#SBATCH --job-name=home-robot-yolo
#SBATCH --output=Logs/slurmLogs/home_robo-ver-%j.out
#SBATCH --error=Logs/slurmLogs/home_robo-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 15
#SBATCH --ntasks-per-node 4
#SBATCH --partition=overcap
#SBATCH --gpus a40:4
#SBATCH --qos="long"
#SBATCH --exclude=spd-13,xaea-12,ig-88,omgwth,baymax
#SBATCH --requeue
#SBATCH --signal=USR1@100

export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH

MAIN_ADDR=$(scontrol show hostnames "
${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

JOB_ID=${SLURM_JOB_ID}
CHECKPOINT_DIR="data/new_checkpoints/navObj_CLIP_ResNet_w_recep_embedv4_939869"
TENSORBOARD_DIR="Logs/tensorLogs/navObj_CLIP_ResNet_w_recep_embedv4_${JOB_ID}"
LOG_DIR="Logs/logs/navObj/navObj_CLIP_ResNet_w_recep_embedv4_${JOB_ID}.log"
set -x
source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh

conda deactivate
conda activate home-robot
cd ~/flash/home-robot 

srun python -um habitat_uncertainty.run \
    --exp-config=projects/habitat_uncertainty/config/rl_skill.yaml \
    --run-type=train \
    habitat_baselines.num_environments=32 \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
    habitat_baselines.log_file=${LOG_DIR} \
    habitat_baselines.load_resume_state_config=True

