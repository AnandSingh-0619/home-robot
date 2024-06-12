#!/bin/bash
#SBATCH --job-name=home-robot-yolo
#SBATCH --output=Logs/slurmLogs/yolosam_fullres_100v_try3-ver-%j.out
#SBATCH --error=Logs/slurmLogs/yolosam_fullres_100v_try3-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --partition=overcap
#SBATCH --gpus a40:4
#SBATCH --qos="long"
#SBATCH --exclude=spd-13,xaea-12,ig-88,omgwth
#SBATCH --requeue
#SBATCH --signal=USR1@100

export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH

MAIN_ADDR=$(scontrol show hostnames "
${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

CHECKPOINT_DIR="Logs/checkpoints/yolosam_fullres_100v_try3"
TENSORBOARD_DIR="Logs/tensorLogs/gaze/yolosam_fullres_100v_try3"
LOG_DIR="Logs/logs/gaze/yolosam_fullres_100v_try3.log"

source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh

conda deactivate
conda activate home-robot
cd ~/flash/home-robot 

srun python -um habitat_uncertainty.run \
    --exp-config=projects/habitat_uncertainty/config/YOLOSAM_gaze_rl_skill.yaml \
    --run-type=train \
    habitat_baselines.num_environments=32 \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
    habitat_baselines.log_file=${LOG_DIR} \
    habitat_baselines.load_resume_state_config=True \


