#!/bin/bash
#SBATCH --job-name=home-robot-yolo
#SBATCH --output=Logs/slurmLogs/home_robo-ver-%j.out
#SBATCH --error=Logs/slurmLogs/home_robo-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 15
#SBATCH --ntasks-per-node 4
#SBATCH --partition=kira-lab
#SBATCH --gpus a40:4
#SBATCH --qos="long"
#SBATCH --exclude=spd-13,xaea-12,ig-88,omgwth,baymax,heistotron
#SBATCH --requeue
#SBATCH --signal=USR1@100

export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH

MAIN_ADDR=$(scontrol show hostnames "
${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

JOB_ID=${SLURM_JOB_ID}
CHECKPOINT_DIR="data/new_checkpoints/navObj_942052_yolo_ft2"
TENSORBOARD_DIR="Logs/tensorLogs/navObj_942052_yolo_ft2_${JOB_ID}"
LOG_DIR="Logs/logs/navObj/navObj_942052_yolo_ft2_${JOB_ID}.log"
set -x
source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh

conda deactivate
conda activate home-robot
cd ~/flash/home-robot 

srun python -um habitat_uncertainty.run \
    --exp-config=projects/habitat_uncertainty/config/yolo_nav_obj.yaml \
    --run-type=train \
    habitat_baselines.num_environments=20 \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
    habitat_baselines.log_file=${LOG_DIR} \
    habitat_baselines.load_resume_state_config=True


    # habitat_baselines.wb.run_name="940665_YOLO_50_inference"

