#!/bin/bash
#SBATCH --job-name=home-robot-yolo
#SBATCH --output=Logs/slurmLogs/home_robo-ver-%j.out
#SBATCH --error=Logs/slurmLogs/home_robo-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --partition=overcap,kira-lab
#SBATCH --gpus a40:4
#SBATCH --qos="long"
#SBATCH --exclude=spd-13
#SBATCH --requeue
#SBATCH --signal=USR1@100

export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH

MAIN_ADDR=$(scontrol show hostnames "
${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

CHECKPOINT_DIR="Logs/checkpoints/gaze/gaze_heatmap"
TENSORBOARD_DIR="Logs/tensorLogs/nGPU4_nENV32/gaze/gaze_heatmap"
LOG_DIR="Logs/logs/gaze/gaze_heatmap.log"

source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh

conda deactivate
conda activate home-robot
cd ~/flash/home-robot 

srun python -um habitat_baselines.run \
   --config-name=ovmm/rl_skill.yaml \
   habitat_baselines.evaluate=False \
   habitat_baselines.num_environments=32 \
   habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
   habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
   habitat_baselines.log_file=${LOG_DIR} \
   habitat_baselines.load_resume_state_config=True \