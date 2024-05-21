#!/bin/bash
#SBATCH --job-name=yolo-eval
#SBATCH --output=Logs/slurmLogs/eval-ver-%j.out
#SBATCH --error=Logs/slurmLogs/eval-ver-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --partition=overcap
#SBATCH --gpus a40:1
#SBATCH --qos="long"
#SBATCH --exclude=spd-13

#SBATCH --signal=USR1@100

export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export PYTHONPATH=~/flash/home-robot/projects/:$PYTHONPATH

MAIN_ADDR=$(scontrol show hostnames "
${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
export CUDA_LAUNCH_BLOCKING=1

source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh

conda deactivate
conda activate home-robot
cd ~/flash/home-robot/

srun python projects/habitat_ovmm/eval_baselines_agent.py --start_episode=1001 --end_episode=1199


