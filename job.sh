#!/bin/bash
#SBATCH --job-name=home-robot12
#SBATCH --output=slurm_logs/dlm-%j.out
#SBATCH --error=logs.home_robo-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 2
#SBATCH --partition=kira-lab
#SBATCH --gpus 2080_ti:1
#SBATCH --qos="short"


source ~/.bashrc
source /nethome/asingh3064/flash/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate home-robot

cd /nethome/asingh3064/flash/home-robot/
python projects/habitat_ovmm/eval_baselines_agent.py --start_episode 1001 --end_episode 1199


# python projects/habitat_ovmm/scripts/summarize_metrics.py

