import argparse
import subprocess

# Define the argument parser
parser = argparse.ArgumentParser(description="Submit a SLURM job with custom paths.")
parser.add_argument("--common-name", required=True, help="Common name to be appended to the directories.")
parser.add_argument("--exp-config", required=True, help="Relative path to the experiment config file within the base directory.")

# Parse the arguments
args = parser.parse_args()

# Paths to the template and the final job script
template_path = "job_template.sh"
job_script_path = "job1.sh"

# Values to replace in the template
exp_config_base_dir = "projects/habitat_uncertainty/config/"
exp_config = exp_config_base_dir + args.exp_config
common_name = args.common_name

# Fixed directories
checkpoint_dir_base = "Logs/checkpoints"
tensorboard_dir_base = "Logs/tensorLogs"
log_dir_base = "Logs/logs/navObj"

# Read the template
with open(template_path, 'r') as file:
    job_script = file.read()

# Placeholder to be replaced at runtime
checkpoint_dir_placeholder = f"{checkpoint_dir_base}/{common_name}_"+"${MAIN_ADDR}_${JOB_ID}"
tensorboard_dir_placeholder = f"{tensorboard_dir_base}/{common_name}_"+"${MAIN_ADDR}_${JOB_ID}"
log_dir_placeholder = f"{log_dir_base}/{common_name}_"+"${MAIN_ADDR}_${JOB_ID}.log"

# Replace the placeholders with actual values
job_script = job_script.replace("__CHECKPOINT_DIR__", checkpoint_dir_placeholder)
job_script = job_script.replace("__TENSORBOARD_DIR__", tensorboard_dir_placeholder)
job_script = job_script.replace("__LOG_DIR__", log_dir_placeholder)
job_script = job_script.replace("__EXP_CONFIG__", exp_config)

# Write the modified script to a new file
with open(job_script_path, 'w') as file:
    file.write(job_script)

# Make sure the script is executable
subprocess.run(["chmod", "+x", job_script_path])

# Run the script using subprocess
result = subprocess.run(["sbatch", job_script_path], capture_output=True, text=True)

# Print the output and error (if any)
print("stdout:", result.stdout)
print("stderr:", result.stderr)

# Check the return code
if result.returncode == 0:
    print("Job submitted successfully.")
else:
    print("Failed to submit job.")