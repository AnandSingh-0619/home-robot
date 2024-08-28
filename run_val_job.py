import os

# Define the parameters for start_episode and end_episode
start_end_pairs = [
    (1, 100), (101, 200), (201, 300), (301, 400), (401, 500), (501, 600),
    (601, 700), (701, 800), (801, 900), (901, 1000), (1001, 1100),(1101, 1199)
]

# Read the template SLURM job script
with open("val_template.sh", "r") as template_file:
    template_script = template_file.read()

# Loop through the start_end_pairs and create a batch job for each pair
for start_episode, end_episode in start_end_pairs:
    # Create the SLURM script for the current pair
    job_script = template_script.replace("${START_EPISODE}", str(start_episode)).replace("${END_EPISODE}", str(end_episode))
    
    # Write the job script to a temporary file
    script_filename = f"temp_slurm_script_{start_episode}_{end_episode}.sh"
    with open(script_filename, "w") as script_file:
        script_file.write(job_script)
    
    # Submit the SLURM script
    os.system(f"sbatch {script_filename}")
    
    # Optionally, delete the temporary SLURM script file after submission
    os.remove(script_filename)
