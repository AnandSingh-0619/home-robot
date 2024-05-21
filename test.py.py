import json
import gzip
import os

def count_episodes_in_json_gz(file_path):
    with gzip.open(file_path, 'rt') as f:
        data = json.load(f)
        # Assuming episodes are stored in a list under the key 'episodes'
        return len(data['episodes'])

# Directory where the .json.gz files are stored
json_gz_directory = 'data/datasets/ovmm/val/'

total_episodes = 0
for file_name in os.listdir(json_gz_directory):
    if file_name.endswith('.json.gz'):
        file_path = os.path.join(json_gz_directory, file_name)
        total_episodes += count_episodes_in_json_gz(file_path)

print(f"Total number of episodes in validation dataset: {total_episodes}")
