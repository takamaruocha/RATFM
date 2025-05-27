import os
import sys
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml


def load_and_normalize_training_data(file_path):
    """
    Load and standardize the training portion of a time series from a .txt file.
    """
    filename_parts = os.path.basename(file_path).split('_')
    metadata = {
        'series_name': '_'.join(filename_parts[:4]),
        'train_end': int(filename_parts[4]),
        'anomaly_start_in_test': int(filename_parts[5]) - int(filename_parts[4]),
        'anomaly_end_in_test': int(filename_parts[6][:-4]) - int(filename_parts[4]),
    }

    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 1:
            values = np.array(
                [float(val) for val in lines[0].strip().split() if val]
            ).reshape((1, -1))
        else:
            values = np.array(
                [float(line.strip()) for line in lines]
            ).reshape((1, -1))

    values = values.reshape(-1, 1)
    train_values = values[:metadata['train_end']]
    scaler = StandardScaler()
    scaler.fit(train_values)
    standardized_train = scaler.transform(train_values).squeeze()

    print(metadata['series_name'], standardized_train.shape)
    return standardized_train.tolist()


def save_sequence_to_jsonl(file_path, output_dir, group_id):
    """
    Convert a .txt file to a JSONL sequence and save it to the appropriate group file.
    """
    sequence = load_and_normalize_training_data(file_path)
    record = {'sequence': sequence}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{group_id}.jsonl")

    with open(output_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(record) + '\n')

    print(f"Saved JSONL file: {output_path}")


with open("../../common/config.yaml", "r") as f:
    config_common = yaml.safe_load(f)

with open("dataset_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ucr_root_path = config_common["ucr_root_path"]
domain_group_dict_path = config["domain_group_dict_path"]
output_domain_group_dir = config["output_domain_group_dir"]

# Load the domain-group mapping
with open(domain_group_dict_path, "r") as f:
    domain_group_map = json.load(f)

# Create a flat mapping: { "Domain_Group": [entityes] }
group_entity_map = {}
for domain, group_info in domain_group_map.items():
    for group, entity_list in group_info.items():
        group_id = f"{domain}_{group}"
        group_entity_map[group_id] = entity_list

# Iterate through UCR dataset and process relevant files
for filename in os.listdir(ucr_root_path):
    if not filename.endswith('.txt'):
        continue

    file_path = os.path.join(ucr_root_path, filename)
    for group_id, entityes in group_entity_map.items():
        if any(filename.startswith(entity) for entity in entityes):
            save_sequence_to_jsonl(file_path, output_domain_group_dir, group_id)

