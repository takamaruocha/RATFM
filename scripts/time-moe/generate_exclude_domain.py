import os
import sys
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml


def normalize_train_data(file_path):
    fields = os.path.basename(file_path).split('_')
    train_end_index = int(fields[4])
    name_id = '_'.join(fields[:4])

    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 1:
            values = [float(v) for v in lines[0].strip().split() if v]
        else:
            values = [float(line.strip()) for line in lines]

    data = np.array(values).reshape(-1, 1)
    train_data = data[:train_end_index]

    scaler = StandardScaler()
    normalized = scaler.fit_transform(train_data).squeeze()

    print(f"[{name_id}] Normalized shape: {normalized.shape}")
    return normalized.tolist()


def write_excluded_domain_jsonl(excluded_domain, domain_dict, data_folder, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for domain, entities in domain_dict.items():
            if domain == excluded_domain:
                continue  # skip excluded domain

            for entity in entities:
                for filename in os.listdir(data_folder):
                    if entity in filename and filename.endswith('.txt'):
                        file_path = os.path.join(data_folder, filename)
                        sequence = normalize_train_data(file_path)
                        jsonl_file.write(json.dumps({'sequence': sequence}) + '\n')

    print(f"Saved excluded-domain JSONL: {output_path}")


if __name__ == "__main__":
    with open("../../common/config.yaml", "r") as f:
        config_common = yaml.safe_load(f)

    with open("dataset_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
 
    ucr_root_path = config_common["ucr_root_path"]
    domain_dict_path = config["domain_dict_path"]
    output_exclude_domain_dir = config["output_exclude_domain_dir"]

    with open(domain_dict_path, 'r', encoding='utf-8') as f:
        domain_dict = json.load(f)

    for excluded_domain in domain_dict.keys():
        save_path = os.path.join(output_exclude_domain_dir, f"{excluded_domain}.jsonl")
        write_excluded_domain_jsonl(excluded_domain, domain_dict, ucr_root_path, save_path)



