import os
import sys
import json
import yaml
import subprocess


# === Load YAML Configs ===
config_path = os.path.join(os.path.dirname(__file__), '../../configs/config_in_domain_FT.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# === Parse Config Values ===
base_dir = os.path.abspath(config['base_dir'])
main_script = os.path.join(base_dir, config['main_script'])
domain_dict_path = os.path.join(base_dir, config['domain_dict_path'])
base_data_path = os.path.join(base_dir, config['base_data_path'])
save_base = os.path.join(base_dir, config['save_base'])
num_train_epochs = config['num_train_epochs']
normalization_method = config['normalization_method']

# Load domain-group dictionary
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Loop over each data domain and group
target_groups = ['group1', 'group2']
for domain, groups in domain_dict.items():
    for group in target_groups:
        # Construct path to domain+group data file
        domain_data_filename = f"{domain}_{group}.jsonl"
        domain_data_path = os.path.join(base_data_path, domain_data_filename)

        # Define output directory
        output_dir = os.path.join(save_base, domain, group)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[{domain}/{group}] Created output directory: {output_dir}")

        # Build training command
        command = [
            'python', main_script,
            '--data_path', domain_data_path,
            '--output_path', output_dir,
            '--num_train_epochs', str(num_train_epochs),
            '--model_data_family', domain,
            '--group', group,
            '--normalization_method', normalization_method,
        ]

        print(f"[{domain}/{group}] Executing: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{domain}/{group}] Error during execution: {e}")
