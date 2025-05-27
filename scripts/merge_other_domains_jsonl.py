import os
import json
import glob

# Function to load a JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Function to write data to a JSONL file
def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"JSONL file has been created at {file_path}.")

# Function to merge multiple JSONL files
def merge_jsonl_files(input_files, output_file):
    """
    Merges the specified list of JSONL files and saves the result to a new JSONL file.

    Parameters:
        input_files (list): List of input JSONL file paths
        output_file (str): Path of the output merged JSONL file
    """
    combined_data = []
    for file_path in input_files:
        combined_data.extend(load_jsonl(file_path))
    write_jsonl(output_file, combined_data)

# Create JSONL files that contain merged data from all other data families (excluding itself)
def create_excluded_jsonl_files(domain_dict, base_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for domain in domain_dict.keys():
        # Get JSONL files for all other data families (excluding itself)
        input_files = [
            os.path.join(base_path, f"{other_domain}.jsonl")
            for other_domain in domain_dict.keys()
            if other_domain != domain and os.path.exists(os.path.join(base_path, f"{other_domain}.jsonl"))
        ]

        if not input_files:
            print(f"No other files exist for: {domain}")
            continue

        # Save the merged file
        output_file_path = os.path.join(output_path, f"{domain}.jsonl")
        merge_jsonl_files(input_files, output_file_path)

# Main process
domain_dict_path = '../common/domains_dict.json'
base_path = '../data/time-moe/ratfm/before_merge'
output_path = '../data/time-moe/ratfm/train'

# Load domain_dict
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Create JSONL files that merge data from all other data families
create_excluded_jsonl_files(domain_dict, base_path, output_path)

