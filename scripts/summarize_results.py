import os
import json
from collections import defaultdict
import yaml


with open("metrics_summary_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

domain_dict_path = config["domain_dict_path"]
save_base = config["save_base"]

# Load domain dictionary
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# List of evaluation metrics to summarize
metrics = [
    'VUS_PR', 'VUS_ROC',
    'F', 'Precision', 'Recall'
]

# Summary containers
results_summary = {}
all_metrics_values = defaultdict(list)

# Process each domain (data_family)
for domain, _ in domain_dict.items():
    domain_result_dir = os.path.join(save_base, domain)
    if not os.path.isdir(domain_result_dir):
        continue

    print(f"Processing domain: {domain}")
    domain_metrics = defaultdict(list)

    # Traverse subdirectories (e.g., per dataset or group)
    for subfolder in os.listdir(domain_result_dir):
        dataset_path = os.path.join(domain_result_dir, subfolder)
        if not os.path.isdir(dataset_path):
            continue

        json_file = os.path.join(dataset_path, 'output_results.json')
        if not os.path.exists(json_file):
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

            # Extract desired metrics
            for metric in metrics:
                if metric in data:
                    domain_metrics[metric].append(data[metric])
                    all_metrics_values[metric].append(data[metric])

    # Compute average for each metric in this domain
    averaged = {
        metric: (sum(vals) / len(vals)) if vals else None
        for metric, vals in domain_metrics.items()
    }
    results_summary[domain] = averaged

# Compute overall average across all domains
overall_averages = {
    metric: (sum(vals) / len(vals)) if vals else None
    for metric, vals in all_metrics_values.items()
}

# Print per-domain averages
for domain, metrics_dict in results_summary.items():
    print(f"\n[Domain: {domain}]")
    for metric, avg in metrics_dict.items():
        print(f"  {metric}: {avg:.5f}" if avg is not None else f"  {metric}: No data")

# Print overall average
print("\n[Overall Averages]")
for metric, avg in overall_averages.items():
    print(f"  {metric}: {avg:.5f}" if avg is not None else f"  {metric}: No data")
