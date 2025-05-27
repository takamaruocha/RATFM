import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file
from tqdm.auto import tqdm
import yaml

# Append paths for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.evaluate import evaluate_model
from utils.data_loader import get_loader
from common.metrics import process_scores
from common.period_data_loader import get_time_series_period
from common.period_detection import detect_period_fourier


# === Load YAML Configs ===
with open("../../configs/config_in_domain_FT.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("../../../../common/config.yaml", "r") as f:
    config_common = yaml.safe_load(f)

# === Parse Config Values ===
context_length = config["context_length"]
forecast_horizon = config["forecast_horizon"]
data_stride_len = config["data_stride_len"]
batch_size = config["batch_size"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Paths
base_dir = os.path.abspath(config['base_dir'])
domain_dict_path = os.path.join(base_dir, config['domain_dict_path'])
save_base = os.path.join(base_dir, config['save_base'])
ucr_root_path = config_common["ucr_root_path"]

# Exclusion list
excluded_files = config_common.get("excluded_files", [])

# Load domain dictionary
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Get sorted list of all dataset files
dataset_files = sorted(os.listdir(ucr_root_path))

target_groups = ['group1', 'group2']

# Loop through each domain and evaluate the model on the opposite group
for domain, group_dict in domain_dict.items():
    for group in target_groups:
        print(f"\n[Domain: {domain} | Group: {group}] Loading model...")

        # Load trained model for the current domain/group
        output_dir = os.path.join(save_base, domain, group)
        model_path = os.path.join(output_dir, "model.safetensors")

        model = AutoModelForCausalLM.from_pretrained(
            'Maple728/TimeMoE-50M',
            device_map=device,
            trust_remote_code=True,
        )
        model.load_state_dict(load_file(model_path))
        model.to(device)
        model.eval()

        # Evaluate on the opposite group
        other_group = 'group2' if group == 'group1' else 'group1'
        eval_entities = set(group_dict.get(other_group, []))

        for entity in eval_entities:
            matching_files = [f for f in dataset_files if entity in f and f.endswith('.txt')]
            if not matching_files:
                print(f"Skipping {entity}: no matching dataset found.")
                continue

            matched_filename = matching_files[0]
            if matched_filename in excluded_files:
                continue

            dataset_path = os.path.join(ucr_root_path, matched_filename)
            if not os.path.exists(dataset_path):
                continue

            print(f"Processing dataset: {matched_filename}")

            # Detect period using training dataset
            train_series = get_time_series_period(
                root_path=ucr_root_path,
                dataset=matched_filename,
                scale=True
            )
            period = detect_period_fourier(train_series)

            # Build test loader
            test_loader = get_loader(
                root_path=ucr_root_path,
                dataset=matched_filename,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
                stride=data_stride_len,
                scale=True,
                mode="test",
                batch_size=batch_size,
            )

            # Run inference
            anomaly_scores_l1, test_labels = evaluate_model(
                model, test_loader, forecast_horizon, device
            )

            # Normalize and smooth scores
            score_l1 = MinMaxScaler().fit_transform(anomaly_scores_l1.reshape(-1, 1)).ravel()
            smoothed_scores = pd.Series(anomaly_scores_l1).rolling(
                window=int(period), center=True
            ).mean().fillna(0).to_numpy()
            smoothed_scores = MinMaxScaler().fit_transform(smoothed_scores.reshape(-1, 1)).ravel()

            # Save raw scores
            print(f"[{entity}] ▶ Saving anomaly scores: Raw L1 Loss")
            save_dir_l1 = os.path.join(save_base, "L1Loss", domain, entity)
            os.makedirs(save_dir_l1, exist_ok=True)
            process_scores(score_l1, test_labels.astype(int), data_stride_len, save_dir_l1)

            # Save smoothed scores
            print(f"[{entity}] ▶ Saving anomaly scores: Smoothed (SMA + L1 Loss)")
            save_dir_sma = os.path.join(save_base, "L1Loss_SMA", domain, entity)
            os.makedirs(save_dir_sma, exist_ok=True)
            process_scores(smoothed_scores, test_labels.astype(int), data_stride_len, save_dir_sma)

            print(f"Completed: {matched_filename}\n")

