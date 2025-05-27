import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yaml

# Append paths for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils/')))

from evaluate_reconstruction import evaluate_model
from momentfm.models.moment import MOMENTPipeline
from data_loader_reconstruction import get_loader
from common.metrics import process_scores
from common.period_data_loader import get_time_series_period
from common.period_detection import detect_period_fourier


# === Load YAML Configs ===
config_path = os.path.join(os.path.dirname(__file__), '../../configs/config_out_domain_FT.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

with open("../../../../common/config.yaml", "r") as f:
    config_common = yaml.safe_load(f)

# === Parse Config Values ===
context_length = config["context_length"]
forecast_horizon = config["forecast_horizon"]
data_stride_len = config["data_stride_len"]
batch_size = config["batch_size"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# === Parse Config Values ===
base_dir = os.path.abspath(config['base_dir'])
domain_dict_path = os.path.join(base_dir, config['domain_dict_path'])
save_base = os.path.join(base_dir, config['save_base'])
ucr_root_path = config_common["ucr_root_path"]

# Exclusion list
excluded_files = config_common.get("excluded_files", [])

# Load domain-to-dataset mapping
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Get list of all available dataset filenames
dataset_files = sorted(os.listdir(ucr_root_path))

# Evaluate each domain-specific model
for domain, entities in domain_dict.items():
    print(f"\n[Domain: {domain}] Loading model...")

    model_path = os.path.join(save_base, domain, "finetuned_model.pth")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "reconstruction"
            },
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Evaluate model on all entities in the domain
    for entity in entities:
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

        # Estimate periodicity from training data
        train_series = get_time_series_period(
            root_path=ucr_root_path,
            dataset=matched_filename,
            scale=True
        )
        period = detect_period_fourier(train_series)

        # Load test set
        test_loader = get_loader(
            root_path=ucr_root_path,
            dataset=matched_filename,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            data_stride_len=data_stride_len,
            scale=True,
            batch_size=batch_size,
            mode='test'
        )

        # Run inference
        anomaly_scores_l1, test_labels = evaluate_model(
            model, test_loader, forecast_horizon, device
        )

        # Normalize raw scores
        score_l1 = MinMaxScaler().fit_transform(anomaly_scores_l1.reshape(-1, 1)).ravel()

        # Apply simple moving average smoothing
        smoothed_scores = pd.Series(anomaly_scores_l1).rolling(
            window=int(period), center=True
        ).mean().fillna(0).to_numpy()
        smoothed_scores = MinMaxScaler().fit_transform(smoothed_scores.reshape(-1, 1)).ravel()

        # Save raw L1 scores
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

