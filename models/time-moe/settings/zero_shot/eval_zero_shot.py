import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
import yaml

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.evaluate import evaluate_model
from utils.data_loader import get_loader
from common.metrics import process_scores
from common.period_data_loader import get_time_series_period
from common.period_detection import detect_period_fourier


# === Load YAML Configs ===
with open("../../configs/config_zero_shot.yaml", "r") as f:
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

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map=device,
    trust_remote_code=True,
)
model.to(device)

# Load domain dictionary
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Get sorted list of datasets
datasets = sorted(os.listdir(ucr_root_path))

# Iterate over datasets
for dataset_filename in datasets:
    if dataset_filename in excluded_files:
        continue

    dataset_path = os.path.join(ucr_root_path, dataset_filename)
    if not os.path.exists(dataset_path):
        continue

    print(f"Processing dataset: {dataset_filename}")

    # Detect period using training dataset
    train_series = get_time_series_period(
        root_path=ucr_root_path,
        dataset=dataset_filename,
        scale=True
    )
    detected_period = detect_period_fourier(train_series)

    # Load test set
    test_loader = get_loader(
        root_path=ucr_root_path,
        dataset=dataset_filename,
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
        window=int(detected_period), center=True
    ).mean().fillna(0).to_numpy()
    smoothed_scores = MinMaxScaler().fit_transform(smoothed_scores.reshape(-1, 1)).ravel()

    # Dataset identifier for saving
    entity = '_'.join(dataset_filename.split('_')[:4])
    domain = None
    for domain, dataset_list in domain_dict.items():
        if any(entity.startswith(prefix) for prefix in dataset_list):
            domain = domain
            break

    if domain is None:
        print(f"Warning: Domain not found for dataset {entity}. Skipping.")
        continue

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

    print(f"Completed: {dataset_filename}\n")
