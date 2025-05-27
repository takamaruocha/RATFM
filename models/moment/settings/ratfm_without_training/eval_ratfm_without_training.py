import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from transformers import logging as hf_logging
import yaml

# Suppress warnings from transformers
hf_logging.set_verbosity_error()

# Set project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils/')))

from data_loader_reconstruction_eval import get_loader_eval
from momentfm.models.moment import MOMENTPipeline
from evaluate_reconstruction import evaluate_model
from common.metrics import process_scores
from common.period_detection import detect_period_fourier
from common.period_data_loader import get_time_series_period


# === Load YAML Configs ===
config_path = os.path.join(os.path.dirname(__file__), '../../configs/config_ratfm_without_training.yaml')
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
test_data_path = os.path.join(base_dir, config['test_data_path'])
save_base = os.path.join(base_dir, config['save_base'])
ucr_root_path = config_common["ucr_root_path"]

# Exclusion list
excluded_files = config_common.get("excluded_files", [])

# Load domain dictionary
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Load model
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "reconstruction",
    },
)
model.to(device)

# Evaluate all data families
for domain, entities in domain_dict.items():
    print(f"Processing Domain: {domain}")

    for target_line_number, entity in enumerate(entities):
        # Search matching dataset file
        dataset_file = next((f for f in os.listdir(ucr_root_path)
                             if entity in f and f.endswith('.txt')), None)
        if not dataset_file or dataset_file in excluded_files:
            continue

        print(f"Using dataset file: {dataset_file}")

        # Detect period using training dataset
        train_data = get_time_series_period(root_path=ucr_root_path, dataset=dataset_file, scale=True)
        period = int(detect_period_fourier(train_data))
        print(f"Detected Period (Fourier): {period}")

        # Load test data from JSONL
        domain_filename_test = f"{domain}.jsonl"
        test_loader = get_loader_eval(
            root_path=test_data_path,
            dataset=domain_filename_test,
            target_line_number=target_line_number,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size
        )

        # Run inference
        anomaly_scores, test_labels = evaluate_model(
            model, test_loader, forecast_horizon, device
        )

        # Normalize and smooth scores
        score_l1 = MinMaxScaler().fit_transform(anomaly_scores.reshape(-1, 1)).ravel()
        smoothed_scores = pd.Series(score_l1).rolling(
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

        print(f"Completed: {dataset_file}\n")

