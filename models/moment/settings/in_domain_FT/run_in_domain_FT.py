import os
import sys
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils/')))

from momentfm.models.moment import MOMENTPipeline
from data_loader_reconstruction import AnomalyDetectionDataset
from masking import Masking


# === Load YAML Configs ===
with open("../../configs/config_in_domain_FT.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("../../../../common/config.yaml", "r") as f:
    config_common = yaml.safe_load(f)

# === Parse Config Values ===
context_length = config["context_length"]
forecast_horizon = config["forecast_horizon"]
batch_size = config["batch_size"]
num_train_epochs = config['num_train_epochs']
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Paths
base_dir = os.path.abspath(config['base_dir'])
domain_dict_path = os.path.join(base_dir, config['domain_dict_path'])
save_base = os.path.join(base_dir, config['save_base'])
ucr_root_path = config_common["ucr_root_path"]

# Load domain-group dictionary
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Get sorted list of all dataset files
dataset_files = sorted(os.listdir(ucr_root_path))

# Loop over each data domain and group
for domain, groups in domain_dict.items():
    print(f"Processing Data Family: {domain}")

    for group in ['group1', 'group2']:
        print(f"Training model for {domain} - {group}")

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "reconstruction",
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': False,
                'freeze_embedder': False,
                'freeze_head': False,
            },
        )
        model.to(device)
        model.train()

        # Get training data: use files from the other group
        other_group = 'group2' if group == 'group1' else 'group1'
        other_entities = set(groups.get(other_group, []))

        train_datasets = []
        for file_name in dataset_files:
            if not file_name.endswith('.txt'):
                continue
            entity = '_'.join(file_name.split('_')[:4])
            if entity in groups[group]:
                continue
            if entity not in other_entities:
                continue

            train_dataset = AnomalyDetectionDataset(
                root_path=ucr_root_path,
                dataset=file_name,
                context_length=context_length,
                data_stride_len=context_length,
                scale=True,
            )
            train_datasets.append(train_dataset)

        if not train_datasets:
            print(f"No training data found for {domain} - {group}")
            continue

        combined_train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(
            dataset=combined_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-4,
            total_steps=len(train_loader) * num_train_epochs,
            pct_start=0.3
        )
        max_norm = 5.0
        mask_generator = Masking(mask_ratio=0.3)

        # Training loop
        for epoch in range(num_train_epochs):
            for timeseries, input_mask in train_loader:
                timeseries, input_mask = timeseries.float().to(device), input_mask.to(device)
                timeseries = timeseries.unsqueeze(1)
                n_channels = timeseries.shape[1]
                input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                mask = mask_generator.generate_mask(x=timeseries, input_mask=input_mask).to(device).long()

                with torch.cuda.amp.autocast():
                    output = model(x_enc=timeseries, input_mask=input_mask, mask=mask)
                    loss = criterion(output.reconstruction[:, :, -forecast_horizon:], timeseries[:, :, -forecast_horizon:])

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            print(f"Epoch {epoch+1}/{num_train_epochs} - Loss: {loss.item():.4f}")

        # Save model
        save_path = os.path.join(save_base, domain, group)
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, "finetuned_model.pth"))
        print(f"Model saved for {domain} - {group}\n")

