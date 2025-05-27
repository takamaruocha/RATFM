import os
import sys
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import OneCycleLR
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils/')))

from momentfm.models.moment import MOMENTPipeline
from data_loader_reconstruction_train import get_loader_train
from masking import Masking


# === Load YAML Configs ===
with open("../../configs/config_ratfm_reconstruction.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Parse Config Values ===
context_length = config["context_length"]
forecast_horizon = config["forecast_horizon"]
batch_size = config["batch_size"]
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Paths
base_dir = os.path.abspath(config['base_dir'])
domain_dict_path = os.path.join(base_dir, config['domain_dict_path'])
base_data_path = os.path.join(base_dir, config['base_data_path'])
save_base = os.path.join(base_dir, config['save_base'])
num_train_epochs = config['num_train_epochs']

# Load domain dictionary
with open(domain_dict_path, 'r', encoding='utf-8') as f:
    domain_dict = json.load(f)

# Iterate over each data family
for domain, _ in domain_dict.items():
    print(f"\nProcessing Data Family: {domain}")

    # Load test data from JSONL
    dataset_filename = f"{domain}.jsonl"
    train_loader = get_loader_train(
        root_path=base_data_path,
        dataset=dataset_filename,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
    )

    # Initialize MOMENT model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'reconstruction',
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': False,
            'freeze_embedder': False,
            'freeze_head': False,
        },
    )
    model.to(device)
    model.train()

    # Set up training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = OneCycleLR(
        optimizer, max_lr=1e-4, total_steps=len(train_loader) * num_train_epochs, pct_start=0.3
    )
    max_grad_norm = 5.0
    mask_generator = Masking(mask_ratio=0.3)

    # Fine-tuning loop
    for epoch in range(num_train_epochs):
        for timeseries, input_mask in train_loader:
            timeseries = timeseries.float().to(device).unsqueeze(1)
            input_mask = input_mask.to(device).long()
            input_mask = input_mask.repeat_interleave(timeseries.shape[1], dim=0)

            # Generate random mask
            mask = mask_generator.generate_mask(x=timeseries, input_mask=input_mask).to(device).long()

            # Forward pass
            with torch.cuda.amp.autocast():
                output = model(x_enc=timeseries, input_mask=input_mask, mask=mask)
                loss = criterion(
                    output.reconstruction[:, :, -forecast_horizon:], 
                    timeseries[:, :, -forecast_horizon:]
                )

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_train_epochs} - Loss: {loss.item():.6f}")

    # Save fine-tuned model
    save_path = os.path.join(save_base, domain)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "finetuned_model.pth"))
    print(f"Model saved to: {save_path}")

