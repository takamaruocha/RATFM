import os
import json
import numpy as np
from numpy.linalg import norm
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetectionDataset(Dataset):
    def __init__(self, root_path, dataset_filename, context_length, forecast_horizon):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = context_length * 2 + forecast_horizon * 2

        self.dataset_path = os.path.join(root_path, dataset_filename)
        self.sequence_data = []

        self._load_data()

    def _load_data(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"Loading {os.path.basename(self.dataset_path)}")):
                try:
                    record = json.loads(line.strip())
                    self.sequence_data.extend(record["sequence"])
                except json.JSONDecodeError as e:
                    print(f"[Error] Failed to parse line {i+1} in {self.dataset_path}: {e}")

        total_len = len(self.sequence_data)
        print(f"Loaded sequence length: {total_len} ({total_len / self.data_stride_len:.2f} windows)")

    def __getitem__(self, index):
        start_idx = self.data_stride_len * index
        middle_idx = start_idx + self.context_length * 2 + self.forecast_horizon
        end_idx = middle_idx + self.forecast_horizon

        input_series = np.array(self.sequence_data[start_idx:middle_idx])
        forecast_target = np.array(self.sequence_data[middle_idx:end_idx])
        input_mask = np.ones_like(input_series)

        return input_series, forecast_target, input_mask

    def __len__(self):
        return len(self.sequence_data) // self.data_stride_len


def get_loader_train(root_path, dataset, context_length, forecast_horizon, batch_size):
    dataset = AnomalyDetectionDataset(root_path, dataset, context_length, forecast_horizon)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

