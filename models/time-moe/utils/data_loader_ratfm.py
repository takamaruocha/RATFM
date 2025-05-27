import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import norm
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class AnomalyDetectionDataset(Dataset):
    def __init__(self, root_path, dataset, target_line_number, context_length, forecast_horizon):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = self.context_length * 2 + self.forecast_horizon * 2

        self.root_path = root_path
        self.file_name = dataset
        self.file_path = os.path.join(root_path, dataset)

        self._load_target_line(target_line_number)

    def _load_target_line(self, target_line_number):
        self.df_raw_test = []
        self.df_raw_test_label = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i != target_line_number:
                    continue
                try:
                    data = json.loads(line.strip())
                    self.df_raw_test = data["sequence"]
                    self.df_raw_test_label = data["label"]
                    break
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in [{self.file_name}] on line {i+1}: {e}")

        if len(self.df_raw_test) < self.data_stride_len or len(self.df_raw_test_label) < self.forecast_horizon:
            print(f"Warning: [{self.file_name}] - Insufficient data for sequence or label length")

        print(f"Loaded sequence windows: {len(self.df_raw_test) / self.data_stride_len:.2f}, label chunks: {len(self.df_raw_test_label) / self.forecast_horizon:.2f}")

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.context_length * 2 + self.forecast_horizon
        pred_end = seq_end + self.forecast_horizon

        label_start = self.forecast_horizon * index
        label_end = label_start + self.forecast_horizon

        timeseries = np.array(self.df_raw_test[seq_start:seq_end])
        forecast = np.array(self.df_raw_test[seq_end:pred_end])
        label = np.array(self.df_raw_test_label[label_start:label_end])

        return timeseries, forecast, label

    def __len__(self):
        return len(self.df_raw_test) // self.data_stride_len


def get_loader(root_path, dataset, target_line_number, context_length, forecast_horizon, batch_size):
    dataset_instance = AnomalyDetectionDataset(root_path, dataset, target_line_number, context_length, forecast_horizon)
    data_loader = DataLoader(
        dataset=dataset_instance,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return data_loader
