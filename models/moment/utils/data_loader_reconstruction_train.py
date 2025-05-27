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
    def __init__(self, root_path, dataset, context_length, forecast_horizon):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = self.context_length * 2  # Fixed stride for training mode

        self.file_path = os.path.join(root_path, dataset)
        self._load_sequence()

    def _load_sequence(self):
        """Load and concatenate all sequence data from a JSONL file."""
        self.df_raw_train = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"Loading {os.path.basename(self.file_path)}")):
                try:
                    data = json.loads(line.strip())
                    self.df_raw_train.extend(data["sequence"])
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {i + 1}: {e}")

        total_context_length = len(self.df_raw_train)
        print(f"Total loaded sequence length: {total_context_length} "
              f"({total_context_length / self.data_stride_len:.2f} samples)")

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.context_length * 2

        target_sequence = np.array(self.df_raw_train[seq_start:seq_end])
        input_mask = np.ones(self.context_length * 2)

        return target_sequence, input_mask

    def __len__(self):
        return len(self.df_raw_train) // self.data_stride_len


def get_loader_train(root_path, dataset, context_length, forecast_horizon, batch_size):
    dataset = AnomalyDetectionDataset(
        root_path=root_path,
        dataset=dataset,
        context_length=context_length,
        forecast_horizon=forecast_horizon
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

