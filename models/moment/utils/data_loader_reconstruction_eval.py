import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetectionDataset(Dataset):
    def __init__(self, root_path, dataset, target_line_number, context_length, forecast_horizon):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = self.context_length * 2

        self.file_path = os.path.join(root_path, dataset)

        self._load_target_line_data(target_line_number)

    def _load_target_line_data(self, target_line_number):
        """Load a specific line from a JSONL file and extract sequence and label."""
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
                    break  # Exit after processing the target line
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {i + 1}: {e}")

        if len(self.df_raw_test) < self.data_stride_len or len(self.df_raw_test_label) < self.forecast_horizon:
            print(f"Warning: Insufficient sequence or label length in {self.file_path}")

        print(f"Loaded sequence length: {len(self.df_raw_test) / self.data_stride_len:.2f}, "
              f"Label length: {len(self.df_raw_test_label) / self.forecast_horizon:.2f}")

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.context_length * 2

        label_start = self.forecast_horizon * index
        label_end = label_start + self.forecast_horizon

        sequence = np.array(self.df_raw_test[seq_start:seq_end])
        label = np.array(self.df_raw_test_label[label_start:label_end])
        input_mask = np.ones(self.data_stride_len)

        return sequence, input_mask, label

    def __len__(self):
        return len(self.df_raw_test) // self.data_stride_len


def get_loader_eval(root_path, dataset, target_line_number, context_length, forecast_horizon, batch_size):
    dataset = AnomalyDetectionDataset(
        root_path=root_path,
        dataset=dataset,
        target_line_number=target_line_number,
        context_length=context_length,
        forecast_horizon=forecast_horizon
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

