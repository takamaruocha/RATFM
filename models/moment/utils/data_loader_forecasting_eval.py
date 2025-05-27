import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class AnomalyDetectionDataset(Dataset):
    def __init__(self, root_path, file_name, target_line_number, context_length, forecast_horizon):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = context_length * 2 + forecast_horizon * 2

        self.root_path = root_path
        self.file_name = file_name
        self.file_path = os.path.join(root_path, file_name)

        self._load_target_sample(target_line_number)

    def _load_target_sample(self, target_line_number):
        """Load a specific line (sample) from a JSONL dataset."""
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
                    print(f"Error parsing JSON in [{self.file_name}] at line {i + 1}: {e}")

        if len(self.df_raw_test) < self.data_stride_len or len(self.df_raw_test_label) < self.forecast_horizon:
            print(f"Warning: [{self.file_name}] - Data too short for required sequence length and forecast horizon")

        print(f"Loaded: sequence length = {len(self.df_raw_test) / self.data_stride_len:.2f}, "
              f"label length = {len(self.df_raw_test_label) / self.forecast_horizon:.2f}")

    def __getitem__(self, index):
        """Return a sample consisting of input sequence, forecast, mask, and label."""
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.context_length * 2 + self.forecast_horizon
        pred_end = seq_end + self.forecast_horizon

        label_start = self.forecast_horizon * index
        label_end = label_start + self.forecast_horizon

        input_series = np.array(self.df_raw_test[seq_start:seq_end])
        forecast_target = np.array(self.df_raw_test[seq_end:pred_end])
        label = np.array(self.df_raw_test_label[label_start:label_end])

        input_mask = np.ones_like(input_series)

        return input_series, forecast_target, input_mask, label

    def __len__(self):
        return len(self.df_raw_test) // self.data_stride_len


def get_loader_eval(root_path, dataset, target_line_number, context_length, forecast_horizon, batch_size):
    """
    Return a DataLoader for evaluation using a single line (sample) from a JSONL dataset.
    """
    dataset = AnomalyDetectionDataset(root_path, dataset, target_line_number, context_length, forecast_horizon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

