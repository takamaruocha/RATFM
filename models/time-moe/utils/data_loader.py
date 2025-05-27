import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetectionDataset(Dataset):
    def __init__(self, root_path, dataset, context_length, forecast_horizon, stride, scale, mode):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.root_path = root_path
        self.dataset = dataset
        self.scale = scale
        self.mode = mode

        self._load_data()

    def _load_data(self):
        # Extract metadata from file name
        fields = self.dataset.split('_')
        train_end = int(fields[4])
        anomaly_start = int(fields[5]) - train_end
        anomaly_end = int(fields[6][:-4]) - train_end

        # Load time series data
        file_path = os.path.join(self.root_path, self.dataset)
        with open(file_path) as f:
            lines = f.readlines()
            if len(lines) == 1:
                values = [eval(val) for val in lines[0].strip().split(" ") if len(val) > 1]
            else:
                values = [eval(line.strip()) for line in lines]

        data = np.array(values).reshape(-1, 1)

        # Scale using training portion only
        train_data = data[:train_end]
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        data = self.scaler.transform(data).squeeze()

        if self.mode == "train":
            self.data = data[:train_end]
            self.length = len(self.data)
        else:
            self.data = data[train_end:]
            self.length = len(self.data)
            self.labels = np.zeros(self.length)
            self.labels[anomaly_start:anomaly_end + 1] = 1

    def __getitem__(self, index):
        start_idx = self.stride * index
        end_idx = start_idx + self.context_length
        pred_end = end_idx + self.forecast_horizon

        if pred_end > self.length:
            pred_end = self.length
            end_idx = pred_end - self.forecast_horizon
            start_idx = end_idx - self.context_length

        input_seq = self.data[start_idx:end_idx]
        forecast_seq = self.data[end_idx:pred_end]

        if self.mode == "train":
            return input_seq, forecast_seq
        else:
            label_seq = self.labels[end_idx:pred_end]
            return input_seq, forecast_seq, label_seq

    def __len__(self):
        return (self.length - self.context_length - self.forecast_horizon) // self.stride + 1


def get_loader(root_path, dataset, context_length, forecast_horizon, stride, scale, mode, batch_size):
    dataset = AnomalyDetectionDataset(
        root_path=root_path,
        dataset=dataset,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
        scale=scale,
        mode=mode
    )

    shuffle = mode == 'train'

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

