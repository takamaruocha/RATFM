import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetectionDataset(Dataset):
    def __init__(self, root_path, dataset, context_length, data_stride_len, scale,
                 mode="train", forecast_horizon=None):
        self.context_length = context_length
        self.data_stride_len = data_stride_len
        self.scale = scale
        self.mode = mode
        self.forecast_horizon = forecast_horizon if forecast_horizon else 0

        self.root_path = root_path
        self.file_name = dataset

        self._read_data()

    def _read_data(self):
        fields = self.file_name.split('_')
        train_end = int(fields[4])
        anomaly_start_in_test = int(fields[5]) - train_end
        anomaly_end_in_test = int(fields[6][:-4]) - train_end

        file_path = os.path.join(self.root_path, self.file_name)
        with open(file_path) as f:
            lines = f.readlines()
            if len(lines) == 1:
                Y = np.array([eval(y) for y in lines[0].strip().split(" ") if len(y) > 0]).reshape((1, -1))
            else:
                Y = np.array([eval(line.strip()) for line in lines]).reshape((1, -1))

        Y = Y.reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(Y[:train_end])
        Y = scaler.transform(Y).squeeze()

        if self.mode == "train":
            self.data = Y[:train_end]
        else:
            self.data = Y[train_end:]
            self.labels = np.zeros_like(self.data)
            self.labels[anomaly_start_in_test:anomaly_end_in_test + 1] = 1

        self.length = len(self.data)

    def __getitem__(self, index):
        if self.mode == "train":
            seq_start = self.data_stride_len * index
            seq_end = seq_start + self.context_length

            if seq_end > self.length:
                seq_end = self.length
                seq_start = seq_end - self.context_length

            timeseries = self.data[seq_start:seq_end]
            return timeseries

        else:
            offset = (1120 - (self.context_length - self.forecast_horizon))
            seq_start = offset + self.data_stride_len * index
            seq_end = seq_start + self.context_length

            if seq_end > self.length:
                seq_end = self.length
                seq_start = seq_end - self.context_length

            timeseries = self.data[seq_start:seq_end]
            label = self.labels[seq_end - self.forecast_horizon:seq_end]
            return timeseries, label

    def __len__(self):
        return (self.length - self.context_length) // self.data_stride_len + 1


def get_loader(root_path, dataset, context_length, data_stride_len, scale,
                        batch_size, mode="train", forecast_horizon=None):
    dataset = AnomalyDetectionDataset(
        root_path=root_path,
        dataset=dataset,
        context_length=context_length,
        data_stride_len=data_stride_len,
        scale=scale,
        mode=mode,
        forecast_horizon=forecast_horizon
    )

    shuffle = mode == "train"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader

