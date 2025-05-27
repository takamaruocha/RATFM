import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def get_time_series_period(root_path, dataset, scale=True):
    """
    Load and return the training portion of a univariate time series from UCR Anomaly Archive.

    Args:
        root_path (str): Path to the dataset directory.
        dataset (str): Filename of the dataset (e.g., '001_UCR_Anomaly_ECG1_1000_1500_1600.txt').
        scale (bool): Whether to apply standard scaling to the series.

    Returns:
        np.ndarray: Scaled training time series data as a 1D array.
    """
    # Parse metadata from filename
    fields = dataset.split('_')
    metadata = {
        'name': '_'.join(fields[:4]),
        'train_end_index': int(fields[4]),
        'anomaly_start_offset': int(fields[5]) - int(fields[4]),
        'anomaly_end_offset': int(fields[6][:-4]) - int(fields[4]),  # remove .txt
    }

    # Load the time series data
    file_path = os.path.join(root_path, dataset)
    with open(file_path) as f:
        lines = f.readlines()
        if len(lines) == 1:
            values = [eval(v) for v in lines[0].strip().split(" ") if len(v) > 1]
        else:
            values = [eval(line.strip()) for line in lines]

    time_series = np.array(values).reshape(-1, 1)  # (num_points, 1)

    # Extract training segment
    train_series = time_series[:metadata['train_end_index']]

    # Apply scaling if specified
    if scale:
        scaler = StandardScaler()
        scaler.fit(train_series)
        time_series = scaler.transform(time_series)

    return time_series[:metadata['train_end_index']].squeeze()

