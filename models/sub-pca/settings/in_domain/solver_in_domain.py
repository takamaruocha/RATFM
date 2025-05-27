import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, ConcatDataset

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.data_loader import get_loader, AnomalyDetectionDataset
from common.metrics import process_scores
from common.period_data_loader import get_time_series_period
from common.period_detection import detect_period_fourier


class SubPCA:
    def __init__(self, n_components, forecast_horizon):
        self.n_components = n_components
        self.forecast_horizon = forecast_horizon
        self.pca = PCA(n_components=n_components)

    def fit(self, data):
        self.pca.fit(data)

    def compute_score(self, data, method):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        reconstructed = self.pca.inverse_transform(self.pca.transform(data))
        error = (data - reconstructed) ** 2 if method == "L2Loss" else np.abs(data - reconstructed)
        return error[:, -self.forecast_horizon:]

    def save(self, path):
        np.savez(path, components=self.pca.components_, mean=self.pca.mean_)

    def load(self, path):
        params = np.load(path)
        self.pca.components_ = params['components']
        self.pca.mean_ = params['mean']
        self.pca.n_features_ = self.pca.components_.shape[1]

class Solver:
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SubPCA(n_components=self.k, forecast_horizon=self.forecast_horizon)
        self.model_save_path = os.path.join(self.model_save_path, "pca_model.npz")

    def train(self, entities):
        print("====================== TRAIN MODE ======================")

        train_datasets = []
        for entity in entities:
            matches = [f for f in os.listdir(self.data_path) if entity in f and f.endswith('.txt')]
            if not matches:
                continue
            dataset_file = matches[0]
            dataset = AnomalyDetectionDataset(
                root_path=self.data_path,
                dataset=dataset_file,
                context_length=self.win_size,
                data_stride_len=self.win_size,
                scale=True
            )
            train_datasets.append(dataset)

        combined_dataset = ConcatDataset(train_datasets)
        loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        all_data = []
        for batch in loader:
            batch = batch.squeeze()
            if batch.ndim == 1:
                batch = batch.reshape(1, -1)
            elif batch.ndim == 3:
                batch = batch.reshape(-1, batch.shape[-1])
            all_data.append(batch)

        all_data = np.concatenate(all_data, axis=0)
        self.model.fit(all_data)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        self.model.save(self.model_save_path)
        print("Sub-PCA model saved.")

    def test(self, entities):
        print("====================== TEST MODE ======================")

        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"Trained PCA model not found at {self.model_save_path}")

        self.model.load(self.model_save_path)

        for entity in entities:
            matches = [f for f in os.listdir(self.data_path) if f not in self.excluded_files and entity in f and f.endswith('.txt')]
            if not matches:
                continue

            dataset = matches[0]

            train_data = get_time_series_period(self.data_path, dataset, scale=True)
            period = detect_period_fourier(train_data)
            print(f"Detected period (Fourier): {period}")

            test_loader = get_loader(
                root_path=self.data_path,
                dataset=dataset,
                context_length=self.win_size,
                forecast_horizon=self.forecast_horizon,
                data_stride_len=self.forecast_horizon,
                scale=True,
                batch_size=self.batch_size,
                mode="test"
            )

            test_labels, anomaly_scores = [], []
            for input_data, labels in tqdm(test_loader):
                input_data = input_data.squeeze().numpy()
                labels = labels.numpy()

                score = self.model.compute_score(input_data, method="L1Loss")
                anomaly_scores.append(score)
                test_labels.append(labels)

            test_labels = np.concatenate(test_labels).reshape(-1)
            anomaly_scores = np.concatenate(anomaly_scores).reshape(-1)

            score_l1 = MinMaxScaler().fit_transform(anomaly_scores.reshape(-1, 1)).ravel()
            smoothed = pd.Series(anomaly_scores).rolling(window=int(period), center=True).mean().fillna(0).to_numpy()
            smoothed = MinMaxScaler().fit_transform(smoothed.reshape(-1, 1)).ravel()

            print(f"[{entity}] ▶ Saving anomaly scores: Raw L1 Loss")
            save_dir_l1 = os.path.join(self.save_base, "L1Loss", self.domain, entity)
            os.makedirs(save_dir_l1, exist_ok=True)
            process_scores(score_l1, test_labels.astype(int), self.step, save_dir_l1)

            print(f"[{entity}] ▶ Saving anomaly scores: Smoothed (SMA + L1 Loss)")
            save_dir_sma = os.path.join(self.save_base, "L1Loss_SMA", self.domain, entity)
            os.makedirs(save_dir_sma, exist_ok=True)
            process_scores(smoothed, test_labels.astype(int), self.step, save_dir_sma)

            print(f"Completed: {dataset}\n")

