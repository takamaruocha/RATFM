import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from utils.data_loader import get_loader
from common.metrics import process_scores
from common.period_data_loader import get_time_series_period
from common.period_detection import detect_period_fourier


class SubPCA:
    def __init__(self, n_components, forecast_horizon):
        self.n_components = n_components
        self.forecast_horizon = forecast_horizon
        self.pca = PCA(n_components=n_components)

    def fit(self, data):
        n = min(self.n_components, data.shape[0])
        self.pca = PCA(n_components=n)
        self.pca.fit(data)

    def compute_score(self, data, method):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        reconstructed = self.pca.inverse_transform(self.pca.transform(data))
        error = (data - reconstructed) ** 2 if method == "L2Loss" else np.abs(data - reconstructed)
        return error[:, -self.forecast_horizon:]


class Solver:
    def __init__(self, config):
        self.__dict__.update(config)
        self.model = SubPCA(n_components=self.k, forecast_horizon=self.forecast_horizon)

        # Period estimation
        train_data = get_time_series_period(self.data_path, self.dataset, scale=True)
        self.period_fourier = detect_period_fourier(train_data)

        # Loaders
        self.train_loader = get_loader(
            root_path=self.data_path,
            dataset=self.dataset,
            context_length=self.win_size,
            data_stride_len=self.win_size,
            scale=True,
            batch_size=self.batch_size,
            mode="train"
        )

        self.test_loader = get_loader(
            root_path=self.data_path,
            dataset=self.dataset,
            context_length=self.win_size,
            forecast_horizon=self.forecast_horizon,
            data_stride_len=self.forecast_horizon,
            scale=True,
            batch_size=self.batch_size,
            mode="test"
        )

    def train(self):
        print("====================== TRAIN MODE ======================")

        all_data = []
        for batch in self.train_loader:
            batch = batch.squeeze()
            if batch.ndim == 1:
                batch = batch.reshape(1, -1)
            elif batch.ndim == 3:
                batch = batch.reshape(-1, batch.shape[-1])
            all_data.append(batch)

        all_data = np.concatenate(all_data, axis=0)
        print("all_data", all_data.shape)
        self.model.fit(all_data)

    def test(self):
        print("====================== TEST MODE ======================")

        print(f"Test dataset: {self.dataset}")
        test_labels = []
        attens_energy_l1 = []

        for input_data, labels in tqdm(self.test_loader, desc="Testing"):
            input_data = input_data.squeeze().numpy()
            labels = labels.numpy()

            score_l1 = self.model.compute_score(input_data, method="L1Loss")
            attens_energy_l1.append(score_l1)
            test_labels.append(labels)

        # Concatenate results
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        anomaly_scores_l1 = np.concatenate(attens_energy_l1, axis=0).reshape(-1)

        # Normalize and smooth
        score_l1 = MinMaxScaler().fit_transform(anomaly_scores_l1.reshape(-1, 1)).ravel()
        smoothed_scores = pd.Series(anomaly_scores_l1).rolling(
            window=int(self.period_fourier), center=True
        ).mean().fillna(0).to_numpy()
        smoothed_scores = MinMaxScaler().fit_transform(smoothed_scores.reshape(-1, 1)).ravel()

        # Save raw L1 anomaly scores
        print(f"[{self.entity}] ▶ Saving anomaly scores: Raw L1 Loss")
        save_dir_l1 = os.path.join(self.save_base, "L1Loss", self.domain, self.entity)
        os.makedirs(save_dir_l1, exist_ok=True)
        process_scores(score_l1, test_labels.astype(int), self.step, save_dir_l1)

        # Save smoothed (SMA + L1 Loss) scores
        print(f"[{self.entity}] ▶ Saving anomaly scores: Smoothed (SMA + L1 Loss)")
        save_dir_sma = os.path.join(self.save_base, "L1Loss_SMA", self.domain, self.entity)
        os.makedirs(save_dir_sma, exist_ok=True)
        process_scores(smoothed_scores, test_labels.astype(int), self.step, save_dir_sma)

        print(f"Completed: {self.dataset}")

