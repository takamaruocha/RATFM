import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.utils import *
from utils.data_loader import get_loader, AnomalyDetectionDataset
from model.AnomalyTransformer import AnomalyTransformer
from DC_Detector.metrics import *
from common.metrics import process_scores
from common.period_data_loader import get_time_series_period
from common.period_detection import detect_period_fourier


def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, f'{self.dataset}_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver:
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, entities):
        """
        Train the Anomaly Transformer using the specified training entities.

        Args:
            entities (list): List of dataset entity prefixes to use for training.
        """
        print("======================TRAIN MODE======================")

        # Prepare training datasets
        train_datasets = []
        for entity in entities:
            matching_files = [
                f for f in os.listdir(self.data_path)
                if entity in f and f.endswith('.txt')
            ]
            if not matching_files:
                continue

            dataset_file = matching_files[0]
            train_dataset = AnomalyDetectionDataset(
                root_path=self.data_path,
                dataset=dataset_file,
                context_length=self.win_size,
                data_stride_len=self.win_size,
                scale=True,
            )
            train_datasets.append(train_dataset)

        # Combine datasets and create DataLoader
        combined_train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(
            dataset=combined_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        time_now = time.time()
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=dataset_file)
        train_steps = len(train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            self.model.train()
            for i, (input_data) in enumerate(train_loader):
                input_data = input_data.unsqueeze(2)
                input = input_data.float().to(self.device)

                self.optimizer.zero_grad()
                iter_count += 1

                output, series, prior, _ = self.model(input)

                # Calculate association discrepancy
                series_loss, prior_loss = 0.0, 0.0
                for u in range(len(prior)):
                    norm_prior = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)
                    series_loss += torch.mean(my_kl_loss(series[u], norm_prior.detach()))
                    series_loss += torch.mean(my_kl_loss(norm_prior.detach(), series[u]))

                    prior_loss += torch.mean(my_kl_loss(norm_prior, series[u].detach()))
                    prior_loss += torch.mean(my_kl_loss(series[u].detach(), norm_prior))

                series_loss /= len(prior)
                prior_loss /= len(prior)

                # Compute reconstruction loss
                rec_loss = self.criterion(
                    output[:, -self.forecast_horizon:, :],
                    input[:, -self.forecast_horizon:, :]
                )

                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                loss1_list.append(loss1.item())

                # Display iteration info
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    remaining_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {remaining_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                # Minimax training
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            # Log average training loss
            train_loss = np.average(loss1_list)

            # Update learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        # Save trained model
        final_model_path = os.path.join(self.save_base, self.domain, 'final_model.pth')
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

    def test(self, entities):
        print("======================TEST MODE======================")

        model_path = os.path.join(self.save_base, self.domain, 'final_model.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        temperature = 50

        for entity in entities:
            matching_files = [
                f for f in os.listdir(self.data_path)
                if f not in self.excluded_files and entity in f and f.endswith('.txt')
            ]
            if not matching_files:
                continue

            dataset = matching_files[0]

            # Load training data to detect period
            train_data = get_time_series_period(
                root_path=self.data_path,
                dataset=dataset,
                scale=True
            )
            period_fourier = detect_period_fourier(train_data)

            print("Detected period:")
            print(f"Fourier-based period: {period_fourier}")

            # Create test loader
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

            criterion_l1 = nn.L1Loss(reduction='none')
            test_labels, attens_energy_l1 = [], []

            # Inference loop
            for input_data, labels in test_loader:
                input_data = input_data.unsqueeze(2)
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)
                loss_l1 = torch.mean(criterion_l1(input, output), dim=-1)

                # Calculate KL divergence-based metric
                series_loss, prior_loss = 0.0, 0.0
                for u in range(len(prior)):
                    norm_prior = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)
                    series_loss += my_kl_loss(series[u], norm_prior.detach()) * temperature
                    prior_loss += my_kl_loss(norm_prior, series[u].detach()) * temperature

                # Attention score weighting
                metric = torch.softmax(-series_loss - prior_loss, dim=-1)
                cri_l1 = (metric * loss_l1).detach().cpu().numpy()[:, -self.forecast_horizon:]

                attens_energy_l1.append(cri_l1)
                test_labels.append(labels)

            # Flatten results
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            anomaly_scores_l1 = np.concatenate(attens_energy_l1, axis=0).reshape(-1)

            # Normalize and smooth scores
            score_l1 = MinMaxScaler().fit_transform(anomaly_scores_l1.reshape(-1, 1)).ravel()
            smoothed_scores = pd.Series(anomaly_scores_l1).rolling(
                window=int(period_fourier), center=True
            ).mean().fillna(0).to_numpy()
            smoothed_scores = MinMaxScaler().fit_transform(smoothed_scores.reshape(-1, 1)).ravel()

            # Save raw L1 anomaly scores
            print(f"[{entity}] ▶ Saving anomaly scores: Raw L1 Loss")
            save_dir_l1 = os.path.join(self.save_base, "L1Loss", self.domain, entity)
            os.makedirs(save_dir_l1, exist_ok=True)
            process_scores(score_l1, test_labels.astype(int), self.step, save_dir_l1)

            # Save smoothed (SMA + L1) anomaly scores
            print(f"[{entity}] ▶ Saving anomaly scores: Smoothed (SMA + L1 Loss)")
            save_dir_sma = os.path.join(self.save_base, "L1Loss_SMA", self.domain, entity)
            os.makedirs(save_dir_sma, exist_ok=True)
            process_scores(smoothed_scores, test_labels.astype(int), self.step, save_dir_sma)

            print(f"Completed: {dataset}")

