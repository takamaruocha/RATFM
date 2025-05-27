import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.utils import *
from utils.data_loader import get_loader
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
        new_lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f'Updating learning rate to {new_lr}')


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
        if self.best_score is None or score > self.best_score + self.delta or score2 > self.best_score2 + self.delta:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join(path, f'{self.dataset}_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver:
    def __init__(self, config):
        self.__dict__.update(config)
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        # Load data and detect period
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

        self.build_model()
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            e_layers=3
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        loss_1, loss_2 = [], []
        for input_data in vali_loader:
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            series_loss, prior_loss = 0.0, 0.0
            for u in range(len(prior)):
                norm_prior = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.forecast_horizon)
                series_loss += torch.mean(my_kl_loss(series[u], norm_prior.detach())) + torch.mean(my_kl_loss(norm_prior.detach(), series[u]))
                prior_loss += torch.mean(my_kl_loss(norm_prior, series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), norm_prior))

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss / len(prior)).item())
            loss_2.append((rec_loss + self.k * prior_loss / len(prior)).item())

        return np.mean(loss_1), np.mean(loss_2)

    def train(self):
        print("====================== TRAIN MODE ======================")

        time_now = time.time()
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            self.model.train()
            loss1_list = []

            for i, input_data in enumerate(self.train_loader):
                input_data = input_data.unsqueeze(2)
                input = input_data.float().to(self.device)

                self.optimizer.zero_grad()
                output, series, prior, _ = self.model(input)

                series_loss, prior_loss = 0.0, 0.0
                for u in range(len(prior)):
                    norm_prior = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)
                    series_loss += torch.mean(my_kl_loss(series[u], norm_prior.detach())) + torch.mean(my_kl_loss(norm_prior.detach(), series[u]))
                    prior_loss += torch.mean(my_kl_loss(norm_prior, series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), norm_prior))

                rec_loss = self.criterion(output, input)
                loss1 = rec_loss - self.k * series_loss / len(prior)
                loss2 = rec_loss + self.k * prior_loss / len(prior)

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

                loss1_list.append(loss1.item())

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {np.mean(loss1_list):.6f}")

        final_model_path = os.path.join(self.save_base, self.domain, self.entity, 'final_model.pth')
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

    def test(self):
        print("====================== TEST MODE ======================")

        self.model.load_state_dict(torch.load(os.path.join(self.save_base, self.domain, self.entity, 'final_model.pth')))
        self.model.eval()

        criterion_l1 = nn.L1Loss(reduction='none')
        temperature = 50
        test_labels, attens_energy_l1 = [], []

        for input_data, labels in self.test_loader:
            input_data = input_data.unsqueeze(2)
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss_l1 = torch.mean(criterion_l1(input, output), dim=-1)

            series_loss, prior_loss = 0.0, 0.0
            for u in range(len(prior)):
                norm_prior = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True).repeat(1, 1, 1, self.win_size)
                series_loss += my_kl_loss(series[u], norm_prior.detach()) * temperature
                prior_loss += my_kl_loss(norm_prior, series[u].detach()) * temperature

            metric = torch.softmax(-series_loss - prior_loss, dim=-1)
            cri_l1 = (metric * loss_l1).detach().cpu().numpy()[:, -self.forecast_horizon:]

            attens_energy_l1.append(cri_l1)
            test_labels.append(labels)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        anomaly_scores_l1 = np.concatenate(attens_energy_l1, axis=0).reshape(-1)

        score_l1 = MinMaxScaler().fit_transform(anomaly_scores_l1.reshape(-1, 1)).ravel()
        smoothed_scores = pd.Series(anomaly_scores_l1).rolling(window=int(self.period_fourier), center=True).mean().fillna(0).to_numpy()
        smoothed_scores = MinMaxScaler().fit_transform(smoothed_scores.reshape(-1, 1)).ravel()

        # Save raw L1 anomaly scores
        print(f"[{self.entity}] ▶ Saving anomaly scores: Raw L1 Loss")
        save_dir_l1 = os.path.join(self.save_base, "L1Loss", self.domain, self.entity)
        os.makedirs(save_dir_l1, exist_ok=True)
        process_scores(score_l1, test_labels.astype(int), self.step, save_dir_l1)

        # Save smoothed (SMA + L1) anomaly scores
        print(f"[{self.entity}] ▶ Saving anomaly scores: Smoothed (SMA + L1 Loss)")
        save_dir_sma = os.path.join(self.save_base, "L1Loss_SMA", self.domain, self.entity)
        os.makedirs(save_dir_sma, exist_ok=True)
        process_scores(smoothed_scores, test_labels.astype(int), self.step, save_dir_sma)

        print(f"Completed: {self.dataset}")

