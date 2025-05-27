import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def evaluate_model(model, test_loader, forecast_horizon, device):
    model.eval()
    loss_fn_l1 = nn.L1Loss(reduction='none')

    all_ground_truth = []
    all_forecasts = []
    all_input_series = []
    all_labels = []
    anomaly_scores_l1 = []

    with torch.no_grad():
        for input_series, target_series, label in tqdm(test_loader, total=len(test_loader)):
            input_series = input_series.float().to(device)
            target_series = target_series.float().to(device)
            all_labels.append(label)

            # Forecast future values
            forecast_series = model.generate(input_series, max_new_tokens=forecast_horizon)
            forecast_series = forecast_series[:, -forecast_horizon:]

            # Accumulate forecast and ground truth
            all_forecasts.append(forecast_series.detach().cpu().numpy())
            all_ground_truth.append(target_series.detach().cpu().numpy())
            all_input_series.append(input_series.detach().cpu().numpy())

            # Compute L1 loss as anomaly score
            loss_l1 = loss_fn_l1(target_series, forecast_series)
            anomaly_scores_l1.append(loss_l1.detach().cpu().numpy())

    # Flatten results
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
    anomaly_scores_l1 = np.concatenate(anomaly_scores_l1, axis=0).reshape(-1)

    return anomaly_scores_l1, all_labels

