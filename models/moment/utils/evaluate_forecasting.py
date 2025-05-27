import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def evaluate_model(model, test_loader, forecast_horizon, context_length, device):
    """
    Evaluate the MOMENT model using L1 forecast error as anomaly score (forecasting setting).

    Args:
        model: Trained MOMENT model.
        test_loader: DataLoader providing test sequences, forecasts, and ground truth labels.
        forecast_horizon: Number of prediction steps.
        context_length: Length of the input context (used to segment inputs).
        device: Torch device.

    Returns:
        anomaly_scores_l1: Flattened L1 anomaly scores (1D numpy array).
        all_labels: Flattened ground truth labels (1D numpy array).
    """
    model.eval()
    loss_fn_l1 = nn.L1Loss(reduction='none')

    all_labels = []
    anomaly_scores_l1 = []

    with torch.no_grad():
        for x, y, input_mask, labels in tqdm(test_loader, total=len(test_loader)):
            x, y, input_mask = x.float().to(device), y.float().to(device), input_mask.to(device)

            x = x.unsqueeze(1)  # [B, 1, L]
            y = y.unsqueeze(1)  # [B, 1, H]
            input_mask = input_mask.long().repeat_interleave(x.shape[1], dim=0)

            with torch.cuda.amp.autocast():
                output = model(x_enc=x, input_mask=input_mask)

            forecast = output.forecast  # [B, C, H]

            # L1 anomaly score
            loss_l1 = loss_fn_l1(forecast, y)  # [B, C, H]
            anomaly_scores_l1.append(loss_l1.detach().cpu().numpy())

            # Labels
            all_labels.append(labels)

    # Flatten outputs
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
    anomaly_scores_l1 = np.concatenate(anomaly_scores_l1, axis=0).reshape(-1)

    return anomaly_scores_l1, all_labels

