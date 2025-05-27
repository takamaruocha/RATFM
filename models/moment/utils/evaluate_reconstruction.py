import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def evaluate_model(model, test_loader, forecast_horizon, device):
    """
    Evaluate the MOMENT model using L1 reconstruction loss as anomaly score.

    Args:
        model: Trained MOMENT model.
        test_loader: DataLoader providing test sequences, masks, and ground truth labels.
        forecast_horizon: Number of steps to forecast (used to extract tail segment).
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        anomaly_scores_l1: Flattened L1 anomaly scores (1D numpy array).
        all_labels: Flattened ground truth anomaly labels (1D numpy array).
    """
    model.eval()
    loss_fn_l1 = nn.L1Loss(reduction='none')

    all_labels = []
    anomaly_scores_l1 = []

    with torch.no_grad():
        for timeseries, input_mask, labels in tqdm(test_loader, total=len(test_loader)):
            # [B, L] -> [B, 1, L]
            timeseries = timeseries.float().to(device).unsqueeze(1)
            input_mask = input_mask.to(device).long()
            labels = labels.numpy()
            all_labels.append(labels)

            # Expand input mask across channels (B -> B*C)
            input_mask = input_mask.repeat_interleave(timeseries.shape[1], dim=0)

            with torch.cuda.amp.autocast():
                output = model(x_enc=timeseries, input_mask=input_mask)

            # Get forecasted region and ground truth segment
            recon = output.reconstruction[:, :, -forecast_horizon:]  # [B, C, H]
            target = timeseries[:, :, -forecast_horizon:]            # [B, C, H]

            # Compute L1 loss as anomaly score
            loss_l1 = loss_fn_l1(recon, target)  # [B, C, H]
            anomaly_scores_l1.append(loss_l1.detach().cpu().numpy())

    # Flatten outputs
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
    anomaly_scores_l1 = np.concatenate(anomaly_scores_l1, axis=0).reshape(-1)

    return anomaly_scores_l1, all_labels

