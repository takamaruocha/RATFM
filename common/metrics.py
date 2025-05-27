import os
import json
import numpy as np
from .VUS.metrics import get_metrics

def convert_to_float(data):
    """
    Recursively convert NumPy types to native Python types (for JSON compatibility).
    """
    if isinstance(data, dict):
        return {key: convert_to_float(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_float(element) for element in data]
    elif isinstance(data, np.generic):
        return data.item()
    return data


def process_scores(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    data_stride_len: int,
    save_dir: str,
    save_flag: bool = True
):
    """
    Evaluate anomaly scores against ground truth and optionally save the result.

    Args:
        scores (np.ndarray): Anomaly scores (shape: [N])
        ground_truth (np.ndarray): Ground truth labels (0 for normal, 1 for anomaly)
        data_stride_len (int): Stride length used during windowing (used in metric)
        save_dir (str): Directory to save the evaluation result
        save_flag (bool): Whether to save the result as a JSON file
    """
    assert len(scores) == len(ground_truth), "Score and ground truth must have the same length."

    # Compute metrics
    metrics_result = get_metrics(scores, ground_truth, metric='all', slidingWindow=data_stride_len)

    # Print and collect metrics
    output_data = {}
    for metric_name, metric_value in metrics_result.items():
        print(f"{metric_name} : {metric_value}")
        output_data[metric_name] = metric_value

    # Save to JSON if required
    if save_flag:
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(save_dir, "output_results.json")

        # Ensure JSON-serializable types
        output_data = convert_to_float(output_data)

        with open(output_file, "w") as json_file:
            json.dump(output_data, json_file, indent=4)

