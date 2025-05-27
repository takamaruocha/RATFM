import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from prompt import time_series_to_image
from utils import (
    view_base64_image,
    display_messages,
    collect_results,
    plot_series_and_predictions,
    interval_to_vector,
    compute_metrics,
    process_dataframe,
    highlight_by_ranking,
    styled_df_to_latex,
)
import pickle
import os
from data.synthetic import SyntheticDataset

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from common.config import ucr_root_path, excluded_files


def load_datasets(entity):
    base_dir = "../../data/gpt-4o/"
    data_dir = os.path.join(base_dir, "eval", entity)
    train_dir = os.path.join(base_dir, "train", entity)
    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()
    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()
    return eval_dataset, train_dataset


def compute_metrics_for_results(eval_dataset, results, num_samples):
    metric_names = [
        "precision",
        "recall",
        "f1",
    ]
    results_dict = {key: [[] for _ in metric_names] for key in results.keys()}
    all_gts_preds = {}

    for name, prediction in results.items():
        print(f"valuating {name}")
        gts = []
        preds = []

        print(len(eval_dataset), len(prediction))

        for i in trange(0, num_samples):
            anomaly_locations = eval_dataset[i][0].numpy()
            gt = interval_to_vector(anomaly_locations[0], start=0, end=96)
            gts.append(gt)
            if prediction[i] is None:
                preds.append(np.zeros(len(gt)))
            else:
                preds.append(prediction[i].flatten())

        gts = np.concatenate(gts, axis=0)
        preds = np.concatenate(preds, axis=0)
        all_gts_preds[name] = (gts, preds)

        metrics = compute_metrics(gts, preds)
        for idx, metric_name in enumerate(metric_names):
            results_dict[name][idx].append(metrics[metric_name])

    df = pd.DataFrame(
        {k: np.mean(v, axis=1) for k, v in results_dict.items()},
        index=["precision", "recall", "f1"]#, "affi precision", "affi recall", "affi f1"],
    )
    return df, all_gts_preds


def main(args):
    datasets = os.listdir(ucr_root_path)
    datasets = sorted(datasets, key=lambda x: int(x.split('_')[0]))
    
    data_name = "UCR_Anomaly"

    for dataset in datasets:
        fields = dataset.split('_')
        entity = '_'.join(fields[:4])

        label_name = f"label-{entity}"
        table_caption = f"Evaluation on {entity}"

        print(f"Processing: {entity}")

        eval_dataset, train_dataset = load_datasets(entity)
        directory = f"../../results/gpt-4o/{data_name}/{entity}"
        results = collect_results(directory, ignore=['phi'])

        df, all_gts_preds = compute_metrics_for_results(eval_dataset, results, len(eval_dataset))
        double_df = process_dataframe(df.T.copy())
        print(double_df)

        for (model, variant), row in double_df.iterrows():
            variant_dir = os.path.join("../../results/gpt-4o/", data_name, variant, entity)
            os.makedirs(variant_dir, exist_ok=True)
            output_path = os.path.join(variant_dir, f"output_results.json")

            metric_dict = {
                "model": str(model),
                "variant": str(variant),
                "entity": str(entity),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
             }

            with open(output_path, "w") as jf:
                import json
                json.dump(metric_dict, jf, indent=2)

            print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    main(None)

