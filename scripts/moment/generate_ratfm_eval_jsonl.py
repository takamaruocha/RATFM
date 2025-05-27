import os
import sys
import json
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
import yaml


# Compute normalized cross-correlation (NCC_c)
def calculate_ncc_c(ts1, ts2):
    numerator = np.dot(ts1, ts2)
    denominator = norm(ts1) * norm(ts2)
    if denominator < 1e-9:
        return 0.0
    return numerator / denominator

# Load and standardize training portion from a .txt file
def process_txt_file_train(folder, filename):
    fields = filename.split('_')
    train_end = int(fields[4])

    with open(os.path.join(folder, filename), 'r') as f:
        lines = f.readlines()
        data = (
            [eval(y) for y in lines[0].strip().split(" ") if len(y) > 1]
            if len(lines) == 1
            else [eval(y.strip()) for y in lines]
        )
    series = np.array(data).reshape(-1, 1)
    scaler = StandardScaler().fit(series[:train_end])
    standardized = scaler.transform(series).squeeze()
    return standardized[:train_end]

# Load and standardize test portion from a .txt file, including anomaly labels
def process_txt_file_test(folder, filename):
    fields = filename.split('_')
    train_end = int(fields[4])
    anomaly_start = int(fields[5]) - train_end
    anomaly_end = int(fields[6][:-4]) - train_end

    with open(os.path.join(folder, filename), 'r') as f:
        lines = f.readlines()
        data = (
            [eval(y) for y in lines[0].strip().split(" ") if len(y) > 1]
            if len(lines) == 1
            else [eval(y.strip()) for y in lines]
        )
    series = np.array(data).reshape(-1, 1)
    scaler = StandardScaler().fit(series[:train_end])
    standardized = scaler.transform(series).squeeze()

    test_data = standardized[train_end:]
    labels = np.zeros(len(test_data))
    labels[anomaly_start:anomaly_end + 1] = 1
    return test_data, labels

# Match dataset file names to prefixes
def find_matching_files(domain_dict, target_domain, folder):
    return [f for p in domain_dict[target_domain] for f in os.listdir(folder) if p in f and f.endswith('.txt')]

# Generate aligned test sequences using best-matching training subsequences
def concatenate_sequences(target_file, candidate_files, folder, context_length, forecast_horizon):
    target_data, target_labels = process_txt_file_test(folder, target_file)
    candidate_data = {f: process_txt_file_train(folder, f) for f in candidate_files}

    new_sequences = []
    stride = forecast_horizon

    for start in range(1120 - context_length, len(target_data) - context_length - forecast_horizon + 1, stride):
        seq_start = start
        seq_end = seq_start + context_length
        pred_end = seq_end + forecast_horizon

        if pred_end > len(target_data):
            pred_end = len(target_data)
            seq_end = pred_end - forecast_horizon
            seq_start = seq_end - context_length
            if seq_start < 0:
                continue

        t_seq = target_data[seq_start:seq_end]
        t_pred = target_data[seq_end:pred_end]
        t_label = target_labels[seq_end:pred_end]

        best_score, best_seq, best_pred = -1, None, None
        for data in candidate_data.values():
            for i in range(len(data) - context_length - forecast_horizon + 1):
                c_seq = data[i:i+context_length]
                c_pred = data[i+context_length:i+context_length+forecast_horizon]
                score = calculate_ncc_c(t_seq, c_seq)
                if score > best_score:
                    best_score, best_seq, best_pred = score, c_seq, c_pred

        if best_seq is not None and best_pred is not None:
            combined = list(best_seq) + list(best_pred) + list(t_seq) + list(t_pred)
            new_sequences.append((combined, list(t_label)))

    return new_sequences

# Convert matching results into .jsonl dataset with labels
def convert_to_jsonl(domain, domain_dict, folder, output_file, context_length, forecast_horizon):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    candidates = find_matching_files(domain_dict, domain, folder)
    if not candidates:
        print(f"No matching files found for: {domain}")
        return

    with open(output_file, 'w', encoding='utf-8') as out:
        for target in candidates:
            print(f"Processing {target}")
            others = [f for f in candidates if f != target]
            if not others:
                continue
            sequences = concatenate_sequences(target, others, folder, context_length, forecast_horizon)

            combined_seq = []
            combined_lbl = []
            for seq, lbl in sequences:
                combined_seq.extend(seq)
                combined_lbl.extend(lbl)

            out.write(json.dumps({'sequence': combined_seq, 'label': combined_lbl}) + '\n')

    print(f"Saved to: {output_file}")

def main():
    # 共通設定ファイルの読み込み
    with open("../../common/config.yaml", "r") as f:
        config_common = yaml.safe_load(f)
    ucr_root_path = config_common["ucr_root_path"]

    # データセット固有設定の読み込み
    with open("dataset_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    domain_dict_path = config["domain_dict_path"]
    context_length = config["context_length"]
    forecast_horizon = config["forecast_horizon"]
    output_test_dir = config["output_test_dir"]

    # ドメイン → エンティティマップの読み込み
    with open(domain_dict_path, 'r', encoding='utf-8') as f:
        domain_dict = json.load(f)

    # 各ドメインについてデータ変換を実行
    for domain in domain_dict:
        print(f"Building dataset for: {domain}")
        output_file = os.path.join(output_test_dir, f"{domain}.jsonl")
        convert_to_jsonl(domain, domain_dict, ucr_root_path, output_file, context_length, forecast_horizon)

if __name__ == "__main__":
    main()
