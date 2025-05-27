import os
import sys
import json
import argparse
import yaml
from torch.backends import cudnn
from solver_time_series_wise import Solver


def str2bool(v):
    return v.lower() in ('true')

def load_configurations(domain_dict_path, common_config_path):
    with open(domain_dict_path, 'r', encoding='utf-8') as f:
        domain_dict = json.load(f)

    with open(common_config_path, 'r') as f:
        common_config = yaml.safe_load(f)

    return domain_dict, common_config["ucr_root_path"], common_config.get("excluded_files", [])

def find_dataset_file(entity, dataset_files):
    matching = [f for f in dataset_files if entity in f and f.endswith('.txt')]
    return matching[0] if matching else None

def main(config):
    cudnn.benchmark = True

    domain_dict, ucr_root_path, excluded_files = load_configurations(
        config.domain_dict_path, "../../../../common/config.yaml"
    )

    dataset_files = sorted(os.listdir(ucr_root_path))

    for domain, entities in domain_dict.items():
        for entity in entities:
            matched_filename = find_dataset_file(entity, dataset_files)
            if not matched_filename:
                print(f"Skipping {entity}: no matching dataset found.")
                continue

            if matched_filename in excluded_files:
                print(f"Skipping {matched_filename}: in excluded list.")
                continue

            dataset_path = os.path.join(ucr_root_path, matched_filename)
            if not os.path.exists(dataset_path):
                print(f"Skipping {dataset_path}: file not found.")
                continue

            print(f"Processing dataset: {entity}")

            config.data_path = ucr_root_path
            config.domain = domain
            config.dataset = matched_filename
            config.entity = entity

            solver = Solver(vars(config))

            if config.mode == 'train':
                solver.train()

            if config.mode in ['test', 'train']:
                solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=512)
    parser.add_argument('--forecast_horizon', type=int, default=96)
    parser.add_argument('--step', type=int, default=96)
    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--save_base', type=str, default='../../../../results/sub-pca/time_series_wise/')
    parser.add_argument('--domain_dict_path', type=str, default='../../../../common/domains_dict.json')

    config = parser.parse_args()

    # Display all configuration options
    print('------------ Options -------------')
    for k, v in sorted(vars(config).items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------')

    main(config)

