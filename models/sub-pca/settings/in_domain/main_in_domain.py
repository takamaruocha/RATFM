import os
import sys
import json
import argparse
from torch.backends import cudnn
from solver_in_domain import Solver
import yaml


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True

    # Load domain-to-entity mapping
    with open(config.domain_dict_path, 'r', encoding='utf-8') as f:
        domain_dict = json.load(f)

    with open("../../../../common/config.yaml", "r") as f:
        config_common = yaml.safe_load(f)

    config.data_path = config_common["ucr_root_path"]
    config.excluded_files = config_common.get("excluded_files", [])

    # Iterate through each data family
    for domain, groups in domain_dict.items():
        print(f"\n=== Processing Data Family: {domain} ===")

        for train_group in ['group1', 'group2']:
            test_group = 'group2' if train_group == 'group1' else 'group1'

            # Use the opposite group for training, and the current group for testing
            train_entities = set(groups.get(test_group, []))
            test_entities = set(groups.get(train_group, []))

            # Configure save directory and domain
            config.model_save_path = os.path.join(config.save_base, domain, train_group)
            config.domain = domain

            # Initialize solver
            solver = Solver(vars(config))

            # Train the model on the opposite group
            if config.mode == 'train':
                solver.train(train_entities)

            # Test the model on the current group
            if config.mode in ['test', 'train']:
                solver.test(test_entities)

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
    parser.add_argument('--save_base', type=str, default='../../../../results/sub-pca/in_domain/')
    parser.add_argument('--domain_dict_path', type=str, default='../../../../common/domain_groups.json')

    config = parser.parse_args()

    print('------------ Options -------------')
    for k, v in sorted(vars(config).items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------')

    main(config)

