import os
import yaml
import argparse
import copy
from src.data.factory import get_dataset

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Generate datasets for Graph Transformer thesis.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML.")
    args = parser.parse_args()

    base_config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    dataset_name = base_config['dataset']['name']
    original_root_dir = base_config['dataset']['root_dir']
    transform_req = base_config['dataset'].get('transform', 'none').lower()

    if transform_req == 'every':
        transforms_to_run = ['graphormer', 'graphgps']
    else:
        transforms_to_run = [transform_req]

    for t_name in transforms_to_run:
        print(f"\n{'=' * 60}")
        print(f" Generating '{dataset_name}' dataset for transform: {t_name.upper()}")
        print(f"{'=' * 60}")

        config = copy.deepcopy(base_config)
        
        config['dataset']['transform'] = t_name
        
        specific_root_dir = os.path.join(original_root_dir, t_name)
        config['dataset']['root_dir'] = specific_root_dir
        
        os.makedirs(specific_root_dir, exist_ok=True)

        print(f"Saving to {specific_root_dir}...")
        
        dataset = get_dataset(config)

        print(f"\n[{t_name.upper()}] Dataset Generation Complete!")
        print("-" * 40)
        
        if dataset_name == 'ba_shapes':
            print("Task Type: Graph Classification (Multiple BA Graphs)")
            print(f"Total Graphs: {len(dataset)}")
            
            splits = dataset.get_split_indices()
            print(f"Train graphs: {len(splits['train'])}")
            print(f"Val graphs:   {len(splits['val'])}")
            print(f"Test graphs:  {len(splits['test'])}")
            print(f"EAP graphs:   {len(splits['eap'])}")
        
        elif dataset_name == 'zinc_no2':
            print("Task Type: Graph Classification (Multiple Molecules)")
            print(f"Total Graphs: {len(dataset)}")
            
            splits = [data.split_mask.item() for data in dataset]
            print(f"Train graphs: {splits.count(0)}")
            print(f"Val graphs:   {splits.count(1)}")
            print(f"Test graphs:  {splits.count(2)}")
            print(f"EAP graphs:   {splits.count(3)}")

if __name__ == "__main__":
    main()