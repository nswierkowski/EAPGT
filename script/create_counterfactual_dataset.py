import argparse
import yaml
from src.data.factory import get_dataset
from src.interpretability.counterfactuals.factory import get_counterfactual_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate or load counterfactual dataset")
    parser.add_argument('--config', type=str, required=True, help="Path to the dataset config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading base dataset...")
    base_dataset = get_dataset(config)
    
    counterfactuals = get_counterfactual_dataset(config, base_dataset=base_dataset)
    
    print(f"Process complete. Dataset contains {len(counterfactuals)} counterfactual pairs.")

if __name__ == "__main__":
    main()