import os
import random
import argparse
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data.factory import get_dataset
from src.data.collator import GraphTransformerCollator
from src.models.factory import get_model
from src.training.trainer import Trainer

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed: int):
    """Ensures completely reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset config")
    parser.add_argument('--model', type=str, required=True, help="Path to model config")
    parser.add_argument('--train', type=str, required=True, help="Path to training config")
    args = parser.parse_args()

    dataset_config = load_config(args.dataset)
    model_config = load_config(args.model)
    train_config = load_config(args.train)

    seed = train_config.get('seed', 42)
    set_seed(seed)
    print(f"Random state initialized with seed: {seed}")

    config = {}
    config.update(dataset_config)
    config['model'] = model_config.get('model', model_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    full_dataset = get_dataset(config)
    
    if hasattr(full_dataset[0], 'split_mask'):
        print("Detected predefined stratified split masks. Applying them...")
        train_dataset = [d for d in full_dataset if d.split_mask.item() == 0]
        val_dataset = [d for d in full_dataset if d.split_mask.item() == 1]
        test_dataset = [d for d in full_dataset if d.split_mask.item() == 2]
    else:
        print("No split_mask detected. Falling back to random_split (80/10/10)...")
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
    
    print(f"Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    collator = GraphTransformerCollator(config)
    batch_size = train_config.get('batch_size', 32)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collator, worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collator, worker_init_fn=seed_worker
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collator, worker_init_fn=seed_worker
    )

    model = get_model(config).to(device)
    
    lr = train_config.get('learning_rate', 1e-3)
    wd = train_config.get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        config=train_config
    )
    
    trainer.fit()    
    trainer.test()

if __name__ == "__main__":
    main()