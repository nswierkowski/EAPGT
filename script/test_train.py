import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.factory import get_dataset
from src.data.collator import GraphTransformerCollator
from src.models.factory import get_model

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    args = parser.parse_args()

    dataset_config = load_config(args.dataset_config)
    model_config = load_config(args.model_config)

    config = {}
    config.update(dataset_config)
    config['model'] = model_config.get('model', model_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Testing Model: {config['model'].get('name', 'graphormer').upper()}")

    dataset = get_dataset(config)
    collator = GraphTransformerCollator(config)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collator
    )
    
    single_batch = next(iter(dataloader))

    model = get_model(config).to(device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting Single-Batch Overfit Test ---")
    
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if hasattr(single_batch, 'to'): 
            batch = single_batch.to(device)
            labels = batch.y
            outputs = model(batch)
            
        elif isinstance(single_batch, dict):
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in single_batch.items()}
            labels = batch['labels']
            outputs = model(batch)
            
        else:
            raise TypeError("Unrecognized batch format from collator.")

        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        if labels.dim() > 1 and labels.size(1) == 1:
            labels = labels.squeeze(-1)
            
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = (correct / len(labels)) * 100

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.1f}%")

    print("\nTest Complete!")
    if loss.item() < 0.1 and accuracy > 90:
        print("SUCCESS: The model successfully overfit the batch.")
    else:
        print("WARNING: The model struggled to overfit.")

if __name__ == "__main__":
    main()