import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support

class Trainer:

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        self.start_epoch = 0
        self.epochs = config.get('epochs', 100)
        self.save_every = config.get('save_every_n_epochs', 10)
        
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.tb_dir = config.get('tensorboard_dir', 'runs')
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        
        resume_path = config.get('resume_from_checkpoint')
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

    def _process_batch(self, batch):
        """Extracts inputs and labels dynamically based on model type."""
        if hasattr(batch, 'to'): 
            batch = batch.to(self.device)
            labels = batch.y
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            labels = batch['labels']
        else:
            raise TypeError("Unrecognized batch format from collator.")
            
        if labels.dim() > 1 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
            
        return batch, labels

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            batch_data, labels = self._process_batch(batch)
            
            outputs = self.model(batch_data)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(all_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = (correct / len(all_labels)) * 100
        
        return avg_loss, accuracy, precision, recall, f1


    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch_data, labels = self._process_batch(batch)
                
                outputs = self.model(batch_data)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        num_samples = max(len(all_labels), 1)
        avg_loss = total_loss / num_samples
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = (correct / num_samples) * 100
        
        return avg_loss, accuracy, precision, recall, f1


    def validate(self, epoch):
        return self._evaluate(self.val_loader)

    def fit(self):
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.start_epoch, self.epochs):
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch(epoch)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(epoch)
            
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            self.writer.add_scalar('F1/Train', train_f1, epoch)
            self.writer.add_scalar('F1/Validation', val_f1, epoch)
            self.writer.add_scalar('Precision/Train', train_prec, epoch)
            self.writer.add_scalar('Precision/Validation', val_prec, epoch)
            self.writer.add_scalar('Recall/Train', train_rec, epoch)
            self.writer.add_scalar('Recall/Validation', val_rec, epoch)
            
            print(f"Epoch {epoch+1:03d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% F1: {val_f1:.4f}")
            
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1)
                
        self.writer.close()
        print("Training Complete!")


    def test(self):
        if self.test_loader is None:
            print("No test loader configured. Skipping test phase.")
            return
            
        print("\n" + "="*50)
        print("RUNNING FINAL EVALUATION ON TEST SET")
        print("="*50)
                
        test_loss, test_acc, test_prec, test_rec, test_f1 = self._evaluate(self.test_loader)
        
        print(f"Test Loss:      {test_loss:.4f}")
        print(f"Test Accuracy:  {test_acc:.2f}%")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall:    {test_rec:.4f}")
        print(f"Test F1 Score:  {test_f1:.4f}")
        print("="*50 + "\n")

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print(f"Resumed training from {path} at epoch {self.start_epoch}")