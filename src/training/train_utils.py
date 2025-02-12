import torch
from tqdm import tqdm
import json
import os
import time
import numpy as np
from src.training.metrics import (
    SegmentationMetrics, 
    compute_mean_metrics,  # Import from metrics.py
)
from collections import defaultdict
import torch.optim as optim

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        print(f"Initializing logger with directory: {log_dir}")  # Debug print
        self.train_losses = []
        self.train_metrics = defaultdict(list)
        self.val_losses = []
        self.val_metrics = defaultdict(list)
        self.epoch_times = []
        self.current_epoch = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
    def update_logs(self, train_metrics, val_metrics, epoch_time):
        """
        Update logs with metrics dictionaries
        Args:
            train_metrics (dict): Dictionary containing training metrics including loss
            val_metrics (dict): Dictionary containing validation metrics including loss
            epoch_time (float): Time taken for the epoch
        """
        # Update epoch counter
        self.current_epoch += 1
        
        # Debug prints
        print(f"Updating logs for epoch {self.current_epoch}")
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_metrics}")
        
        # Store epoch time
        self.epoch_times.append(float(epoch_time))
        
        # Store losses separately for backward compatibility
        self.train_losses.append(float(train_metrics['loss']))
        self.val_losses.append(float(val_metrics['loss']))
        
        # Store all metrics
        for metric_name, value in train_metrics.items():
            self.train_metrics[metric_name].append(float(value))
        for metric_name, value in val_metrics.items():
            self.val_metrics[metric_name].append(float(value))
        
        # Save logs to JSON file
        log_path = os.path.join(self.log_dir, 'training_logs.json')
        log_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'current_epoch': self.current_epoch,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        print(f"Saving logs to: {log_path}")  # Debug print
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)

def train_epoch(model, data_loader, criterion, optimizer, device, scheduler=None):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    valid_batches = 0  # Counter for valid loss batches
    metrics = SegmentationMetrics()
    all_metrics = []
    
    start_time = time.time()
    num_batches = len(data_loader)
    
    progress_bar = tqdm(data_loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Step scheduler if it's OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        batch_loss = loss.item()
        
        # Skip abnormal loss values
        if not np.isfinite(batch_loss) or batch_loss > 100:
            print(f"Warning: Skipping batch {batch_idx} due to abnormal loss value: {batch_loss}")
            continue
            
        # Update metrics only for valid losses
        running_loss += batch_loss
        valid_batches += 1
        
        batch_metrics = metrics.update(outputs, masks)
        all_metrics.append(batch_metrics)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': batch_loss, 'avg_loss': running_loss/valid_batches})
    
    # Calculate average loss using only valid batches
    avg_loss = running_loss / valid_batches if valid_batches > 0 else float('inf')
    print(f"Final running_loss: {running_loss:.4f}")
    print(f"Number of valid batches: {valid_batches}")
    print(f"Average loss: {avg_loss:.4f}")
    
    epoch_metrics = compute_mean_metrics(all_metrics)
    epoch_metrics['loss'] = avg_loss
    
    return epoch_metrics, time.time() - start_time

def validate(model, data_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    metrics = SegmentationMetrics()
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            running_loss += loss.item()
            batch_metrics = metrics.update(outputs, masks)
            all_metrics.append(batch_metrics)
    
    # Calculate average loss and metrics
    avg_loss = running_loss / len(data_loader)
    epoch_metrics = compute_mean_metrics(all_metrics)  # Using imported function
    epoch_metrics['loss'] = avg_loss  # Add loss to metrics dictionary
    
    return epoch_metrics 