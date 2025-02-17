import torch
import torch.nn as nn
import torch.optim as optim
import os
import segmentation_models_pytorch as smp
from training.train_utils import TrainingLogger, train_epoch, validate
from training.metrics import run_inference, SegmentationMetrics
import json
from src.config.base_config import TRAINING, METRICS, LOGS_DIR, CHECKPOINTS_DIR
from src.utils.utils import find_optimal_lr

class CombinedLoss(nn.Module):
    """Combined DiceLoss and CrossEntropyLoss"""
    def __init__(self, ignore_background=True):
        super().__init__()
        self.dice = smp.losses.DiceLoss(
            mode='multiclass',
            classes=[1,2,3] if ignore_background else None,
            from_logits=True
        )
        self.ce = nn.CrossEntropyLoss(
            ignore_index=0 if ignore_background else -100
        )
        
    def forward(self, outputs, targets):
        ce_loss = self.ce(outputs, targets)
        probs = torch.softmax(outputs, dim=1)
        dice_loss = self.dice(probs, targets)
        return dice_loss + ce_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, mode='max'):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): 'min' for loss, 'max' for metrics like IoU
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.min_delta *= 1 if mode == 'max' else -1

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'max':
            delta = current_value - self.best_value
        else:
            delta = self.best_value - current_value

        if delta > self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        experiment_name,
        exp_id,
        hyper_params_experiment,
        device,
        learning_rate=0.001,
        num_epochs=100,
        optimizer=TRAINING['OPTIMIZER'],
        log_dir=LOGS_DIR,
        ignore_background=True,
        training_config=None,
        resume=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.exp_id = exp_id
        self.hyper_params_experiment = hyper_params_experiment
        self.training_config = training_config
        
        # Initialize scaler only for CUDA devices
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Initialize criterion using smp for multi-class segmentation
        if TRAINING['LOSS_FUNCTION'] == 'DiceLoss+CrossEntropyLoss':
            self.criterion = CombinedLoss(ignore_background=ignore_background)
        elif TRAINING['LOSS_FUNCTION'] == 'JaccardLoss':
            self.criterion = smp.losses.JaccardLoss(
                mode='multiclass',
                classes=[1,2,3] if ignore_background else None
            )
        else:
            raise ValueError(f"Unsupported loss function: {TRAINING['LOSS_FUNCTION']}")
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(ignore_background=ignore_background)
        
        # Initialize optimizer based on config
        if optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=training_config['optim_params']['betas'],  # Default Adam betas
                eps=training_config['optim_params']['eps'],  # Default Adam epsilon
                weight_decay=training_config['optim_params']['weight_decay']  # L2 regularization
            )
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=training_config['optim_params']['momentum'],  # Common momentum value
                weight_decay=training_config['optim_params']['weight_decay'],  # L2 regularization
                nesterov=training_config['optim_params']['nesterov']  # Enable Nesterov momentum
            )
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=training_config['optim_params']['betas'],  # Default AdamW betas
                eps=training_config['optim_params']['eps'],  # Default AdamW epsilon
                weight_decay=training_config['optim_params']['weight_decay']  # Default AdamW weight decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Initialize logger with absolute paths
        self.experiment_name = experiment_name
        if hyper_params_experiment:
            self.log_dir = os.path.join(log_dir, exp_id, hyper_params_experiment)
        else:
            self.log_dir = os.path.join(log_dir, exp_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = TrainingLogger(self.log_dir)
        
        # Attach logger to model for batch-level logging
        self.model.logger = self.logger
        self.model.current_epoch = 0  # Add epoch counter to model
        
        # Save experiment configuration
        self.experiment_config = {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'ignore_background': ignore_background,
            'model_name': self.model.__class__.__name__,
            'criterion_name': self.criterion.__class__.__name__,
            'optimizer_name': self.optimizer.__class__.__name__
        }
        
        # Create checkpoint directory with absolute path
        if hyper_params_experiment:
            self.checkpoint_dir = os.path.join(CHECKPOINTS_DIR, exp_id, hyper_params_experiment)
        else:
            self.checkpoint_dir = os.path.join(CHECKPOINTS_DIR, exp_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize scheduler if specified in config
        if training_config and 'SCHEDULER' in training_config:
            if training_config['SCHEDULER'] == 'reduce_lr_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    **training_config['SCHEDULER_PARAMS']
                )
            elif training_config['SCHEDULER'] == 'one_cycle':
                # If using one cycle, first find optimal max_lr
                if not training_config.get('MAX_LR'):
                    suggested_lr = find_optimal_lr(
                        self.model, 
                        self.train_loader,
                        self.criterion,
                        self.optimizer,
                        self.device,
                        log_dir=self.log_dir
                    )
                    max_lr = suggested_lr
                else:
                    max_lr = training_config['MAX_LR']
                    
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=max_lr,
                    epochs=num_epochs,
                    steps_per_epoch=len(self.train_loader),
                    **training_config['SCHEDULER_PARAMS']
                )
                
        # Load checkpoint if resuming
        self.start_epoch = 0
        if resume:
            latest_checkpoint = self._get_latest_checkpoint()
            if latest_checkpoint:
                self.start_epoch = self.load_checkpoint(latest_checkpoint)
                print(f"Resuming from epoch {self.start_epoch}")
        
        self.early_stopping = EarlyStopping(
            patience=7,  # Stop if no improvement for 7 epochs
            min_delta=0.0005,  # Minimum IoU improvement of 0.05%
            mode='max'
        )
        
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        best_val_iou = 0.0
        train_losses, val_losses = [], []
        train_metrics_history, val_metrics_history = [], []
        epoch_times = []
        
        gradient_accumulation_steps = self.training_config.get('GRADIENT_ACCUMULATION_STEPS', 1)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            train_metrics, epoch_time = train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.scheduler if self.training_config['SCHEDULER'] == 'one_cycle' else None,
                gradient_accumulation_steps
            )
            
            try:
                # Validation phase
                val_metrics = validate(
                    self.model,
                    self.val_loader,
                    self.criterion,
                    self.device
                )
                
                # Get building IoU for early stopping
                val_building_iou = val_metrics.get('IoU_Building', 0.0)
                
                # Check early stopping
                if self.early_stopping(val_building_iou):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation Building IoU: {self.early_stopping.best_value:.4f}")
                    break
                
                # Update logs
                self.logger.update_logs(
                    train_metrics,
                    val_metrics,
                    epoch_time
                )
                
                # Save best model
                if val_building_iou > best_val_iou:
                    best_val_iou = val_building_iou
                    self.save_checkpoint(is_best=True)
                
                # Save checkpoint every epoch
                self.save_checkpoint()
                
            except Exception as e:
                print(f"Error during validation at epoch {epoch+1}: {str(e)}")
                # Save checkpoint on error
                self.save_checkpoint(error=True)
                raise e
            
            # Print metrics
            print("Training Metrics:")
            for metric_name, value in train_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            print("\nValidation Metrics:")
            for metric_name, value in val_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            print(f"\nEpoch Time: {epoch_time:.2f} seconds")
            
            # Update scheduler
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
    
    def save_checkpoint(self, is_best=False, error=False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.logger.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'train_loss': self.logger.train_losses[-1] if self.logger.train_losses else None,
            'val_loss': self.logger.val_losses[-1] if self.logger.val_losses else None,
            'config': self.experiment_config
        }
        
        # Save checkpoint with absolute paths
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
        elif error:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'error_checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.logger.current_epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        # Delete previous epoch checkpoints
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith('checkpoint_epoch_') and f != f'checkpoint_epoch_{self.logger.current_epoch}.pth':
                os.remove(os.path.join(self.checkpoint_dir, f))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    
    def inference(self, test_loader, save_predictions=False):
        """Run inference on test set"""
        metrics = run_inference(
            self.model,
            test_loader,
            self.device,
            save_predictions,
            save_dir=os.path.join(self.logger.log_dir, 'predictions')
        )
        
        # Save metrics to JSON
        metrics_path = os.path.join(self.logger.log_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics 
    
    def _get_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return None
            
        # Extract epoch numbers and find latest
        checkpoint_epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
        latest_epoch = max(checkpoint_epochs)
        return os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pth') 