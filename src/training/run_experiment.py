import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from datasets.custom_datasets.spacenet_dataset import CustomSpacenetDataset
from training.train import Trainer
from src.utils.utils import compute_mean_std, get_transforms
from src.config.base_config import (
    MODEL, TRAINING, DATASET, DEVICE, AUGMENTATION, LOGS_DIR, 
    EXPERIMENTS, BENCHMARK_EXPERIMENTS
)
from src.models.get_model import get_model
import os
import argparse


def run_experiment(model_conf, training_config=None, learning_rate=None, optimizer=None, scheduler=None, resume=False, log_dir=None):
    device = DEVICE
    
    # Override training config with hyperparameter settings if provided
    if training_config is None:
        training_config = TRAINING.copy()
    if learning_rate is not None:
        training_config['LEARNING_RATE'] = learning_rate
    if optimizer is not None:
        training_config['OPTIMIZER'] = optimizer
    if scheduler is not None:
        training_config['SCHEDULER'] = scheduler
    
    # Get transforms
    train_transforms = get_transforms('train', augmentations=training_config['AUGMENTATION']['TRAIN'])
    val_transforms = get_transforms('val', augmentations=training_config['AUGMENTATION']['VAL_TEST'])
    
    # Initialize datasets and dataloaders
    train_dataset = CustomSpacenetDataset(
        images_dir=DATASET['TRAIN_IMAGES_DIR'],
        masks_dir=DATASET['TRAIN_MASKS_DIR'],
        transform=train_transforms,
    )
    
    val_dataset = CustomSpacenetDataset(
        images_dir=DATASET['VAL_IMAGES_DIR'],
        masks_dir=DATASET['VAL_MASKS_DIR'],
        transform=val_transforms,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['BATCH_SIZE'],
        shuffle=True,
        num_workers=training_config['NUM_WORKERS']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['BATCH_SIZE'],
        shuffle=False,
        num_workers=training_config['NUM_WORKERS']
    )
    
    # Initialize model with experiment config
    model = get_model(
        architecture=model_conf['architecture'],
        encoder=model_conf['encoder'], 
        encoder_weights=model_conf['encoder_weights'],
        encoder_depth=model_conf['encoder_depth'],
        decoder_channels=model_conf['decoder_channels'],
        decoder_use_batchnorm=model_conf['decoder_use_batchnorm'],
        decoder_attention_type=model_conf['decoder_attention_type'],
        in_channels=model_conf['in_channels'],
        classes=DATASET['NUM_CLASSES'],
        activation=model_conf['activation']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=model_conf['name'],
        exp_id=model_conf['id'],
        hyper_params_experiment=model_conf.get('hyper_params_experiment', None),
        device=device,
        learning_rate=training_config['LEARNING_RATE'],
        num_epochs=training_config['NUM_EPOCHS'],
        optimizer=training_config['OPTIMIZER'],
        log_dir=log_dir if log_dir else LOGS_DIR,
        ignore_background=training_config['IGNORE_BACKGROUND'],
        training_config=training_config,
        resume=resume
    )
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training experiment')
    parser.add_argument('--exp_id', type=str, required=True, 
                      help='Experiment ID from config.py (e.g., unet_resnet18, unet_resnet34)')
    parser.add_argument('--benchmark', action='store_true',
                      help='Use benchmark experiments configuration')
    
    # Add optional hyperparameter arguments
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--optimizer', type=str, help='Override optimizer')
    parser.add_argument('--scheduler', type=str, help='Override scheduler')
    
    args = parser.parse_args()

    # Get experiment config
    experiments_dict = BENCHMARK_EXPERIMENTS if args.benchmark else EXPERIMENTS
    if args.exp_id in experiments_dict:
        model_config = experiments_dict[args.exp_id]
        print(f"Running experiment: {model_config['name']}")
    else:
        raise ValueError(f"Invalid {'benchmark ' if args.benchmark else ''}experiment ID: {args.exp_id}")

    # Create experiment directory
    if args.benchmark:
        experiment_dir = os.path.join(LOGS_DIR, 'benchmark', args.exp_id)
    else:
        experiment_dir = os.path.join(LOGS_DIR, args.exp_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # To suppress specific CUDA/GPU related warnings
    if torch.cuda.is_available():
        warnings.filterwarnings('ignore', category=UserWarning)
    
    run_experiment(
        model_conf=model_config, 
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        log_dir=experiment_dir
    )