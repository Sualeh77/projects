import torch
from torch.utils.data import DataLoader
from datasets.custom_datasets.spacenet_dataset import CustomSpacenetDataset
from training.metrics import run_inference
from training.run_experiment import get_transforms
from src.config.base_config import DATASET, DEVICE, TRAINING, CHECKPOINTS_DIR, LOGS_DIR, EXPERIMENTS, BENCHMARK_EXPERIMENTS
from src.models.get_model import get_model
import os
import json
import argparse

def run_inference_experiment(exp_id, hyperparameters_experiment_id, save_predictions, benchmark=False):
    """Run inference for a specific experiment"""
    print(f"Running inference for experiment: {exp_id}")
    
    # Get experiment config
    if benchmark:
        exp_config = BENCHMARK_EXPERIMENTS[exp_id]
    else:
        if exp_id not in EXPERIMENTS:
            raise ValueError(f"Experiment ID {exp_id} not found in config")
        exp_config = EXPERIMENTS[exp_id]
    
    # Create test dataset and dataloader
    test_dataset = CustomSpacenetDataset(
        images_dir=DATASET['TEST_IMAGES_DIR'],
        masks_dir=DATASET['TEST_MASKS_DIR'],
        transform=get_transforms(mode='test', augmentations=TRAINING['AUGMENTATION']['VAL_TEST'])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING['BATCH_SIZE'],
        shuffle=False,
        num_workers=TRAINING['NUM_WORKERS'],
        pin_memory=TRAINING['PIN_MEMORY']
    )
    
    # Initialize model with experiment config
    model = get_model(
        architecture=exp_config['architecture'],
        encoder=exp_config['encoder'],
        encoder_weights=exp_config['encoder_weights'],
        encoder_depth=exp_config['encoder_depth'],
        decoder_channels=exp_config['decoder_channels'],
        decoder_use_batchnorm=exp_config['decoder_use_batchnorm'],
        decoder_attention_type=exp_config['decoder_attention_type'],
        in_channels=exp_config['in_channels'],
        classes=DATASET['NUM_CLASSES'],
        activation=exp_config['activation']
    ).to(DEVICE)
    
    # Load trained model
    if benchmark:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, 'benchmark', exp_id, 'best_model.pth')
    else:
        if hyperparameters_experiment_id:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, exp_id, hyperparameters_experiment_id, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, exp_id, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(f"Checkpoint loaded from {checkpoint_path}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create save directory for predictions
    if benchmark:
        save_dir = os.path.join(LOGS_DIR, 'benchmark', exp_id, 'predictions')
    else:
        if hyperparameters_experiment_id:
            save_dir = os.path.join(LOGS_DIR, exp_id, hyperparameters_experiment_id, 'predictions')
        else:
            save_dir = os.path.join(LOGS_DIR, exp_id, 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving predictions to {save_dir}")
    
    # Run inference
    test_metrics = run_inference(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        save_predictions=save_predictions,
        save_dir=save_dir
    )
    
    # Save metrics to JSON
    if benchmark:
        metrics_path = os.path.join(LOGS_DIR, 'benchmark', exp_id, 'test_metrics.json')
    else:
        if hyperparameters_experiment_id:
            metrics_path = os.path.join(LOGS_DIR, exp_id, hyperparameters_experiment_id, 'test_metrics.json')
        else:
            metrics_path = os.path.join(LOGS_DIR, exp_id, 'test_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"\nTest Metrics for {exp_id}:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run inference for experiments')
    parser.add_argument('--exp_id', type=str, required=True,
                      choices=list(EXPERIMENTS.keys()) + list(BENCHMARK_EXPERIMENTS.keys()),
                      help='Experiment ID from config.py (e.g., unet_resnet18, unet_resnet34)')
    parser.add_argument('--hyperparameters_experiment_id', type=str,
                         help='If provided will use model from this experiment to run inference')
    parser.add_argument('--save_predictions', type=bool, default=False,
                        help='If True, will save predictions to disk')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='If True, will run inference on benchmark experiments')
    args = parser.parse_args()
    
    run_inference_experiment(args.exp_id, args.hyperparameters_experiment_id, args.save_predictions, args.benchmark)

if __name__ == '__main__':
    main() 