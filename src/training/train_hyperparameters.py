import argparse
from src.config.base_config import EXPERIMENTS, TRAINING
from src.config.hyperparameter_config import (
    BASE_MODEL_ID,
    LEARNING_RATE_EXPERIMENTS,
    OPTIMIZER_EXPERIMENTS,
    SCHEDULER_EXPERIMENTS,
    AUGMENTATION_EXPERIMENTS
)
from src.training.run_experiment import run_experiment

def run_hyperparams_experiment(exp_type, exp_id, resume=False):
    """Run a hyperparameter experiment"""
    base_model_config = EXPERIMENTS[BASE_MODEL_ID]
    base_model_config.update({"hyper_params_experiment": exp_id})
    exp_training_config = None
    if exp_type == 'learning_rate':
        exp_training_config = LEARNING_RATE_EXPERIMENTS[exp_id]
    elif exp_type == 'optimizer':
        exp_training_config = OPTIMIZER_EXPERIMENTS[exp_id]
    elif exp_type == 'scheduler':
        exp_training_config = SCHEDULER_EXPERIMENTS[exp_id]
    elif exp_type == 'augmentation':
        exp_config = AUGMENTATION_EXPERIMENTS[exp_id]
        # Update training config with augmentations
        exp_training_config = TRAINING.copy()
        exp_training_config['AUGMENTATION'] = exp_config['augmentations']
        exp_training_config['learning_rate'] = exp_training_config.get('LEARNING_RATE')
    else:
        raise ValueError(f"Invalid hyperparameter type: {exp_type}")
    
    run_experiment(
        model_conf=base_model_config,
        training_config=exp_training_config,
        learning_rate=exp_training_config['learning_rate'],
        resume=resume
    )

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning experiments')
    parser.add_argument(
        '--exp_type',
        required=True,
        choices=['learning_rate', 'optimizer', 'scheduler', 'augmentation'],
        help='Type of experiment to run'
    )
    parser.add_argument('--exp_id', required=True, 
                      help='Experiment ID from hyperparameter config')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    args = parser.parse_args()

    if args.exp_type == 'learning_rate':
        if args.exp_id not in LEARNING_RATE_EXPERIMENTS:
            raise ValueError(f"Invalid learning rate experiment: {args.exp_id}")
    elif args.exp_type == 'optimizer':
        if args.exp_id not in OPTIMIZER_EXPERIMENTS:
            raise ValueError(f"Invalid optimizer experiment: {args.exp_id}")
    elif args.exp_type == 'scheduler':
        if args.exp_id not in SCHEDULER_EXPERIMENTS:
            raise ValueError(f"Invalid scheduler experiment: {args.exp_id}")
    
    run_hyperparams_experiment(args.exp_type, args.exp_id, args.resume)

if __name__ == "__main__":
    main()