from src.config.base_config import TRAINING

# Selected base model from UNet experiments
BASE_MODEL_ID = 'unet_resnet101'

# Augmentation Experiments
AUGMENTATION_EXPERIMENTS = {
    'spatial_basic': {
        'id': 'spatial_basic',
        'name': 'Basic Spatial Augmentations',
        'augmentations': {
            'TRAIN': {
                'RESIZE': True,
                'D4': True,
                'CROP': True,
                'SHIFT_SCALE_ROTATE': True,
                'NORMALIZE': True
            },
            'VAL_TEST': {
                'RESIZE': True,
                'NORMALIZE': True
            }
        }
    },
    'spatial_advanced': {
        'id': 'spatial_advanced',
        'name': 'Advanced Spatial Augmentations',
        'augmentations': {
            'TRAIN': {
                'RESIZE': True,
                'D4': True,
                'CROP': True,
                'SHIFT_SCALE_ROTATE': True,
                'DISTORTION': True,
                'DROP_OUT': True,
                'NORMALIZE': True
            },
            'VAL_TEST': {
                'RESIZE': True,
                'NORMALIZE': True
            }
        }
    },
    'pixel_basic': {
        'id': 'pixel_basic',
        'name': 'Basic Pixel-Level Augmentations',
        'augmentations': {
            'TRAIN': {
                'RESIZE': True,
                'BRIGHTNESS_CONTRAST': True,
                'BLUR': True,
                'RANDOM_GAMMA': True,
                'NORMALIZE': True
            },
            'VAL_TEST': {
                'RESIZE': True,
                'NORMALIZE': True
            }
        }
    },
    'combined_optimal': {
        'id': 'combined_optimal',
        'name': 'Combined Optimal Augmentations',
        'augmentations': {
            'TRAIN': {
                'RESIZE': True,
                # Spatial
                'SHIFT_SCALE_ROTATE': True,  # Keep this
                'DROP_OUT': True,  # Keep this Val-IOU: 0.8258
                # Pixel
                'BRIGHTNESS_CONTRAST': True,
                'BLUR': True,
                'RANDOM_GAMMA': True,
                'CLAHE': True,  # Contrast Limited Adaptive Histogram Equalization
                'SHARPEN': True,  # Image sharpening
                'COLOR_JITTER': True,  # Random changes in saturation and hue
                'ISO_NOISE': True,  # Simulate camera ISO noise
                'GAUSSIAN_NOISE': True,  # Add random gaussian noise
                'RANDOM_SHADOW': True,  # Add random shadows (helpful for building detection)
                'RANDOM_SUNFLARE': True,  # Simulate sun flare effects
                'NORMALIZE': True
            },
            'VAL_TEST': {
                'RESIZE': True,
                'NORMALIZE': True
            }
        }
    }
}

# Hyperparameter Tuning Experiments
LEARNING_RATE_EXPERIMENTS = {
    'lr_1e-5': {
        'id': 'lr_1e-5',
        'name': 'Learning Rate 1e-5',
        'learning_rate': 1e-5,
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'OPTIMIZER': TRAINING['OPTIMIZER'],
        'optim_params': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,  # Default Adam epsilon
            'weight_decay': 0.0001  # L2 regularization
        },
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'AUGMENTATION': TRAINING['AUGMENTATION']
    },
    'lr_1e-6': {
        'id': 'lr_1e-6',
        'name': 'Learning Rate 1e-6',
        'learning_rate': 1e-6,
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'OPTIMIZER': TRAINING['OPTIMIZER'],
        'optim_params': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,  # Default Adam epsilon
            'weight_decay': 0.0001  # L2 regularization
        },
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'AUGMENTATION': TRAINING['AUGMENTATION']
    },
    'lr_1e-4': {
        'id': 'lr_1e-4',
        'name': 'Learning Rate 1e-4',
        'learning_rate': 1e-4,
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'OPTIMIZER': TRAINING['OPTIMIZER'],
        'optim_params': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,  # Default Adam epsilon
            'weight_decay': 0.0001  # L2 regularization
        },
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'AUGMENTATION': TRAINING['AUGMENTATION']
    }
}

OPTIMIZER_EXPERIMENTS = {
    'adamw': {
        'id': 'adamw',
        'name': 'AdamW Optimizer',
        'OPTIMIZER': 'adamw',
        'learning_rate': TRAINING['LEARNING_RATE'],
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'optim_params': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,  # Default Adam epsilon
            'weight_decay': 0.0001  # L2 regularization
        },
        'AUGMENTATION': TRAINING['AUGMENTATION']
    },
    'sgd': {
        'id': 'sgd',
        'name': 'SGD Optimizer',
        'OPTIMIZER': 'sgd',
        'learning_rate': TRAINING['LEARNING_RATE'],
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'optim_params': {
            'momentum': 0,
            'nesterov': False,
            'weight_decay': 0.0001
        },
        'AUGMENTATION': TRAINING['AUGMENTATION']
    },
    'sgd_nesterov_momentum': {
        'id': 'sgd_nesterov_momentum',
        'name': 'SGD with Nesterov Momentum',
        'OPTIMIZER': 'sgd',
        'learning_rate': TRAINING['LEARNING_RATE'],
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'optim_params': {
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 0.0001
        },
        'AUGMENTATION': TRAINING['AUGMENTATION']
    }
}

SCHEDULER_EXPERIMENTS = {
    'reduce_lr_on_plateau': {
        'id': 'reduce_lr_on_plateau',
        'name': 'Reduce LR on Plateau',
        'SCHEDULER': 'reduce_lr_on_plateau',
        'learning_rate': TRAINING['LEARNING_RATE'],
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'OPTIMIZER': TRAINING['OPTIMIZER'],
        'optim_params': TRAINING['optim_params'],
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'SCHEDULER_PARAMS': {
            'mode': 'min',
            'factor': 0.1,
            'patience': 3,
            'threshold_mode': 'rel',
            'min_lr': 1e-6,
            'verbose': True
        },
        'AUGMENTATION': TRAINING['AUGMENTATION']
    },
    'one_cycle': {
        'id': 'one_cycle',
        'name': 'One Cycle Policy',
        'SCHEDULER': 'one_cycle',
        'learning_rate': TRAINING['LEARNING_RATE'],  # This will be updated by LR finder
        'BATCH_SIZE': TRAINING['BATCH_SIZE'],
        'NUM_EPOCHS': TRAINING['NUM_EPOCHS'],
        'OPTIMIZER': TRAINING['OPTIMIZER'],
        'optim_params': TRAINING['optim_params'],
        'LOSS_FUNCTION': TRAINING['LOSS_FUNCTION'],
        'IGNORE_BACKGROUND': TRAINING['IGNORE_BACKGROUND'],
        'NUM_WORKERS': TRAINING['NUM_WORKERS'],
        'PIN_MEMORY': TRAINING['PIN_MEMORY'],
        'SAVE_CHECKPOINT_FREQ': TRAINING['SAVE_CHECKPOINT_FREQ'],
        'SCHEDULER_PARAMS': {
            'pct_start': 0.5,
            'anneal_strategy': 'linear',
            'cycle_momentum': True,
            'base_momentum': 0.85,
            'max_momentum': 0.95,
            'div_factor': 25.0,
            'final_div_factor': 1e4
        },
        'AUGMENTATION': TRAINING['AUGMENTATION']
    }
} 