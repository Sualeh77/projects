import os
import torch

# Get the project root directory (bfp-segmentation)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get parent directory of project root (Dissertation)
DISSERTATION_ROOT = os.path.dirname(PROJECT_ROOT)

# Define paths relative to Dissertation root
DATASET_ROOT = os.path.join(DISSERTATION_ROOT, 'datasets')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# Dataset paths
DATA_ROOT = os.path.join(
    DATASET_ROOT, 
    "SpaceNet/AOI_3_Paris_Train/SN2_buildings/processed/multi_channel_masks"
)

# Make sure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Dataset Configuration
DATASET = {
    'TRAIN_IMAGES_DIR': os.path.join(DATA_ROOT, "train/images"),
    'TRAIN_MASKS_DIR': os.path.join(DATA_ROOT, "train/masks"),
    'VAL_IMAGES_DIR': os.path.join(DATA_ROOT, "val/images"),
    'VAL_MASKS_DIR': os.path.join(DATA_ROOT, "val/masks"),
    'TEST_IMAGES_DIR': os.path.join(DATA_ROOT, "test/images"),
    'TEST_MASKS_DIR': os.path.join(DATA_ROOT, "test/masks"),
    'IMAGE_SIZE': 512,  # Size to resize images to
    'NUM_CLASSES': 4,
    'CLASS_LABELS': ['Background', 'Building', 'Boundary', 'Contact Point'],
    'CLASS_COLORS': [  # RGB format
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Building - red
        [0, 255, 0],      # Boundary - green
        [0, 0, 255]       # Contact - blue
    ],
}

# Augmentation Configuration
# keeping only basic pixel-level augmentations as it performed best.
AUGMENTATION = {
    'TRAIN': {
        'RESIZE': True,
        'HORIZONTAL_FLIP': True,
        'VERTICAL_FLIP': True,
        'RANDOM_BRIGHTNESS_CONTRAST': True,
        'RANDOM_GAMMA': True,
        'BLUR': True,
        'SHIFT_SCALE_ROTATE': {
            'shift_limit': 0.0625,
            'scale_limit': 0.1,
            'rotate_limit': 15,
            'p': 0.5
        },
        'NORMALIZE': True
    },
    'VAL_TEST': {
        'RESIZE': True,
        'NORMALIZE': True
    }
}

SEGFORMER_AUGMENTATION = {
    'TRAIN': {
        'RESIZE': True,
        'D4': True,
        'CROP': True,
        'SHIFT_SCALE_ROTATE': True,
        'DISTORTION': True,
        'RANDOM_BRIGHTNESS_CONTRAST': True,
        'CLAHE': True,
        'GAUSSIAN_NOISE': True,
        'DROP_OUT': True,
        'NORMALIZE': True
    },
    'VAL_TEST': {
        'RESIZE': True,
        'NORMALIZE': True
    }
}

# Model Configuration
MODEL = {
    'ENCODER_DEPTH':5,
    'ENCODER_WEIGHTS': None,
    'DECODER_CHANNELS': [256, 128, 64, 32, 16],
    'DECODER_USE_BATCHNORM': False,
    'DECODER_ATTENTION_TYPE': None,  # scse, cbam, se, none
    'IN_CHANNELS': 3,
    'ACTIVATION': None, # Keep as None since we handle activations in loss function
}

# Add new EXPERIMENTS configuration
EXPERIMENTS = {
    'unet_resnet18': {
        'name': 'UNet with ResNet18 Encoder',
        'architecture': 'Unet',
        'encoder': 'resnet18',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    'unet_resnet34': {
        'name': 'UNet with ResNet34 Encoder',
        'architecture': 'Unet',
        'encoder': 'resnet34',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    'unet_resnet50': {
        'name': 'UNet with ResNet50 Encoder',
        'architecture': 'Unet',
        'encoder': 'resnet50',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    'unet_resnet101': {
        'id': 'unet_resnet101',
        'name': 'UNet with ResNet101 Encoder',
        'architecture': 'Unet',
        'encoder': 'resnet101',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    'unet_resnext50_32x4d': {
        'name': 'UNet with ResNeXt50 32x4d Encoder',
        'architecture': 'Unet',
        'encoder': 'resnext50_32x4d',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    }
}

BENCHMARK_EXPERIMENTS = {
    'unet_resnet101_benchmark': {
        'id': 'unet_resnet101_benchmark',
        'name': 'UNet with ResNet101 Encoder (Benchmark)',
        'architecture': 'Unet',
        'encoder': 'resnet101',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': False,
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    'unetpp_resnet50_benchmark': {
        'id': 'unetpp_resnet50_benchmark',
        'name': 'UNet++ with ResNet50 Encoder (Benchmark)', 
        'architecture': 'UnetPlusPlus',
        'encoder': 'resnet50',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': True,
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    # 'segformer_b2_benchmark': {
    #     'id': 'segformer_b2_benchmark',
    #     'name': 'SegFormer-B2 (Benchmark)',
    #     'architecture': 'SegFormer',
    #     'encoder': 'mit_b2',
    #     'encoder_depth': MODEL['ENCODER_DEPTH'],
    #     'encoder_weights': 'imagenet',
    #     'decoder_channels': 768,
    #     'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
    #     'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
    #     'in_channels': MODEL['IN_CHANNELS'],
    #     'activation': MODEL['ACTIVATION'],
    #     'aux_params': {
    #         'dropout': 0.1
    #     }
    # },
    # 'segformer_b0_benchmark': {
    #     'id': 'segformer_b0_benchmark',
    #     'name': 'SegFormer-B0 (Benchmark)',
    #     'architecture': 'SegFormer',
    #     'encoder': 'mit_b0',
    #     'encoder_depth': 4,  # SegFormer has 4 stages
    #     'decoder_channels': [32, 64, 160, 256],  # B0 dimensions
    #     'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
    #     'decoder_attention_type': 'mlp',
    #     'in_channels': MODEL['IN_CHANNELS'],
    #     'activation': MODEL['ACTIVATION'],
    #     'decoder_embed_dim': 256,
    #     'decoder_mlp_channels': 256,  # B0 channels
    #     'aux_params': {
    #         'dropout': 0.1
    #     }
    # },
    'segformer_b3_benchmark': {
        'id': 'segformer_b3_benchmark',
        'name': 'SegFormer-B3 (Benchmark)',
        'architecture': 'SegFormer',
        'encoder': 'mit_b3',
        'encoder_depth': MODEL['ENCODER_DEPTH'],  # SegFormer-B3 has 4 stages
        'encoder_weights': 'imagenet',
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_channels': 256,
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION'],
    },
    'deeplabv3plus_resnet101_benchmark': {
        'id': 'deeplabv3plus_resnet101_benchmark',
        'name': 'DeepLabV3+ with ResNet101 Encoder (Benchmark)',
        'architecture': 'DeepLabV3Plus',
        'encoder': 'resnet101',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'encoder_output_stride': 16,  # Essential for DeepLabV3+
        'decoder_channels': 256,
        'decoder_atrous_rates': (6, 12, 18),  # Reduced rates for our image size
        'decoder_aspp_separable': False,  # Start with standard convolutions
        'decoder_aspp_dropout': 0.2,  # Reduced dropout
        'in_channels': MODEL['IN_CHANNELS'],
        'decoder_use_batchnorm': True,
        'decoder_attention_type': None,
        'activation': MODEL['ACTIVATION'],
    },
    'clipseg': {
        'id': 'clipseg',
        'name': 'CLIPSeg RD64 Refined',
        'architecture': 'CLIPSeg',
        'model_id': 'CIDAS/clipseg-rd64-refined',
        'threshold': 0.5,  # Confidence threshold
        'post_process': True  # Always post-process CLIPSeg outputs
    },
    'sam2': {
        'name': 'SAM2',
        'id': 'sam2',
        'architecture': 'sam2',
        'post_process': False,  # SAM2 doesn't need post-processing
        'model_cfg': '../../sam_finetuning/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml'  # Going up two levels to Dissertation
    }
}

# Training Configuration
TRAINING = {
    'BATCH_SIZE': 4,  # Reduce batch size
    'GRADIENT_ACCUMULATION_STEPS': 2,  # Accumulate for stability
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 0.0001,  # Lower learning rate
    'OPTIMIZER': 'adamw',
    'optim_params': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01  # Increase regularization
    },
    'LOSS_FUNCTION': 'DiceLoss+CrossEntropyLoss',
    'IGNORE_BACKGROUND': False,
    'NUM_WORKERS': 8,
    'PIN_MEMORY': True,
    'SAVE_CHECKPOINT_FREQ': 5,  # Save checkpoint every N epochs
    'SCHEDULER': 'one_cycle',
    'SCHEDULER_PARAMS': {
        'pct_start': 0.5,
        'anneal_strategy': 'linear',
        'cycle_momentum': True,
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 10.0,  # Less aggressive LR range
        'final_div_factor': 1e4
    },
    'AUGMENTATION': AUGMENTATION
}

# Add SegFormer specific configuration
SEGFORMER_TRAINING_CONFIG = {
    'BATCH_SIZE': 8,  # They used 8 for high-res images like Cityscapes
    'GRADIENT_ACCUMULATION_STEPS': 1,
    'NUM_EPOCHS': 50,  # They used 160K iterations
    'LEARNING_RATE': 0.00006,  # This matches their setting
    'OPTIMIZER': 'adamw',
    'optim_params': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01
    },
    # 'MAX_LR': 0.0012,
    'SCHEDULER': 'reduce_lr_on_plateau',
    'SCHEDULER_PARAMS': {
        'mode': 'max',           # Monitor IoU which we want to maximize
        'factor': 0.5,          # Reduce LR by half when plateauing
        'patience': 5,          # Number of epochs with no improvement
        'min_lr': 1e-7,        # Don't reduce LR below this
        'threshold': 1e-4,      # Minimum change to qualify as an improvement
        'threshold_mode': 'rel', # Use relative change
        'cooldown': 2,         # Number of epochs to wait before resuming normal operation
        'verbose': True        # Print message when reducing LR
    },
    'LOSS_FUNCTION': 'DiceLoss+CrossEntropyLoss',
    'IGNORE_BACKGROUND': False,
    'NUM_WORKERS': 8,
    'PIN_MEMORY': True,
    'AUGMENTATION': SEGFORMER_AUGMENTATION,
} 

# Metrics Configuration
METRICS = {
    'COMPUTE_CLASS_WISE': True,
    'COMPUTE_BUILDING_ONLY': True,
    'BUILDING_CLASS_INDEX': 1,
    'METRICS_LIST': [
        'IoU',
        'F1_Score',
        'Accuracy',
        'Precision',
        'Recall'
    ],
    'THRESHOLD': None,  # Not needed for multiclass
    'ACTIVATION': 'softmax2d'  # This will be converted to 'mode' in the metrics
}

# Web Application Configuration
WEBAPP = {
    'HOST': '0.0.0.0',
    'PORT': 8000,
    'DEBUG': True,
    'CHART_UPDATE_INTERVAL': 10000,  # milliseconds
    'EXPERIMENT_NAMES': ['unet_resnet18'],  # List of experiments to display
}

# Logging Configuration
LOGGING = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_TO_FILE': True,
    'LOG_FILE': os.path.join(LOGS_DIR, 'training.log'),
}

# Visualization Configuration
VISUALIZATION = {
    'SAVE_PREDICTIONS': True,
    'SAVE_FREQUENCY': 10,  # Save visualizations every N epochs
    'DPI': 300,
    'PLOT_SIZE': (10, 10),
    'GRID_SIZE': (2, 2),  # For subplot layout
}

# Device Configuration
def get_device():
    # Set MPS fallback for unsupported operations
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# Add upsampling mode configuration - bilinear is more compatible with MPS
UPSAMPLING_MODE = 'bilinear'  # Options: 'bilinear', 'bicubic', 'nearest'

# Function to patch model with bilinear upsampling
def patch_upsampling_mode(model):
    """
    Patches a PyTorch model to use the configured upsampling mode.
    Helps with MPS compatibility issues.
    """
    import torch.nn as nn
    
    # Find all upsample modules and replace their mode
    for module in model.modules():
        if hasattr(module, 'mode') and hasattr(module, 'align_corners') and hasattr(module, 'upsample'):
            # This is likely an upsampling module
            module.mode = UPSAMPLING_MODE
        
        # For functional calls within forward methods, we can't directly patch
        # Users should manually replace F.interpolate calls with the right mode
    
    return model