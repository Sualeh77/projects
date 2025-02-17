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
        # Spatial
        # 'SHIFT_SCALE_ROTATE': True,  # Keep this
        # 'DROP_OUT': True,  # Keep this Val-IOU: 0.8258
        # Pixel
        'BRIGHTNESS_CONTRAST': True,
        'BLUR': True,
        'RANDOM_GAMMA': True,
        # 'CLAHE': True,  # Contrast Limited Adaptive Histogram Equalization
        # 'SHARPEN': True,  # Image sharpening
        # 'COLOR_JITTER': True,  # Random changes in saturation and hue
        # 'ISO_NOISE': True,  # Simulate camera ISO noise
        # 'GAUSSIAN_NOISE': True,  # Add random gaussian noise
        # 'RANDOM_SHADOW': True,  # Add random shadows (helpful for building detection)
        # 'RANDOM_SUNFLARE': True,  # Simulate sun flare effects
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
    'DECODER_USE_BATCHNORM': True,
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
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    'unetpp_resnet50_benchmark': {
        'id': 'unetpp_resnet50_benchmark',
        'name': 'UNet++ with ResNet50 Encoder (Benchmark)', 
        'architecture': 'UnetPlusPlus',
        'encoder': 'resnet50', # since resnet50 in unet++ has 50M parameters equivalent to resnet101 in unet
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': MODEL['DECODER_CHANNELS'],
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    },
    'segformer_b3_benchmark': {
        'id': 'segformer_b3_benchmark',
        'name': 'SegFormer-B3 (Benchmark)',
        'architecture': 'SegFormer',
        'encoder': 'mit_b3',
        'encoder_depth': MODEL['ENCODER_DEPTH'],
        'encoder_weights': MODEL['ENCODER_WEIGHTS'],
        'decoder_channels': 256,
        'decoder_use_batchnorm': MODEL['DECODER_USE_BATCHNORM'],
        'decoder_attention_type': MODEL['DECODER_ATTENTION_TYPE'],
        'in_channels': MODEL['IN_CHANNELS'],
        'activation': MODEL['ACTIVATION']
    }
}

# Training Configuration
TRAINING = {
    'BATCH_SIZE': 4,
    'GRADIENT_ACCUMULATION_STEPS': 2,  # Simulate batch_size of 8
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 0.001, # 1e-3
    'MAX_LR': 1.32E-03, # Found from find_optimal_lr for one cycle scheduler, perfromed best.
    'OPTIMIZER': 'adamw',
    'optim_params': {
            'betas': (0.9, 0.999),
            'eps': 1e-8,  # Default Adam epsilon
            'weight_decay': 0.0001  # L2 regularization
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
            'div_factor': 25.0,
            'final_div_factor': 1e4
        },
    'AUGMENTATION': AUGMENTATION
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
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()