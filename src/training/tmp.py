from utils.utils import find_optimal_lr
from models.get_model import get_model
from run_experiment import get_transforms
from config.base_config import DATASET
from datasets.custom_datasets.spacenet_dataset import CustomSpacenetDataset
from torch.utils.data import DataLoader
from config.hyperparameter_config import TRAINING
from training.train import CombinedLoss
import torch.optim as optim

train_transforms = get_transforms('train')
# Initialize datasets and dataloaders
train_dataset = CustomSpacenetDataset(
    images_dir=DATASET['TRAIN_IMAGES_DIR'],
    masks_dir=DATASET['TRAIN_MASKS_DIR'],
    transform=train_transforms,
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAINING['BATCH_SIZE'],
    shuffle=True,
    num_workers=TRAINING['NUM_WORKERS']
)

model = get_model('Unet', 'resnet18', encoder_weights=None, in_channels=3, classes=4)

criterion = CombinedLoss(ignore_background=TRAINING['IGNORE_BACKGROUND'])

optimizer = optim.AdamW(
    model.parameters(),
    lr=TRAINING['LEARNING_RATE'],
    betas=TRAINING['optim_params']['betas'],  # Default AdamW betas
    eps=TRAINING['optim_params']['eps'],  # Default AdamW epsilon
    weight_decay=TRAINING['optim_params']['weight_decay']  # Default AdamW weight decay
)

suggested_lr = find_optimal_lr(model, train_loader, criterion, optimizer, device="mps", 
                   min_lr=1e-7, max_lr=10, num_iter=500, log_dir='logs/lr_finder')

print(suggested_lr)