# utils.py
import os
import numpy as np
import shutil
from tqdm import tqdm
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config.base_config import DATASET, AUGMENTATION

def get_transforms(mode='train', augmentations=None):
    """
    Get transformations for different modes
    Args:
        mode: 'train', 'val', or 'test'
    """
    mean, std = compute_mean_std(DATASET['TRAIN_IMAGES_DIR'])

    if mode == 'train':
        transforms_list = []
        
        # Add transforms based on config
        if augmentations.get('RESIZE', False):
            transforms_list.append(A.Resize(DATASET['IMAGE_SIZE'], DATASET['IMAGE_SIZE']))
            
        if augmentations.get('D4', False):
            transforms_list.append(A.D4(p=0.5))

        if augmentations.get('CROP', False):
            transforms_list.append(
                A.OneOf([
                    A.Compose([
                        A.RandomCrop(height=DATASET['IMAGE_SIZE']//2, width=DATASET['IMAGE_SIZE']//2),
                        A.PadIfNeeded(
                            min_height=DATASET['IMAGE_SIZE'],
                            min_width=DATASET['IMAGE_SIZE']
                        )
                    ]),
                    A.Compose([
                        A.CenterCrop(height=DATASET['IMAGE_SIZE']//2, width=DATASET['IMAGE_SIZE']//2),
                        A.PadIfNeeded(
                            min_height=DATASET['IMAGE_SIZE'],
                            min_width=DATASET['IMAGE_SIZE']
                        )
                    ]),
                    A.Compose([
                        A.CropNonEmptyMaskIfExists(height=400, width=400, ignore_channels=[0, 2, 3]),
                        A.Resize(height=DATASET['IMAGE_SIZE'], width=DATASET['IMAGE_SIZE'])
                    ])
                ], p=0.2)
            )

        if augmentations.get('SHIFT_SCALE_ROTATE', False):
            transforms_list.append(A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.3))

        if augmentations.get('DISTORTION', False):
            transforms_list.append(
                A.OneOf([
                    A.GridDistortion(num_steps=10, distort_limit=0.2),
                    A.OpticalDistortion(distort_limit=0.5)
                ], p=0.1)
            )

        if augmentations.get('DROP_OUT', False):
            transforms_list.append(
                A.OneOf([
                    A.GridDropout(ratio=0.05),
                    A.CoarseDropout(num_holes_range=(1, 5), max_height=2, max_width=2, fill_mask=0, p=1),
                    A.PixelDropout(p=1, dropout_prob=0.05)
                ], p=0.05)
            )

        if augmentations.get('RANDOM_GAMMA', False):
            transforms_list.append(A.RandomGamma(gamma_limit=(50, 150), p=0.2))

        if augmentations.get('BRIGHTNESS_CONTRAST', False):
            transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2))

        if augmentations.get('BLUR', False):
            transforms_list.append(A.Blur(blur_limit=7, p=0.2))

        if augmentations.get('CLAHE', False):
            transforms_list.append(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5))

        if augmentations.get('SHARPEN', False):
            transforms_list.append(A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5))

        if augmentations.get('COLOR_JITTER', False):
            transforms_list.append(
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                )
            )

        if augmentations.get('ISO_NOISE', False):
            transforms_list.append(
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=0.5
                )
            )

        if augmentations.get('GAUSSIAN_NOISE', False):
            transforms_list.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.5))

        if augmentations.get('RANDOM_SHADOW', False):
            transforms_list.append(
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=3,
                    shadow_dimension=5,
                    p=0.5
                )
            )

        if augmentations.get('RANDOM_SUNFLARE', False):
            transforms_list.append(
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=6,
                    num_flare_circles_upper=10,
                    src_radius=400,
                    src_color=(255, 255, 255),
                    p=0.3
                )
            )

        if augmentations.get('NORMALIZE', False):
            transforms_list.append(A.Normalize(mean=mean, std=std))
            
        transforms_list.append(ToTensorV2())
        
        return A.Compose(transforms_list)
    else:  # val or test
        transforms_list = []
        
        if augmentations.get('RESIZE', False):
            transforms_list.append(A.Resize(DATASET['IMAGE_SIZE'], DATASET['IMAGE_SIZE']))
            
        if augmentations.get('NORMALIZE', False):
            transforms_list.append(A.Normalize(mean=mean, std=std))
            
        transforms_list.append(ToTensorV2())
        
        return A.Compose(transforms_list)

def get_mask_transforms():
    """Get transformations for masks"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

def compute_mean_std(images_dir):
    """
    Compute the mean and standard deviation across all images in a directory.
    
    This function loads .npy image files from the specified directory, normalizes them to [0,1] range,
    and computes the channel-wise mean and standard deviation across all images.
    
    Args:
        images_dir (str): Path to directory containing .npy image files
        
    Returns:
        tuple: Returns (mean, std) where both mean and std are numpy arrays of shape (3,)
              containing the channel-wise statistics
              
    Example:
        >>> image_directory = "path/to/images"
        >>> mean, std = compute_mean_std(image_directory)
        >>> print(f"Channel-wise mean: {mean}")
        >>> print(f"Channel-wise std: {std}")
    """
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.npy')]
    mean = np.zeros(3)
    std = np.zeros(3)
    total_pixels = 0

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = np.load(image_path).astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Accumulate mean and std
        mean += image.mean(axis=(0, 1))
        std += image.std(axis=(0, 1))
        total_pixels += 1

    mean /= total_pixels
    std /= total_pixels

    return tuple(mean), tuple(std)

# Split SpaceNet data into train, validation and test sets
def train_val_split(root_dir, save_dir, train_percent=0.7, val_percent=0.15):
    # Create 'train' directory if it doesn't exist
    train_dir = os.path.join(root_dir, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Move 'images' and 'masks' folders into 'train' folder
    train_images_dir = os.path.join(train_dir, "images")
    train_masks_dir = os.path.join(train_dir, "masks")
    
    if not os.path.exists(train_images_dir):
        shutil.move(os.path.join(root_dir, "images"), train_images_dir)
    if not os.path.exists(train_masks_dir):
        shutil.move(os.path.join(root_dir, "masks"), train_masks_dir)

    # Create validation and test directories
    val_dir = os.path.join(save_dir, "val")
    test_dir = os.path.join(save_dir, "test")
    
    for dir_path in [val_dir, test_dir]:
        if not os.path.exists(os.path.join(dir_path, "images")):
            os.makedirs(os.path.join(dir_path, "images"))
        if not os.path.exists(os.path.join(dir_path, "masks")):
            os.makedirs(os.path.join(dir_path, "masks"))

    # List and shuffle all images
    all_images = list(os.listdir(train_images_dir))
    np.random.shuffle(all_images)

    # Calculate split indices
    num_train = int(len(all_images) * train_percent)
    num_val = int(len(all_images) * val_percent)
    
    # Split images into train, validation and test sets
    train_images = all_images[:num_train]
    val_images = all_images[num_train:num_train + num_val]
    test_images = all_images[num_train + num_val:]

    # Move validation images
    print("Moving validation set...")
    for img in tqdm(val_images):
        shutil.move(os.path.join(train_images_dir, img), 
                   os.path.join(val_dir, "images", img))
        shutil.move(os.path.join(train_masks_dir, img.replace(".npy", "_multi_channel_mask.npy")), 
                   os.path.join(val_dir, "masks", img.replace(".npy", "_multi_channel_mask.npy")))

    # Move test images
    print("Moving test set...")
    for img in tqdm(test_images):
        shutil.move(os.path.join(train_images_dir, img), 
                   os.path.join(test_dir, "images", img))
        shutil.move(os.path.join(train_masks_dir, img.replace(".npy", "_multi_channel_mask.npy")), 
                   os.path.join(test_dir, "masks", img.replace(".npy", "_multi_channel_mask.npy")))

def find_optimal_lr(model, train_loader, criterion, optimizer, device, 
                   min_lr=1e-7, max_lr=10, num_iter=500, log_dir=None):
    """
    Use PyTorch LR Finder to find optimal learning rate for training
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer instance
        device: Device to run on
        min_lr: Minimum learning rate to test
        max_lr: Maximum learning rate to test
        num_iter: Number of iterations for the test
        log_dir: Directory to save the LR finder plot
        
    Returns:
        float: Suggested learning rate
    """
    
    # Create a new dataloader with num_workers=0 for LR finder
    temp_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    # Run range test
    lr_finder.range_test(temp_loader, end_lr=max_lr, num_iter=num_iter, step_mode="exp")
    
    # Get learning rates and losses from history
    history = lr_finder.history
    lrs = history['lr']
    losses = history['loss']
    
    # Find the point of steepest descent
    gradients = np.gradient(losses)
    steepest_point = np.argmin(gradients)
    suggested_lr = lrs[steepest_point]
    
    # Plot and save
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_finder.plot(ax=ax)  # Plot on the specific axis
        
        # Customize plot
        ax.set_title('Learning Rate Finder Results')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
        ax.legend()
        
        # Save plot
        plot_path = os.path.join(log_dir, 'lr_finder_plot.png')
        plt.savefig(plot_path)
        plt.close(fig)  # Close the figure to free memory
    
    # Reset the model and optimizer to their initial state
    lr_finder.reset()
    
    return suggested_lr
