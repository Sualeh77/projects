import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomSpacenetDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, combine_channels=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.combine_channels = combine_channels
        self.image_ids = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if idx >= len(self.image_ids):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self.image_ids)}")
        
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_id}.npy")
        mask_path = os.path.join(self.masks_dir, f"{image_id}_multi_channel_mask.npy")

        image = np.load(image_path)
        mask = np.load(mask_path)

        if self.combine_channels:
            mask = self.combine_channels_fn(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1)  # Convert to CxHxW
            mask = torch.from_numpy(mask).long()  # Convert to long tensor for CrossEntropyLoss

        return image, mask
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def combine_channels_fn(self, mask):
        combined_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        combined_mask[mask[:, :, 0] == 255] = 1 # Building Footprint
        combined_mask[mask[:, :, 1] == 255] = 2 # Boundary
        combined_mask[mask[:, :, 2] == 255] = 3 # Contact Points
        return combined_mask
