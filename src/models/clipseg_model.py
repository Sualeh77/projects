from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
# from cv2 import (
#     THRESH_BINARY,
#     THRESH_OTSU,
#     MORPH_CLOSE,
#     CC_STAT_AREA,
#     threshold,
#     morphologyEx,
#     connectedComponentsWithStats
# )

class CLIPSegPredictor:
    def __init__(self, device='cuda'):
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.to(device)
        self.model.eval()
        self.target_size = (512, 512)  # Match training size
        
        # Prompts tuned for satellite building detection
        self.prompts = [
            "square and rectangular shapes in satellite image",
            "geometric building structures viewed from above",
            "building rooftops in overhead view",
            "urban structures in aerial imagery",
            "buildings with flat roofs from satellite",
            "dense urban buildings from above",
            "residential and commercial buildings in satellite view"
        ]

    def predict(self, image):
        """
        Predict building segmentation mask for a given image
        Args:
            image: PIL Image or numpy array
        Returns:
            Binary mask as numpy array
        """
        with torch.no_grad():
            # If image is tensor, convert to numpy
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
                image = (image * 255).astype(np.uint8)
                image = np.transpose(image, (1, 2, 0))  # CHW to HWH
                image = Image.fromarray(image)
            
            # Preserve original size for later resizing
            original_size = image.size[::-1]  # (W,H) to (H,W)
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.BILINEAR)
            
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=["building"],
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            
            # Get prediction
            outputs = self.model(**inputs)
            pred = torch.sigmoid(outputs.logits)[0]
            
            # Apply Otsu's thresholding instead of fixed threshold
            pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
            _, mask = cv2.threshold(pred_np, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove small objects and fill holes
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 100:  # Minimum area threshold
                    mask[labels == i] = 0
            
            # Resize back to original size
            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = mask.resize(original_size[::-1], Image.Resampling.NEAREST)
            mask = np.array(mask) > 0
        
        return mask

    def __call__(self, image):
        return self.predict(image)

    def eval(self):
        """Add eval() method for compatibility"""
        self.model.eval()
        return self 