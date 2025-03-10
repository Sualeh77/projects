import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from collections import defaultdict
import os
import torchvision.utils as vutils
from src.config.base_config import DATASET, METRICS, TRAINING
from src.utils.post_processing import sharpen_building_masks

class MetricFunction:
    """Wrapper class for smp.metrics.functional metrics"""
    def __init__(self, metric_fn, **kwargs):
        self.metric_fn = metric_fn
        self.kwargs = kwargs
    
    def __call__(self, pred, target):
        # Convert predictions to class indices for multiclass case
        if pred.shape[1] > 1:  # multiclass case (N, C, H, W)
            pred = torch.argmax(pred, dim=1)  # Convert to (N, H, W) with class indices
        
        # Get stats first
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred,
            target,
            mode='multiclass',
            num_classes=self.kwargs.get('num_classes'),
            ignore_index=None if not self.kwargs.get('ignore_channels') else 0
        )
        
        # Remove incompatible kwargs
        metric_kwargs = self.kwargs.copy()
        metric_kwargs.pop('mode', None)
        metric_kwargs.pop('ignore_channels', None)
        metric_kwargs.pop('num_classes', None)  # Remove as it's not needed for metric computation
        
        # Calculate metric using stats
        return self.metric_fn(tp, fp, fn, tn, **metric_kwargs)

class SegmentationMetrics:
    def __init__(self, num_classes=DATASET['NUM_CLASSES'], ignore_background=TRAINING['IGNORE_BACKGROUND']):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        
        # Initialize base metrics
        self.metrics = {}
        for metric_name in METRICS['METRICS_LIST']:
            self.metrics[f'{metric_name}_all'] = self._get_metric_fn(metric_name)
            self.metrics[f'{metric_name}_building'] = self._get_metric_fn(metric_name)
    
    def _get_metric_fn(self, metric_name):
        """Helper function to get metric function from smp"""
        kwargs = {
            'mode': 'multiclass',
            'ignore_channels': [0] if self.ignore_background else None,
            'num_classes': self.num_classes,
            'reduction': 'micro'  # Use micro averaging by default
        }
        
        if metric_name == 'IoU':
            return MetricFunction(smp.metrics.functional.iou_score, **kwargs)
        elif metric_name == 'F1_Score':
            return MetricFunction(smp.metrics.functional.f1_score, **kwargs)
        elif metric_name == 'Accuracy':
            return MetricFunction(smp.metrics.functional.accuracy, **kwargs)
        elif metric_name == 'Precision':
            return MetricFunction(smp.metrics.functional.precision, **kwargs)
        elif metric_name == 'Recall':
            return MetricFunction(smp.metrics.functional.recall, **kwargs)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def update(self, pred, target):
        """Update metrics for a batch"""
        # Convert predictions to probabilities
        pred = torch.softmax(pred, dim=1)  # Keep this for class-specific metrics
        
        metrics_dict = {}
        
        # Calculate general metrics (all classes)
        for metric_name in METRICS['METRICS_LIST']:
            metric_fn = self.metrics[f'{metric_name}_all']
            metrics_dict[f'{metric_name}_all'] = metric_fn(pred, target)
            
            # Calculate per-class metrics
            if METRICS['COMPUTE_CLASS_WISE']:
                for class_idx in range(self.num_classes):
                    if self.ignore_background and class_idx == 0:
                        continue
                    class_name = DATASET['CLASS_LABELS'][class_idx]
                    # For per-class metrics, create binary predictions
                    # Keep the same shape as target: (N, H, W)
                    class_pred = (torch.argmax(pred, dim=1) == class_idx).long()  # Shape: (N, H, W)
                    class_target = (target == class_idx).long()  # Shape: (N, H, W)
                    # Add channel dimension for get_stats
                    class_pred = class_pred.unsqueeze(1)  # Shape: (N, 1, H, W)
                    class_target = class_target.unsqueeze(1)  # Shape: (N, 1, H, W)
                    metrics_dict[f'{metric_name}_{class_name}'] = metric_fn(class_pred, class_target)
        
        # Compute building-specific metrics
        if METRICS['COMPUTE_BUILDING_ONLY']:
            building_idx = METRICS['BUILDING_CLASS_INDEX']
            # Create binary predictions for buildings
            building_pred = (torch.argmax(pred, dim=1) == building_idx).long()  # Shape: (N, H, W)
            building_target = (target == building_idx).long()  # Shape: (N, H, W)
            # Add channel dimension
            building_pred = building_pred.unsqueeze(1)  # Shape: (N, 1, H, W)
            building_target = building_target.unsqueeze(1)  # Shape: (N, 1, H, W)
            
            for metric_name in METRICS['METRICS_LIST']:
                metric_fn = self.metrics[f'{metric_name}_building']
                metrics_dict[f'{metric_name}_building_only'] = metric_fn(building_pred, building_target)
        
        return metrics_dict

    @staticmethod
    def update_clipseg(pred_mask, gt_mask):
        """
        Compute metrics for CLIPSeg predictions
        Args:
            pred_mask: Binary prediction mask
            gt_mask: Ground truth mask with all classes
        """
        # Extract building class from ground truth
        gt_buildings = (gt_mask == DATASET['CLASS_LABELS'].index('Building')).cpu().numpy()
        
        # Compute metrics
        tp = np.logical_and(pred_mask, gt_buildings).sum()
        fp = np.logical_and(pred_mask, np.logical_not(gt_buildings)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), gt_buildings).sum()
        
        # Calculate metrics
        metrics = {}
        metrics['IoU'] = tp / (tp + fp + fn + 1e-6)
        metrics['F1_Score'] = 2 * tp / (2 * tp + fp + fn + 1e-6)
        metrics['Precision'] = tp / (tp + fp + 1e-6)
        metrics['Recall'] = tp / (tp + fn + 1e-6)
        
        return metrics

def compute_mean_metrics(metrics_list):
    """Compute mean metrics across batches"""
    mean_metrics = defaultdict(float)
    num_batches = len(metrics_list)
    
    for metrics in metrics_list:
        for key, value in metrics.items():
            # Convert tensor to float if needed
            if torch.is_tensor(value):
                value = value.item()
            mean_metrics[key] += value
    
    # Convert all values to native Python types
    return {key: float(value / num_batches) for key, value in mean_metrics.items()}

def save_prediction(pred_mask, save_path):
    """Save prediction mask with different colors for each class"""
    # Create color map for visualization
    colors = torch.tensor([
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Building - red
        [0, 255, 0],      # Boundary - green
        [0, 0, 255]       # Contact - blue
    ]).float()
    
    # Convert prediction to color image
    pred_mask = pred_mask.cpu()
    colored_mask = colors[pred_mask.long()]
    
    # Save image
    vutils.save_image(colored_mask.permute(2, 0, 1) / 255.0, save_path)

def run_inference(model, data_loader, device, save_predictions=False, post_process=False, model_pred_dir=None, post_pred_dir=None):
    """Run inference and compute metrics"""
    # Set model to eval mode if it has the method
    if hasattr(model, 'eval'):
        model.eval()
    
    metrics = SegmentationMetrics()
    all_metrics = []
    
    if save_predictions and model_pred_dir:
        os.makedirs(model_pred_dir, exist_ok=True)
    if post_process and post_pred_dir:
        os.makedirs(post_pred_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(data_loader, desc='Inference')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Handle CLIPSeg differently
            if hasattr(model, 'prompts'):  # Check if it's CLIPSeg model
                batch_metrics = []
                for i in range(len(images)):
                    # Get prediction for single image
                    pred_mask = model.predict(
                        images[i],  # Pass tensor directly, predict method will handle conversion
                        threshold=None  # Using Otsu's thresholding instead
                    )
                    
                    if save_predictions:
                        save_path = os.path.join(model_pred_dir, f'pred_{batch_idx}_{i}.png')
                        save_prediction(torch.from_numpy(pred_mask), save_path)
                    
                    if post_process:
                        pred_mask = sharpen_building_masks(pred_mask, building_class_idx=1)
                        if save_predictions:
                            post_save_path = os.path.join(post_pred_dir, f'pred_{batch_idx}_{i}.png')
                            save_prediction(torch.from_numpy(pred_mask), post_save_path)
                    
                    # Compute metrics for single image
                    img_metrics = metrics.update_clipseg(pred_mask, masks[i])
                    batch_metrics.append(img_metrics)
                
                # Average batch metrics
                avg_metrics = {k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0]}
                all_metrics.append(avg_metrics)
                continue
            
            # Forward pass for other models
            outputs = model(images)
            
            # Compute metrics for batch
            batch_metrics = metrics.update(outputs, masks)
            
            # Convert tensor values to Python native types
            batch_metrics = {k: v.item() if torch.is_tensor(v) else float(v) 
                           for k, v in batch_metrics.items()}
            all_metrics.append(batch_metrics)
            
            # Save predictions if required
            if save_predictions:
                pred_masks = torch.argmax(outputs, dim=1)
                for i in range(len(images)):
                    save_path = os.path.join(model_pred_dir, f'pred_{batch_idx}_{i}.png')
                    save_prediction(pred_masks[i], save_path)
                    if post_process:
                        post_processed_mask = torch.from_numpy(
                            sharpen_building_masks(
                                pred_masks[i],
                                building_class_idx=DATASET['CLASS_LABELS'].index('Building')
                            )
                        ).to(pred_masks.device)
                        post_save_path = os.path.join(post_pred_dir, f'pred_{batch_idx}_{i}.png')
                        save_prediction(post_processed_mask, post_save_path)
    
    # Compute mean metrics and ensure they are JSON serializable
    mean_metrics = compute_mean_metrics(all_metrics)
    
    # Convert any remaining tensor values to Python native types
    return {k: float(v) if torch.is_tensor(v) else v 
            for k, v in mean_metrics.items()}

def create_visualization(image, true_mask, pred_mask):
    """Create a visualization grid of input, true mask, and predicted mask"""
    # Create color map for masks
    colors = torch.tensor([
        [0, 0, 0],        # Background - black
        [255, 0, 0],      # Building - red
        [0, 255, 0],      # Boundary - green
        [0, 0, 255]       # Contact - blue
    ]).float().to(true_mask.device)
    
    # Convert masks to colored images
    true_mask = true_mask.cpu()
    pred_mask = pred_mask.cpu()
    true_colored = colors[true_mask.long()]
    pred_colored = colors[pred_mask.long()]
    
    # Normalize image for visualization
    image = image.cpu()
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # Create visualization grid
    viz = torch.cat([
        image_norm,
        true_colored.permute(2, 0, 1) / 255.0,
        pred_colored.permute(2, 0, 1) / 255.0
    ], dim=-1)
    
    return viz

def compute_single_iou(pred_mask, true_mask, num_classes=4, ignore_background=True):
    """Compute IoU for a single image"""
    metrics = SegmentationMetrics(num_classes, ignore_background)
    pred_mask = torch.softmax(pred_mask.unsqueeze(0), dim=1)
    true_mask = true_mask.unsqueeze(0)
    
    return metrics.update(pred_mask, true_mask) 