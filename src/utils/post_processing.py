import cv2
import numpy as np
import torch

def sharpen_building_masks(pred_mask, building_class_idx=1, min_area=100):
    """
    Sharpen building predictions to create precise polygonal shapes.
    Uses contour approximation with orientation preservation.
    
    Args:
        pred_mask: Predicted mask (H, W) with class indices
        building_class_idx: Index of building class
        min_area: Minimum area to consider for a building
    """
    # Convert to numpy if tensor
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    
    # Extract building mask
    building_mask = (pred_mask == building_class_idx).astype(np.uint8)
    
    # Create empty mask for refined buildings
    refined_mask = np.zeros_like(building_mask)
    
    # Find contours with hierarchy to detect holes
    contours, hierarchy = cv2.findContours(
        building_mask, 
        cv2.RETR_CCOMP,  # Retrieve both external and internal contours
        cv2.CHAIN_APPROX_TC89_KCOS  # More precise approximation
    )
    
    if len(contours) == 0:
        return pred_mask
    
    # Process each contour
    for i, contour in enumerate(contours):
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Get rotated rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get convex hull
        hull = cv2.convexHull(contour)
        
        # Calculate solidity (area ratio)
        solidity = area / cv2.contourArea(hull)
        
        if solidity > 0.9:
            # If shape is mostly convex, use rotated rectangle
            cv2.fillPoly(refined_mask, [box], 1)
        else:
            # For complex shapes, use polygon approximation
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ensure minimum points for a building
            if len(approx) >= 4:
                cv2.fillPoly(refined_mask, [approx], 1)
            else:
                cv2.fillPoly(refined_mask, [box], 1)
    
    # Replace building class in original prediction
    result_mask = pred_mask.copy()
    building_area = (pred_mask == building_class_idx)
    result_mask[building_area] = 0  # Clear original building area
    result_mask[refined_mask == 1] = building_class_idx  # Add refined buildings
    
    return result_mask 