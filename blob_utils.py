"""
Blob utilities for BlobTrace Art
Handles blob detection, expansion, and mask creation
"""
import cv2
import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional
from config import (
    BLOB_THRESHOLD, MIN_BLOB_AREA, MAX_BLOB_AREA, 
    BLOB_EXPANSION_FACTOR, NOISE_INTENSITY, 
    BLUR_KERNEL_SIZE, BLUR_SIGMA
)


def detect_blobs(frame: np.ndarray, threshold: int = BLOB_THRESHOLD) -> List[np.ndarray]:
    """
    Detect blob-like regions in a frame using contour detection.
    
    Args:
        frame: Input frame (BGR)
        threshold: Threshold for binary conversion
        
    Returns:
        List of contours representing blobs
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Create binary mask
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            valid_blobs.append(contour)
    
    return valid_blobs


def expand_blob_splat(contour: np.ndarray, 
                     expansion_factor: float = BLOB_EXPANSION_FACTOR,
                     noise_intensity: float = NOISE_INTENSITY) -> np.ndarray:
    """
    Expand and distort a blob contour to create a splat-like shape.
    
    Args:
        contour: Original contour points
        expansion_factor: How much to expand the blob
        noise_intensity: Intensity of Gaussian noise for distortion
        
    Returns:
        Expanded and distorted contour
    """
    if len(contour) == 0:
        return contour
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return contour
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])
    
    # Reshape contour for easier manipulation
    points = contour.reshape(-1, 2).astype(np.float32)
    
    # Expand points away from centroid
    vectors = points - centroid
    expanded_points = centroid + vectors * expansion_factor
    
    # Add Gaussian noise for organic distortion
    noise = np.random.normal(0, noise_intensity * np.linalg.norm(vectors, axis=1, keepdims=True), 
                           expanded_points.shape)
    distorted_points = expanded_points + noise
    
    # Ensure points are within reasonable bounds
    distorted_points = np.clip(distorted_points, 0, None)
    
    return distorted_points.astype(np.int32).reshape(-1, 1, 2)


def create_blob_mask(frame_shape: Tuple[int, int], contours: List[np.ndarray]) -> np.ndarray:
    """
    Create a binary mask from blob contours.
    
    Args:
        frame_shape: Shape of the frame (height, width)
        contours: List of contours
        
    Returns:
        Binary mask with blobs filled
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    
    return mask


def create_soft_mask(binary_mask: np.ndarray, 
                    blur_size: int = BLUR_KERNEL_SIZE,
                    blur_sigma: float = BLUR_SIGMA) -> np.ndarray:
    """
    Create a soft alpha mask by blurring the binary mask.
    
    Args:
        binary_mask: Binary mask
        blur_size: Size of Gaussian blur kernel
        blur_sigma: Standard deviation for Gaussian blur
        
    Returns:
        Soft mask with values between 0 and 1
    """
    # Apply Gaussian blur
    soft_mask = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), blur_sigma)
    
    # Normalize to 0-1 range
    soft_mask = soft_mask.astype(np.float32) / 255.0
    
    return soft_mask


def blend_with_mask(original: np.ndarray, 
                   generated: np.ndarray, 
                   mask: np.ndarray) -> np.ndarray:
    """
    Blend original frame with AI-generated content using a soft mask.
    
    Args:
        original: Original frame
        generated: AI-generated content
        mask: Soft mask (0-1 values)
        
    Returns:
        Blended frame
    """
    # Ensure mask has the right dimensions
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
    
    # Ensure all images have the same dimensions
    h, w = original.shape[:2]
    generated = cv2.resize(generated, (w, h))
    mask = cv2.resize(mask, (w, h))
    
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
    
    # Convert to float for blending
    original_f = original.astype(np.float32)
    generated_f = generated.astype(np.float32)
    
    # Blend using the mask
    blended = original_f * (1 - mask) + generated_f * mask
    
    # Convert back to uint8
    return np.clip(blended, 0, 255).astype(np.uint8)


def visualize_blobs(frame: np.ndarray, 
                   original_contours: List[np.ndarray],
                   expanded_contours: List[np.ndarray]) -> np.ndarray:
    """
    Visualize detected and expanded blobs for debugging.
    
    Args:
        frame: Original frame
        original_contours: Original blob contours
        expanded_contours: Expanded blob contours
        
    Returns:
        Frame with blob visualizations
    """
    vis_frame = frame.copy()
    
    # Draw original contours in green
    cv2.drawContours(vis_frame, original_contours, -1, (0, 255, 0), 2)
    
    # Draw expanded contours in red
    cv2.drawContours(vis_frame, expanded_contours, -1, (0, 0, 255), 2)
    
    return vis_frame


def process_frame_blobs(frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Complete blob processing pipeline for a single frame.
    
    Args:
        frame: Input frame
        
    Returns:
        Tuple of (binary_mask, expanded_contours)
    """
    # Detect blobs
    original_contours = detect_blobs(frame)
    
    # Expand blobs
    expanded_contours = []
    for contour in original_contours:
        expanded = expand_blob_splat(contour)
        expanded_contours.append(expanded)
    
    # Create mask from expanded contours
    binary_mask = create_blob_mask(frame.shape, expanded_contours)
    
    return binary_mask, expanded_contours


def get_blob_regions(frame: np.ndarray, mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract individual blob regions and their masks.
    
    Args:
        frame: Original frame
        mask: Combined mask of all blobs
        
    Returns:
        List of (region_image, region_mask) tuples
    """
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask)
    
    regions = []
    for label in range(1, num_labels):  # Skip background (label 0)
        # Create mask for this specific blob
        blob_mask = (labels == label).astype(np.uint8) * 255
        
        # Find bounding box
        coords = np.where(blob_mask > 0)
        if len(coords[0]) == 0:
            continue
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Extract region
        region_image = frame[y_min:y_max+1, x_min:x_max+1]
        region_mask = blob_mask[y_min:y_max+1, x_min:x_max+1]
        
        regions.append((region_image, region_mask))
    
    return regions 
