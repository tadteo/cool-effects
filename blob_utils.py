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
    BLUR_KERNEL_SIZE, BLUR_SIGMA,
    ENABLE_OUTWARD_NOISE, NOISE_STRENGTH, NOISE_RADIUS, NOISE_DECAY,
    ENABLE_MASK_SMOOTHING, SMOOTHING_INTENSITY, GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIGMA,
    MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_ITERATIONS, ENABLE_CONVEX_HULL_SMOOTHING, 
    CONVEX_HULL_EXPANSION, ENABLE_MASK_PADDING, MASK_PADDING_SIZE, PADDING_INTENSITY,
    ENABLE_ORGANIC_CURVES, CURVE_INTENSITY, CURVE_FREQUENCY, CURVE_AMPLITUDE, 
    CURVE_OCTAVES, ENABLE_SPLINE_SMOOTHING
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


def create_outward_noise_mask(binary_mask: np.ndarray, 
                             noise_strength: float = NOISE_STRENGTH,
                             noise_radius: int = NOISE_RADIUS,
                             noise_decay: float = NOISE_DECAY) -> np.ndarray:
    """
    Create outward-flowing noise from mask boundaries to make masks overflow.
    
    Args:
        binary_mask: Original binary mask
        noise_strength: Strength of the noise effect (0-1)
        noise_radius: How far the noise extends outward (pixels)
        noise_decay: How quickly noise decays with distance (0-1)
        
    Returns:
        Enhanced mask with outward noise
    """
    # Convert to float for processing
    mask_float = binary_mask.astype(np.float32) / 255.0
    
    # Create distance transform to find edges
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Create outward distance map (distance from mask edges)
    inverted_mask = 255 - binary_mask
    outward_distance = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
    
    # Create noise field
    h, w = binary_mask.shape
    noise_field = np.random.normal(0, 1, (h, w)).astype(np.float32)
    
    # Apply multiple octaves of noise for more organic look
    noise_octave1 = cv2.GaussianBlur(noise_field, (15, 15), 5)
    noise_field_small = cv2.resize(noise_field, (w//2, h//2))
    noise_octave2 = cv2.resize(cv2.GaussianBlur(noise_field_small, (7, 7), 2), (w, h))
    
    combined_noise = noise_octave1 * 0.7 + noise_octave2 * 0.3
    
    # Create distance-based decay for outward noise
    decay_mask = np.exp(-outward_distance / (noise_radius * noise_decay))
    decay_mask = np.clip(decay_mask, 0, 1)
    
    # Apply noise only outside the original mask, with distance decay
    outward_mask = np.where(binary_mask > 0, 0, 1)  # Only outside original mask
    noise_contribution = (combined_noise * noise_strength * decay_mask * outward_mask)
    
    # Combine original mask with outward noise
    enhanced_mask = mask_float + np.clip(noise_contribution, 0, 1)
    enhanced_mask = np.clip(enhanced_mask, 0, 1)
    
    return enhanced_mask


def create_soft_mask(binary_mask: np.ndarray, 
                    blur_size: int = BLUR_KERNEL_SIZE,
                    blur_sigma: float = BLUR_SIGMA,
                    enable_outward_noise: bool = ENABLE_OUTWARD_NOISE) -> np.ndarray:
    """
    Create a soft alpha mask by blurring the binary mask with optional outward noise.
    
    Args:
        binary_mask: Binary mask
        blur_size: Size of Gaussian blur kernel
        blur_sigma: Standard deviation for Gaussian blur
        enable_outward_noise: Whether to add outward-flowing noise
        
    Returns:
        Soft mask with values between 0 and 1
    """
    if enable_outward_noise:
        # First add outward noise to make mask overflow
        enhanced_mask = create_outward_noise_mask(binary_mask)
        
        # Then apply Gaussian blur for soft edges
        soft_mask = cv2.GaussianBlur((enhanced_mask * 255).astype(np.uint8), 
                                   (blur_size, blur_size), blur_sigma)
        soft_mask = soft_mask.astype(np.float32) / 255.0
    else:
        # Original behavior: just blur
        soft_mask = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), blur_sigma)
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


def create_flowing_mask_connections(binary_mask: np.ndarray, 
                                   connection_distance: int = 100,
                                   flow_strength: float = 0.6,
                                   wave_frequency: float = 0.05,
                                   smoothing_iterations: int = 3) -> np.ndarray:
    """
    Create flowing, wavy connections between nearby mask regions.
    
    Args:
        binary_mask: Original binary mask
        connection_distance: Maximum distance to connect masks
        flow_strength: Strength of the flowing effect (0-1)
        wave_frequency: Frequency of wavy patterns
        smoothing_iterations: Number of smoothing passes
        
    Returns:
        Enhanced mask with flowing connections
    """
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Create distance transform from mask regions
    distance_transform = cv2.distanceTransform(
        255 - binary_mask, cv2.DIST_L2, 5
    )
    
    # Find individual mask components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    
    if num_labels <= 2:  # Only background + 1 mask
        return binary_mask
    
    # Create flow field connections
    flowing_mask = binary_mask.copy()
    h, w = binary_mask.shape
    
    # Generate connection paths between nearby components
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):
            center1 = centroids[i]
            center2 = centroids[j]
            
            # Check if components are close enough to connect
            distance = np.linalg.norm(center1 - center2)
            if distance > connection_distance:
                continue
            
            # Create flowing connection between these components
            connection_mask = create_wavy_connection(
                center1, center2, distance_transform, 
                flow_strength, wave_frequency, (h, w)
            )
            
            # Add connection to flowing mask
            flowing_mask = cv2.bitwise_or(flowing_mask, connection_mask)
    
    # Apply smoothing to make connections more organic
    for _ in range(smoothing_iterations):
        flowing_mask = smooth_mask_organically(flowing_mask)
    
    return flowing_mask


def create_wavy_connection(point1: np.ndarray, point2: np.ndarray, 
                          distance_field: np.ndarray,
                          flow_strength: float, wave_frequency: float,
                          shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a wavy connection between two points using flow fields.
    
    Args:
        point1, point2: Connection endpoints
        distance_field: Distance transform for guidance
        flow_strength: Strength of wavy effect
        wave_frequency: Frequency of waves
        shape: Image shape (h, w)
        
    Returns:
        Binary mask with wavy connection
    """
    h, w = shape
    connection_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate connection vector
    direction = point2 - point1
    distance = np.linalg.norm(direction)
    
    if distance < 10:  # Too close, no connection needed
        return connection_mask
    
    # Normalize direction
    direction = direction / distance
    perpendicular = np.array([-direction[1], direction[0]])
    
    # Create points along the connection path
    num_points = int(distance / 5)  # One point every 5 pixels
    
    connection_points = []
    for i in range(num_points + 1):
        t = i / max(num_points, 1)
        
        # Base position along straight line
        base_pos = point1 + t * (point2 - point1)
        
        # Add wavy displacement
        wave_offset = np.sin(t * distance * wave_frequency) * flow_strength * 30
        wavy_pos = base_pos + perpendicular * wave_offset
        
        # Add some organic randomness
        organic_noise = np.random.normal(0, flow_strength * 5, 2)
        final_pos = wavy_pos + organic_noise
        
        # Clamp to image bounds
        final_pos[0] = np.clip(final_pos[0], 0, w - 1)
        final_pos[1] = np.clip(final_pos[1], 0, h - 1)
        
        connection_points.append(final_pos.astype(int))
    
    # Draw thick, organic connection
    connection_thickness = max(8, int(flow_strength * 20))
    
    for i in range(len(connection_points) - 1):
        pt1 = tuple(connection_points[i])
        pt2 = tuple(connection_points[i + 1])
        
        # Draw line with varying thickness for organic feel
        thickness_variation = int(connection_thickness * (0.7 + 0.6 * np.random.random()))
        cv2.line(connection_mask, pt1, pt2, 255, thickness_variation)
        
        # Add organic "bubbles" along the path
        if i % 3 == 0:  # Every 3rd point
            bubble_radius = int(connection_thickness * (0.8 + 0.4 * np.random.random()))
            cv2.circle(connection_mask, pt1, bubble_radius, 255, -1)
    
    return connection_mask


def smooth_mask_organically(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Apply organic smoothing to mask using custom kernels.
    
    Args:
        mask: Binary mask to smooth
        kernel_size: Size of smoothing kernel
        
    Returns:
        Smoothed mask
    """
    # Create organic-shaped kernel (not perfectly circular)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    
    # Create an irregular, organic shape
    center = kernel_size // 2
    for y in range(kernel_size):
        for x in range(kernel_size):
            # Distance from center with organic distortion
            dx = x - center
            dy = y - center
            
            # Add organic distortion using sine waves
            organic_factor = (
                1 + 0.3 * np.sin(dx * 0.5) * np.cos(dy * 0.5) +
                0.2 * np.sin(dx * 0.8 + dy * 0.3)
            )
            
            distance = np.sqrt(dx*dx + dy*dy) * organic_factor
            
            if distance <= center * 0.9:  # Slightly smaller than circle
                kernel[y, x] = 1
    
    # Apply morphological operations with organic kernel
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    
    # Additional Gaussian smoothing for final organic feel
    smoothed = cv2.GaussianBlur(smoothed, (7, 7), 2)
    
    # Threshold back to binary
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    return smoothed


def create_convex_hull_flowing_mask(binary_mask: np.ndarray,
                                   hull_expansion: float = 1.3,
                                   smoothing_strength: float = 0.4) -> np.ndarray:
    """
    Create flowing mask using convex hull approach with organic smoothing.
    
    Args:
        binary_mask: Original binary mask
        hull_expansion: How much to expand the convex hull
        smoothing_strength: Strength of organic smoothing
        
    Returns:
        Flowing mask with convex hull connections
    """
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary_mask
    
    # Group nearby contours
    contour_groups = group_nearby_contours(contours, max_distance=150)
    
    flowing_mask = np.zeros_like(binary_mask)
    
    for group in contour_groups:
        if len(group) == 1:
            # Single contour, just add it
            cv2.fillPoly(flowing_mask, [group[0]], 255)
        else:
            # Multiple contours, create flowing connection
            group_mask = create_group_flowing_hull(group, binary_mask.shape, 
                                                 hull_expansion, smoothing_strength)
            flowing_mask = cv2.bitwise_or(flowing_mask, group_mask)
    
    return flowing_mask


def group_nearby_contours(contours: List[np.ndarray], max_distance: int = 150) -> List[List[np.ndarray]]:
    """
    Group contours that are close to each other.
    
    Args:
        contours: List of contours
        max_distance: Maximum distance to group contours
        
    Returns:
        List of contour groups
    """
    if not contours:
        return []
    
    # Calculate centroids
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
        else:
            centroids.append((0, 0))
    
    # Group contours by distance
    groups = []
    used = set()
    
    for i, centroid1 in enumerate(centroids):
        if i in used:
            continue
            
        group = [contours[i]]
        used.add(i)
        
        # Find nearby contours
        for j, centroid2 in enumerate(centroids):
            if j <= i or j in used:
                continue
                
            distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
            if distance <= max_distance:
                group.append(contours[j])
                used.add(j)
        
        groups.append(group)
    
    return groups


def create_group_flowing_hull(contour_group: List[np.ndarray], 
                             shape: Tuple[int, int],
                             hull_expansion: float,
                             smoothing_strength: float) -> np.ndarray:
    """
    Create a flowing hull around a group of contours.
    
    Args:
        contour_group: List of contours to connect
        shape: Image shape
        hull_expansion: Hull expansion factor
        smoothing_strength: Smoothing strength
        
    Returns:
        Flowing hull mask
    """
    # Combine all points from all contours
    all_points = []
    for contour in contour_group:
        all_points.extend(contour.reshape(-1, 2))
    
    all_points = np.array(all_points)
    
    # Create convex hull
    hull = cv2.convexHull(all_points)
    
    # Expand hull
    hull_center = np.mean(hull.reshape(-1, 2), axis=0)
    expanded_hull = []
    
    for point in hull.reshape(-1, 2):
        direction = point - hull_center
        expanded_point = hull_center + direction * hull_expansion
        expanded_hull.append(expanded_point)
    
    expanded_hull = np.array(expanded_hull, dtype=np.int32)
    
    # Create mask from expanded hull
    hull_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(hull_mask, [expanded_hull], 255)
    
    # Apply organic smoothing
    for _ in range(int(smoothing_strength * 5)):
        hull_mask = smooth_mask_organically(hull_mask)
    
    return hull_mask


def smooth_mask_advanced(binary_mask: np.ndarray, 
                        smoothing_intensity: float = SMOOTHING_INTENSITY,
                        enable_convex_hull: bool = ENABLE_CONVEX_HULL_SMOOTHING) -> np.ndarray:
    """
    Apply advanced smoothing to binary mask to create ultra-smooth shapes.
    
    Args:
        binary_mask: Input binary mask
        smoothing_intensity: Intensity of smoothing (0-1)
        enable_convex_hull: Whether to use convex hull for maximum smoothness
        
    Returns:
        Smoothed binary mask
    """
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Step 1: Initial morphological smoothing
    result = morphological_smoothing(binary_mask, smoothing_intensity)
    
    # Step 2: Convex hull smoothing for ultra-smooth shapes (optional)
    if enable_convex_hull:
        result = convex_hull_smooth(result, smoothing_intensity)
    
    # Step 3: Final Gaussian smoothing for soft edges
    result = gaussian_smooth_mask(result, smoothing_intensity)
    
    return result


def morphological_smoothing(binary_mask: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    """
    Apply morphological operations to smooth mask shapes.
    
    Args:
        binary_mask: Input binary mask
        intensity: Smoothing intensity (0-1)
        
    Returns:
        Morphologically smoothed mask
    """
    # Safety check
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Calculate kernel size based on intensity (reduced maximum size)
    max_kernel_size = min(MORPHOLOGY_KERNEL_SIZE, 9)  # Cap at 9 to prevent over-smoothing
    kernel_size = max(3, int(max_kernel_size * intensity))
    if kernel_size % 2 == 0:  # Ensure odd kernel size
        kernel_size += 1
    
    if kernel_size < 3 or intensity <= 0:
        return binary_mask
    
    # Create elliptical kernel for more organic smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply morphological operations with reduced iterations
    iterations = max(1, int(MORPHOLOGY_ITERATIONS * intensity * 0.5))  # Reduce iterations by half
    
    # Store original for comparison
    original_area = np.sum(binary_mask > 0)
    
    # Close gaps and holes (less aggressive)
    result = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Only apply opening if we didn't lose too much area
    temp_result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
    temp_area = np.sum(temp_result > 0)
    
    if temp_area > original_area * 0.3:  # Only apply if we keep at least 30% of area
        result = temp_result
    
    # Additional light smoothing with smaller kernel
    small_kernel_size = max(3, kernel_size // 2)
    if small_kernel_size % 2 == 0:
        small_kernel_size += 1
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_kernel_size, small_kernel_size))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, small_kernel, iterations=1)
    
    return result


def convex_hull_smooth(binary_mask: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    """
    Apply convex hull smoothing for ultra-smooth blob shapes.
    
    Args:
        binary_mask: Input binary mask
        intensity: Smoothing intensity (0-1)
        
    Returns:
        Convex hull smoothed mask
    """
    # Safety check
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary_mask
    
    # Create result mask
    result = np.zeros_like(binary_mask)
    
    for contour in contours:
        # Calculate area to decide if we should apply convex hull
        area = cv2.contourArea(contour)
        if area < 100:  # Much lower threshold - was 500, now 100
            cv2.drawContours(result, [contour], -1, 255, -1)
            continue
        
        # Only apply convex hull if intensity is high enough
        if intensity < 0.3:
            # Low intensity: just copy original contour
            cv2.drawContours(result, [contour], -1, 255, -1)
            continue
        
        # Create convex hull
        hull = cv2.convexHull(contour)
        
        # Blend between original contour and convex hull based on intensity
        if intensity < 0.7:
            # Partial convex hull - blend with original
            hull_mask = np.zeros_like(binary_mask)
            cv2.fillPoly(hull_mask, [hull], 255)
            
            original_mask = np.zeros_like(binary_mask)
            cv2.drawContours(original_mask, [contour], -1, 255, -1)
            
            # Blend based on intensity (0.3-0.7 range maps to 0-1 blend factor)
            blend_factor = (intensity - 0.3) / 0.4
            blended = original_mask.astype(np.float32) * (1 - blend_factor) + hull_mask.astype(np.float32) * blend_factor
            result = cv2.bitwise_or(result, (blended > 127).astype(np.uint8) * 255)
        else:
            # High intensity: full convex hull with optional expansion
            hull_center = np.mean(hull.reshape(-1, 2), axis=0)
            expansion_factor = 1.0 + (CONVEX_HULL_EXPANSION - 1.0) * min(intensity, 1.0)
            
            expanded_hull = []
            for point in hull.reshape(-1, 2):
                direction = point - hull_center
                expanded_point = hull_center + direction * expansion_factor
                expanded_hull.append(expanded_point)
            
            hull = np.array(expanded_hull, dtype=np.int32)
            cv2.fillPoly(result, [hull], 255)
    
    return result


def gaussian_smooth_mask(binary_mask: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    """
    Apply Gaussian smoothing to mask edges.
    
    Args:
        binary_mask: Input binary mask
        intensity: Smoothing intensity (0-1)
        
    Returns:
        Gaussian smoothed mask (still binary)
    """
    if intensity <= 0:
        return binary_mask
    
    # Calculate blur parameters based on intensity
    blur_size = int(GAUSSIAN_BLUR_SIZE * intensity)
    if blur_size % 2 == 0:  # Ensure odd size
        blur_size += 1
    
    blur_sigma = GAUSSIAN_BLUR_SIGMA * intensity
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), blur_sigma)
    
    # Threshold back to binary (with slight bias toward keeping shape)
    threshold_value = int(127 * (1 - intensity * 0.2))  # Lower threshold = more inclusive
    _, result = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    return result


def create_ultra_smooth_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Create an ultra-smooth version of the mask using all smoothing techniques.
    
    Args:
        binary_mask: Input binary mask
        
    Returns:
        Ultra-smooth binary mask
    """
    if not ENABLE_MASK_SMOOTHING:
        return binary_mask
    
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Store original for fallback
    original_mask = binary_mask.copy()
    
    # Apply the full smoothing pipeline
    result = smooth_mask_advanced(
        binary_mask, 
        smoothing_intensity=SMOOTHING_INTENSITY,
        enable_convex_hull=ENABLE_CONVEX_HULL_SMOOTHING
    )
    
    # Safety check: if smoothing destroyed the mask, return original
    if np.sum(result) == 0:
        print("Warning: Smoothing removed all mask content, using original mask")
        return original_mask
    
    # Check if we lost too much area (more than 90% reduction is suspicious)
    original_area = np.sum(original_mask > 0)
    result_area = np.sum(result > 0)
    if result_area < original_area * 0.1:
        print(f"Warning: Smoothing reduced mask area too much ({result_area}/{original_area}), using original mask")
        return original_mask
    
    return result 


def pad_mask(binary_mask: np.ndarray, 
            padding_size: int = MASK_PADDING_SIZE,
            padding_intensity: float = PADDING_INTENSITY) -> np.ndarray:
    """
    Add padding to binary mask by expanding it outward.
    
    Args:
        binary_mask: Input binary mask
        padding_size: Size of padding in pixels
        padding_intensity: Intensity of padding (0-1)
        
    Returns:
        Padded binary mask
    """
    if not ENABLE_MASK_PADDING or padding_intensity <= 0:
        return binary_mask
    
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Calculate effective padding size based on intensity
    effective_padding = int(padding_size * padding_intensity)
    
    if effective_padding <= 0:
        return binary_mask
    
    # Create dilation kernel - use elliptical for more organic expansion
    kernel_size = effective_padding * 2 + 1  # Ensure odd size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply dilation to expand the mask
    padded_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    return padded_mask


def pad_mask_gradient(binary_mask: np.ndarray, 
                     padding_size: int = MASK_PADDING_SIZE,
                     padding_intensity: float = PADDING_INTENSITY) -> np.ndarray:
    """
    Add gradient padding to binary mask for smoother expansion.
    
    Args:
        binary_mask: Input binary mask
        padding_size: Size of padding in pixels
        padding_intensity: Intensity of padding (0-1)
        
    Returns:
        Padded binary mask with gradient edges
    """
    if not ENABLE_MASK_PADDING or padding_intensity <= 0:
        return binary_mask
    
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Calculate effective padding size
    effective_padding = int(padding_size * padding_intensity)
    
    if effective_padding <= 0:
        return binary_mask
    
    # Create distance transform from mask edges
    distance_transform = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 5)
    
    # Create gradient mask based on distance
    gradient_mask = np.zeros_like(binary_mask, dtype=np.float32)
    
    # Original mask areas = 1.0
    gradient_mask[binary_mask > 0] = 1.0
    
    # Create gradient in padding area
    padding_area = (distance_transform <= effective_padding) & (binary_mask == 0)
    gradient_values = 1.0 - (distance_transform[padding_area] / effective_padding)
    gradient_mask[padding_area] = gradient_values
    
    # Convert back to binary with threshold
    threshold = 0.3  # Adjust this to control how much of the gradient becomes solid
    padded_mask = (gradient_mask >= threshold).astype(np.uint8) * 255
    
    return padded_mask


def create_padded_mask(binary_mask: np.ndarray, 
                      padding_size: int = MASK_PADDING_SIZE,
                      padding_intensity: float = PADDING_INTENSITY,
                      use_gradient: bool = True) -> np.ndarray:
    """
    Create padded version of mask with optional gradient padding.
    
    Args:
        binary_mask: Input binary mask
        padding_size: Size of padding in pixels
        padding_intensity: Intensity of padding (0-1)
        use_gradient: Whether to use gradient padding (smoother) or simple dilation
        
    Returns:
        Padded binary mask
    """
    if not ENABLE_MASK_PADDING:
        return binary_mask
    
    if use_gradient:
        return pad_mask_gradient(binary_mask, padding_size, padding_intensity)
    else:
        return pad_mask(binary_mask, padding_size, padding_intensity)


def generate_perlin_noise_field(width, height, scale=0.005, octaves=4):
    """
    🌀 KISS Perlin noise generation for organic vertex displacement.
    
    Args:
        width, height: Noise field dimensions
        scale: Noise frequency (lower = smoother)
        octaves: Number of noise layers for complexity
        
    Returns:
        Tuple of (noise_x, noise_y) arrays for 2D displacement
    """
    def simple_noise(x, y, freq):
        """Simple Perlin-like noise using sine waves"""
        return (np.sin(x * freq) * np.cos(y * freq * 1.3) + 
                np.sin(x * freq * 1.7) * np.cos(y * freq * 0.8) +
                np.sin(x * freq * 0.6) * np.cos(y * freq * 2.1)) / 3.0
    
    # Create coordinate grids
    x = np.linspace(0, width * scale, width)
    y = np.linspace(0, height * scale, height)
    X, Y = np.meshgrid(x, y)
    
    # Accumulate octaves
    noise_x = np.zeros((height, width))
    noise_y = np.zeros((height, width))
    
    for octave in range(octaves):
        freq = 2 ** octave
        amp = 0.5 ** octave
        
        noise_x += simple_noise(X, Y, freq) * amp
        noise_y += simple_noise(X + 100, Y + 200, freq) * amp  # Offset for different pattern
    
    return noise_x, noise_y


def chaikin_corner_cutting(points, refinements=3):
    """
    🎨 Chaikin corner-cutting algorithm for ultra-smooth curves.
    
    Args:
        points: Array of [x, y] points forming a closed curve
        refinements: Number of refinement passes (2-4 recommended)
        
    Returns:
        Smoothed points array
    """
    points = np.array(points, dtype=np.float32)
    
    for _ in range(refinements):
        if len(points) < 3:
            break
            
        new_points = []
        n = len(points)
        
        for i in range(n):
            # Current point and next point (wrap around)
            p0 = points[i]
            p1 = points[(i + 1) % n]
            
            # Chaikin corner cutting: replace each edge with two points
            # at 1/4 and 3/4 along the edge
            q1 = 0.75 * p0 + 0.25 * p1  # 1/4 from p0 toward p1
            q2 = 0.25 * p0 + 0.75 * p1  # 3/4 from p0 toward p1
            
            new_points.extend([q1, q2])
        
        points = np.array(new_points)
    
    return points


def displace_contour_vertices(contour, noise_x, noise_y, amplitude):
    """
    🌀 Displace contour vertices using Perlin noise field.
    
    Args:
        contour: Contour points array
        noise_x, noise_y: Noise fields for displacement
        amplitude: Displacement strength
        
    Returns:
        Displaced contour points
    """
    displaced = []
    h, w = noise_x.shape
    
    for point in contour:
        x, y = int(point[0]), int(point[1])
        
        # Clamp coordinates to noise field bounds
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # Sample noise and displace
        dx = noise_x[y, x] * amplitude
        dy = noise_y[y, x] * amplitude
        
        displaced.append([point[0] + dx, point[1] + dy])
    
    return np.array(displaced)


def smooth_boundaries_with_splines(binary_mask: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    """
    Smooth mask boundaries using spline interpolation for ultra-smooth curves.
    
    Args:
        binary_mask: Input binary mask
        intensity: Smoothing intensity
        
    Returns:
        Mask with spline-smoothed boundaries
    """
    if intensity <= 0:
        return binary_mask
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary_mask
    
    result = np.zeros_like(binary_mask)
    
    for contour in contours:
        if len(contour) < 6:  # Need at least 6 points for spline
            cv2.drawContours(result, [contour], -1, 255, -1)
            continue
        
        # Extract points
        points = contour.reshape(-1, 2).astype(np.float32)
        
        # Create smooth spline curve
        smooth_contour = create_spline_curve(points, intensity)
        
        # Draw smooth contour
        cv2.fillPoly(result, [smooth_contour], 255)
    
    return result


def create_spline_curve(points: np.ndarray, smoothness: float = 0.8) -> np.ndarray:
    """
    Create a smooth spline curve from points.
    
    Args:
        points: Input points array
        smoothness: Curve smoothness factor
        
    Returns:
        Smoothed curve points
    """
    if len(points) < 4:
        return points.astype(np.int32)
    
    # Ensure closed curve
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Calculate distances for parameterization
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    distances = np.concatenate([[0], np.cumsum(distances)])
    
    # Create smooth parameter values
    total_length = distances[-1]
    num_points = max(len(points) * 2, 50)  # Increase resolution
    smooth_params = np.linspace(0, total_length, num_points)
    
    # Interpolate x and y coordinates separately using cubic interpolation
    try:
        from scipy.interpolate import interp1d
        
        # Cubic interpolation for smooth curves
        interp_x = interp1d(distances, points[:, 0], kind='cubic', assume_sorted=True)
        interp_y = interp1d(distances, points[:, 1], kind='cubic', assume_sorted=True)
        
        smooth_x = interp_x(smooth_params)
        smooth_y = interp_y(smooth_params)
        
        smooth_points = np.column_stack([smooth_x, smooth_y])
        
        # Apply additional smoothing based on intensity
        if smoothness > 0.5:
            smooth_points = smooth_curve_simple(smooth_points, smoothness)
        
        return smooth_points.astype(np.int32)
        
    except (ImportError, Exception):
        # Fallback to simple averaging if scipy is not available or interpolation fails
        return smooth_curve_simple(points, smoothness).astype(np.int32)





def smooth_curve_simple(points: np.ndarray, intensity: float) -> np.ndarray:
    """
    Simple curve smoothing using moving average.
    
    Args:
        points: Input curve points
        intensity: Smoothing intensity
        
    Returns:
        Smoothed curve points
    """
    if intensity <= 0 or len(points) < 3:
        return points
    
    # Window size based on intensity
    window_size = max(3, int(intensity * 7))
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed = points.copy()
    
    for i in range(len(points)):
        # Get neighborhood indices (circular)
        indices = [(i + j - half_window) % len(points) for j in range(window_size)]
        neighborhood = points[indices]
        
        # Weighted average (higher weight for center)
        weights = np.exp(-0.5 * ((np.arange(window_size) - half_window) / (window_size / 4))**2)
        weights /= np.sum(weights)
        
        smoothed[i] = np.average(neighborhood, axis=0, weights=weights)
    
    return smoothed


def create_kiss_organic_blob(binary_mask: np.ndarray, 
                            curve_intensity: float = CURVE_INTENSITY) -> np.ndarray:
    """
    🌀 KISS Edition: Pure generative art organic blob creation.
    
    Pipeline:
    1. Extract contour from mask
    2. Jitter vertices with Perlin noise → organic irregularity  
    3. Round corners with Chaikin → velvety contour
    4. Optional blur-threshold → painterly softness
    
    Args:
        binary_mask: Input binary mask
        curve_intensity: Overall organic intensity (0-1)
        
    Returns:
        Organic blob mask with flowing, artistic boundaries
    """
    if not ENABLE_ORGANIC_CURVES or curve_intensity <= 0:
        return binary_mask
    
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    h, w = binary_mask.shape
    
    # Step 1: Extract contour from mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary_mask
    
    result = np.zeros_like(binary_mask)
    
    for contour in contours:
        if len(contour) < 6:  # Skip tiny contours
            cv2.drawContours(result, [contour], -1, 255, -1)
            continue
        
        # Convert to simple points array
        points = contour.reshape(-1, 2).astype(np.float32)
        
        # Calculate amplitude based on shape size
        bbox = cv2.boundingRect(contour)
        diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
        amplitude = diag * 0.03 * curve_intensity  # 3% of diagonal
        
        # Step 2: Generate Perlin noise field for organic jitter
        noise_x, noise_y = generate_perlin_noise_field(w, h, scale=CURVE_FREQUENCY, octaves=CURVE_OCTAVES)
        
        # Step 3: Displace vertices with noise → organic irregularity
        jittered_points = displace_contour_vertices(points, noise_x, noise_y, amplitude)
        
        # Step 4: Chaikin corner cutting → velvety smooth contour
        refinements = max(2, int(curve_intensity * 4))  # 2-4 refinements based on intensity
        smooth_points = chaikin_corner_cutting(jittered_points, refinements)
        
        # Ensure points are within bounds
        smooth_points[:, 0] = np.clip(smooth_points[:, 0], 0, w - 1)
        smooth_points[:, 1] = np.clip(smooth_points[:, 1], 0, h - 1)
        
        # Step 5: Rasterize the organic curve back to mask
        cv2.fillPoly(result, [smooth_points.astype(np.int32)], 255)
    
    # Step 6: Optional painterly blur-threshold for extra softness
    if curve_intensity > 0.6 and ENABLE_SPLINE_SMOOTHING:
        # Light Gaussian blur for painterly effect
        blur_sigma = curve_intensity * 1.5
        kernel_size = int(blur_sigma * 4) | 1  # Ensure odd
        blurred = cv2.GaussianBlur(result, (kernel_size, kernel_size), blur_sigma)
        
        # Re-threshold to maintain binary mask
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Safety check: if organic transformation destroyed the mask, return original
    if np.sum(result) == 0:
        print("🌀 KISS: Organic transformation removed all content, using original")
        return binary_mask
    
    return result
