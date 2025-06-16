"""
Debug utilities for BlobTrace Art
Provides comprehensive debugging and visualization capabilities
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import DEBUG_DIR, DEBUG_SAVE_INTERMEDIATES, DEBUG_SHOW_VISUALIZATIONS, DEBUG_STEP_BY_STEP, DEBUG_VERBOSE


class DebugSession:
    """
    Debug session manager for tracking processing steps and saving intermediate results.
    """
    
    def __init__(self, session_name: str = None):
        if session_name is None:
            session_name = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_name = session_name
        self.session_dir = os.path.join(DEBUG_DIR, session_name)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.step_counter = 0
        self.frame_counter = 0
        self.log_entries = []
        
        self.log(f"Debug session started: {session_name}")
        
    def log(self, message: str, level: str = "INFO"):
        """Log a debug message."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.log_entries.append(log_entry)
        
        if DEBUG_VERBOSE:
            print(log_entry)
            
    def save_image(self, image: np.ndarray, filename: str, step: str = None) -> str:
        """Save an image to the debug directory."""
        if not DEBUG_SAVE_INTERMEDIATES:
            return ""
            
        if step:
            filepath = os.path.join(self.session_dir, f"step_{self.step_counter:03d}_{step}_{filename}")
        else:
            filepath = os.path.join(self.session_dir, filename)
            
        try:
            cv2.imwrite(filepath, image)
            self.log(f"Saved debug image: {filepath}")
            return filepath
        except Exception as e:
            self.log(f"Error saving image {filepath}: {e}", "ERROR")
            return ""
    
    def save_data(self, data: Any, filename: str, step: str = None) -> str:
        """Save debug data to JSON file."""
        if not DEBUG_SAVE_INTERMEDIATES:
            return ""
            
        if step:
            filepath = os.path.join(self.session_dir, f"step_{self.step_counter:03d}_{step}_{filename}")
        else:
            filepath = os.path.join(self.session_dir, filename)
            
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.log(f"Saved debug data: {filepath}")
            return filepath
        except Exception as e:
            self.log(f"Error saving data {filepath}: {e}", "ERROR")
            return ""
    
    def next_step(self, step_name: str):
        """Move to the next processing step."""
        self.step_counter += 1
        self.log(f"Step {self.step_counter}: {step_name}")
        
        if DEBUG_STEP_BY_STEP:
            input(f"Press Enter to continue from step {self.step_counter}: {step_name}...")
    
    def next_frame(self, frame_index: int):
        """Move to the next frame."""
        self.frame_counter = frame_index
        self.log(f"Processing frame {frame_index}")
    
    def finalize(self):
        """Finalize debug session."""
        # Save log to file
        log_file = os.path.join(self.session_dir, "debug_log.txt")
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.log_entries))
        
        self.log(f"Debug session finalized. Results saved to: {self.session_dir}")


def visualize_blob_detection(frame: np.ndarray, 
                           contours: List[np.ndarray],
                           expanded_contours: List[np.ndarray],
                           debug_session: DebugSession) -> np.ndarray:
    """
    Visualize blob detection results.
    
    Args:
        frame: Original frame
        contours: Original detected contours
        expanded_contours: Expanded contours
        debug_session: Debug session for logging
        
    Returns:
        Visualization image
    """
    debug_session.next_step("Blob Detection Visualization")
    
    # Create visualization
    vis_frame = frame.copy()
    
    # Draw original contours in green
    cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)
    
    # Draw expanded contours in red
    cv2.drawContours(vis_frame, expanded_contours, -1, (0, 0, 255), 2)
    
    # Add labels and statistics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_frame, f"Original Blobs: {len(contours)}", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_frame, f"Expanded Blobs: {len(expanded_contours)}", (10, 60), font, 0.7, (0, 0, 255), 2)
    
    # Calculate statistics
    stats = {
        "original_blob_count": len(contours),
        "expanded_blob_count": len(expanded_contours),
        "original_areas": [cv2.contourArea(c) for c in contours],
        "expanded_areas": [cv2.contourArea(c) for c in expanded_contours],
    }
    
    debug_session.save_data(stats, "blob_stats.json", "blob_detection")
    debug_session.save_image(vis_frame, "blob_visualization.jpg", "blob_detection")
    
    if DEBUG_SHOW_VISUALIZATIONS:
        show_debug_image(vis_frame, "Blob Detection Results")
    
    debug_session.log(f"Detected {len(contours)} blobs, expanded to {len(expanded_contours)}")
    
    return vis_frame


def visualize_masks(binary_mask: np.ndarray, 
                   soft_mask: np.ndarray,
                   debug_session: DebugSession) -> np.ndarray:
    """
    Visualize binary and soft masks.
    
    Args:
        binary_mask: Binary mask
        soft_mask: Soft (blurred) mask
        debug_session: Debug session for logging
        
    Returns:
        Visualization image
    """
    debug_session.next_step("Mask Visualization")
    
    # Create side-by-side visualization
    h, w = binary_mask.shape
    vis_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Binary mask (left side)
    vis_image[:, :w, 0] = binary_mask
    vis_image[:, :w, 1] = binary_mask
    vis_image[:, :w, 2] = binary_mask
    
    # Soft mask (right side)
    soft_mask_vis = (soft_mask * 255).astype(np.uint8)
    vis_image[:, w:, 0] = soft_mask_vis
    vis_image[:, w:, 1] = soft_mask_vis
    vis_image[:, w:, 2] = soft_mask_vis
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_image, "Binary Mask", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, "Soft Mask", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    debug_session.save_image(vis_image, "mask_comparison.jpg", "mask_generation")
    debug_session.save_image(binary_mask, "binary_mask.jpg", "mask_generation")
    debug_session.save_image((soft_mask * 255).astype(np.uint8), "soft_mask.jpg", "mask_generation")
    
    if DEBUG_SHOW_VISUALIZATIONS:
        show_debug_image(vis_image, "Mask Comparison")
    
    mask_coverage = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1])
    debug_session.log(f"Mask coverage: {mask_coverage:.2%}")
    
    return vis_image


def visualize_ai_generation(original: np.ndarray,
                          ai_generated: np.ndarray,
                          mask: np.ndarray,
                          prompt: str,
                          debug_session: DebugSession) -> np.ndarray:
    """
    Visualize AI generation results.
    
    Args:
        original: Original frame
        ai_generated: AI-generated content
        mask: Mask used for generation
        prompt: AI prompt used
        debug_session: Debug session for logging
        
    Returns:
        Visualization image
    """
    debug_session.next_step("AI Generation Visualization")
    
    # Create comparison visualization
    h, w = original.shape[:2]
    vis_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Original (top-left)
    vis_image[:h, :w] = original
    
    # AI Generated (top-right)
    vis_image[:h, w:] = ai_generated
    
    # Mask (bottom-left)
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    if len(mask.shape) == 2:
        mask_vis[:, :, 0] = mask
        mask_vis[:, :, 1] = mask
        mask_vis[:, :, 2] = mask
    else:
        mask_vis = mask
    vis_image[h:, :w] = mask_vis
    
    # Masked AI content (bottom-right)
    if len(mask.shape) == 2:
        mask_3d = mask[:, :, np.newaxis] / 255.0
    else:
        mask_3d = mask / 255.0
    masked_ai = (ai_generated * mask_3d).astype(np.uint8)
    vis_image[h:, w:] = masked_ai
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_image, "Original", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, "AI Generated", (w + 10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, "Mask", (10, h + 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, "Masked AI", (w + 10, h + 30), font, 0.7, (0, 255, 0), 2)
    
    debug_session.save_image(vis_image, "ai_generation_comparison.jpg", "ai_generation")
    debug_session.save_image(original, "original_frame.jpg", "ai_generation")
    debug_session.save_image(ai_generated, "ai_generated.jpg", "ai_generation")
    
    # Save generation metadata
    metadata = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "frame_shape": original.shape,
        "mask_coverage": float(np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]))
    }
    debug_session.save_data(metadata, "ai_generation_metadata.json", "ai_generation")
    
    if DEBUG_SHOW_VISUALIZATIONS:
        show_debug_image(vis_image, f"AI Generation: {prompt[:50]}...")
    
    debug_session.log(f"AI generation completed with prompt: {prompt}")
    
    return vis_image


def visualize_blending(original: np.ndarray,
                      ai_generated: np.ndarray,
                      mask: np.ndarray,
                      blended: np.ndarray,
                      debug_session: DebugSession) -> np.ndarray:
    """
    Visualize blending process.
    
    Args:
        original: Original frame
        ai_generated: AI-generated content
        mask: Blending mask
        blended: Final blended result
        debug_session: Debug session for logging
        
    Returns:
        Visualization image
    """
    debug_session.next_step("Blending Visualization")
    
    # Create 2x2 comparison
    h, w = original.shape[:2]
    vis_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Original (top-left)
    vis_image[:h, :w] = original
    
    # AI Generated (top-right)
    vis_image[:h, w:] = ai_generated
    
    # Mask (bottom-left)
    if len(mask.shape) == 2:
        mask_vis = np.stack([mask, mask, mask], axis=2)
    else:
        mask_vis = mask
    vis_image[h:, :w] = (mask_vis * 255).astype(np.uint8)
    
    # Blended result (bottom-right)
    vis_image[h:, w:] = blended
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_image, "Original", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, "AI Generated", (w + 10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, "Blend Mask", (10, h + 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, "Final Result", (w + 10, h + 30), font, 0.7, (0, 255, 0), 2)
    
    debug_session.save_image(vis_image, "blending_comparison.jpg", "blending")
    debug_session.save_image(blended, "final_result.jpg", "blending")
    
    if DEBUG_SHOW_VISUALIZATIONS:
        show_debug_image(vis_image, "Blending Results")
    
    debug_session.log("Blending completed successfully")
    
    return vis_image


def show_debug_image(image: np.ndarray, title: str = "Debug Image", wait_key: bool = True):
    """
    Show debug image in a window.
    
    Args:
        image: Image to display
        title: Window title
        wait_key: Whether to wait for key press
    """
    if not DEBUG_SHOW_VISUALIZATIONS:
        return
    
    # Check if we're running in a thread (GUI mode) - skip matplotlib on threads
    import threading
    if threading.current_thread() != threading.main_thread():
        print(f"Debug: {title} - Image saved (GUI mode - visualization disabled)")
        return
        
    try:
        # Convert BGR to RGB for display
        if len(image.shape) == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
            
        # Use matplotlib backend that works in console
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        plt.imshow(display_image, cmap='gray' if len(image.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        if wait_key and DEBUG_STEP_BY_STEP:
            input("Press Enter to continue...")
            
    except Exception as e:
        print(f"Debug: {title} - Visualization skipped ({e})")


def create_processing_summary(debug_session: DebugSession,
                            video_info: Dict[str, Any],
                            total_frames_processed: int,
                            processing_time: float):
    """
    Create a summary of the processing session.
    
    Args:
        debug_session: Debug session
        video_info: Video information
        total_frames_processed: Number of frames processed
        processing_time: Total processing time
    """
    summary = {
        "session_name": debug_session.session_name,
        "video_info": video_info,
        "frames_processed": total_frames_processed,
        "processing_time_seconds": processing_time,
        "fps": total_frames_processed / processing_time if processing_time > 0 else 0,
        "debug_settings": {
            "save_intermediates": DEBUG_SAVE_INTERMEDIATES,
            "show_visualizations": DEBUG_SHOW_VISUALIZATIONS,
            "step_by_step": DEBUG_STEP_BY_STEP,
            "verbose": DEBUG_VERBOSE
        },
        "log_entries_count": len(debug_session.log_entries)
    }
    
    debug_session.save_data(summary, "processing_summary.json")
    debug_session.log(f"Processing summary created: {total_frames_processed} frames in {processing_time:.2f}s")


def analyze_blob_characteristics(contours: List[np.ndarray]) -> Dict[str, Any]:
    """
    Analyze characteristics of detected blobs.
    
    Args:
        contours: List of blob contours
        
    Returns:
        Analysis results
    """
    if not contours:
        return {"blob_count": 0}
    
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    
    analysis = {
        "blob_count": len(contours),
        "total_area": sum(areas),
        "average_area": np.mean(areas),
        "min_area": min(areas),
        "max_area": max(areas),
        "area_std": np.std(areas),
        "average_perimeter": np.mean(perimeters),
        "circularity": []
    }
    
    # Calculate circularity for each blob
    for area, perimeter in zip(areas, perimeters):
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            analysis["circularity"].append(circularity)
    
    if analysis["circularity"]:
        analysis["average_circularity"] = np.mean(analysis["circularity"])
    
    return analysis 
