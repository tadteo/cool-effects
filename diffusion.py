"""
Diffusion model utilities for BlobTrace Art
Handles Stable Diffusion Inpainting integration
"""
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from typing import Optional, Union
import cv2
import warnings
from config import (
    MODEL_NAME, DEVICE, TORCH_DTYPE, DEFAULT_PROMPT, 
    NEGATIVE_PROMPT, GUIDANCE_SCALE, NUM_INFERENCE_STEPS
)

# Global pipeline instance for caching
_pipeline = None


def load_pipeline(model_name: str = MODEL_NAME, 
                 device: str = DEVICE,
                 torch_dtype=TORCH_DTYPE) -> StableDiffusionInpaintPipeline:
    """
    Load and cache the Stable Diffusion Inpainting pipeline.
    
    Args:
        model_name: Name of the model to load
        device: Device to run on ('cuda' or 'cpu')
        torch_dtype: Torch data type for optimization
        
    Returns:
        Loaded pipeline
    """
    global _pipeline
    
    if _pipeline is None:
        print(f"Loading diffusion model: {model_name}")
        print(f"Device: {device}")
        
        try:
            _pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            _pipeline = _pipeline.to(device)
            
            # Enable memory efficient attention if available
            if hasattr(_pipeline, 'enable_attention_slicing'):
                _pipeline.enable_attention_slicing()
            
            if hasattr(_pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    _pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    print("xformers not available, continuing without memory optimization")
            
            print("Pipeline loaded successfully!")
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise
    
    return _pipeline


def preprocess_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Preprocess image for diffusion model.
    
    Args:
        image: Input image (numpy array or PIL Image)
        
    Returns:
        PIL Image in RGB format
    """
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def preprocess_mask(mask: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Preprocess mask for diffusion model.
    
    Args:
        mask: Input mask (numpy array or PIL Image)
        
    Returns:
        PIL Image in 'L' (grayscale) mode
    """
    if isinstance(mask, np.ndarray):
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
    
    # Ensure grayscale mode
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    return mask


def extract_masked_patches(image: np.ndarray, mask: np.ndarray, patch_size: int = 512, padding: int = 64):
    """
    Extract patches from image where mask has content, suitable for diffusion processing.
    
    Args:
        image: Input image (H, W, 3)
        mask: Binary mask (H, W)
        patch_size: Target size for diffusion model (default 512)
        padding: Extra padding around mask regions
        
    Returns:
        List of dictionaries containing patch info: {
            'patch_image': PIL Image,
            'patch_mask': PIL Image, 
            'bbox': (x, y, w, h),
            'original_size': (w, h)
        }
    """
    patches = []
    
    # Find contours in mask to get bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return patches
    
    h, w = image.shape[:2]
    
    for contour in contours:
        # Get bounding box
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bbox_w + padding)
        y2 = min(h, y + bbox_h + padding)
        
        # Extract patch
        patch_image = image[y1:y2, x1:x2]
        patch_mask = mask[y1:y2, x1:x2]
        
        # Skip if patch is too small or has no mask content
        if patch_image.shape[0] < 32 or patch_image.shape[1] < 32 or np.sum(patch_mask) == 0:
            continue
        
        # Convert to PIL
        patch_image_pil = preprocess_image(patch_image)
        patch_mask_pil = preprocess_mask(patch_mask)
        
        # Resize to target size while maintaining aspect ratio
        original_size = patch_image_pil.size
        patch_image_pil.thumbnail((patch_size, patch_size), Image.LANCZOS)
        patch_mask_pil = patch_mask_pil.resize(patch_image_pil.size, Image.LANCZOS)
        
        patches.append({
            'patch_image': patch_image_pil,
            'patch_mask': patch_mask_pil,
            'bbox': (x1, y1, x2 - x1, y2 - y1),
            'original_size': original_size
        })
    
    return patches


def inpaint_patches(image: Union[np.ndarray, Image.Image],
                   mask: Union[np.ndarray, Image.Image],
                   prompt: str = DEFAULT_PROMPT,
                   negative_prompt: str = NEGATIVE_PROMPT,
                   guidance_scale: float = GUIDANCE_SCALE,
                   num_inference_steps: int = NUM_INFERENCE_STEPS,
                   strength: float = 1.0) -> Image.Image:
    """
    Inpaint using patch-based processing to handle large images while keeping original dimensions.
    
    Args:
        image: Input image
        mask: Mask indicating regions to inpaint
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of denoising steps
        strength: How much to transform the image (0-1)
        
    Returns:
        Inpainted PIL Image with original dimensions
    """
    # Convert inputs to numpy arrays for processing
    if isinstance(image, Image.Image):
        original_image = np.array(image)
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            # PIL uses RGB, convert to BGR for OpenCV operations
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        original_image = image.copy()
    
    if isinstance(mask, Image.Image):
        original_mask = np.array(mask.convert('L'))
    else:
        original_mask = mask.copy()
        if original_mask.dtype != np.uint8:
            original_mask = (original_mask * 255).astype(np.uint8)
    
    # Create result image starting with original
    result_image = original_image.copy()
    
    # Extract patches where mask has content
    patches = extract_masked_patches(original_image, original_mask)
    
    if not patches:
        # No patches to process, return original
        result_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        return result_pil
    
    print(f"Extracted {len(patches)} initial patches")
    
    # OPTIMIZATION 1: Filter out unimportant patches
    patches = filter_important_patches(patches)
    
    if not patches:
        print("No important patches found after filtering")
        result_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        return result_pil
    
    # OPTIMIZATION 2: Merge nearby patches to reduce diffusion calls
    patches = merge_nearby_patches(patches)
    
    # OPTIMIZATION 3: Recreate merged patches from original image
    optimized_patches = []
    for patch_info in patches:
        x, y, w, h = patch_info['bbox']
        
        # Extract the actual patch from the original image
        patch_image = original_image[y:y+h, x:x+w]
        patch_mask = original_mask[y:y+h, x:x+w]
        
        # Convert to PIL and resize for diffusion
        patch_image_pil = preprocess_image(patch_image)
        patch_mask_pil = preprocess_mask(patch_mask)
        
        # Resize to target size while maintaining aspect ratio
        original_size = patch_image_pil.size
        patch_image_pil.thumbnail((512, 512), Image.LANCZOS)
        patch_mask_pil = patch_mask_pil.resize(patch_image_pil.size, Image.LANCZOS)
        
        optimized_patches.append({
            'patch_image': patch_image_pil,
            'patch_mask': patch_mask_pil,
            'bbox': patch_info['bbox'],
            'original_size': original_size
        })
    
    print(f"Processing {len(optimized_patches)} optimized patches for inpainting...")
    
    # OPTIMIZATION 4: Batch process patches
    processed_patches = batch_process_patches(
        optimized_patches, prompt, negative_prompt, 
        guidance_scale, num_inference_steps, strength, 
        batch_size=4
    )
    
    # Apply processed patches back to result image
    for i, patch_info in enumerate(processed_patches):
        try:
            # Get the processed image from batch processing
            inpainted_patch = patch_info['processed_image']
            
            # Resize back to original patch size
            inpainted_patch = inpainted_patch.resize(patch_info['original_size'], Image.LANCZOS)
            
            # Convert to OpenCV format
            inpainted_patch_cv = np.array(inpainted_patch)
            inpainted_patch_cv = cv2.cvtColor(inpainted_patch_cv, cv2.COLOR_RGB2BGR)
            
            # Get patch coordinates
            x, y, w, h = patch_info['bbox']
            
            # Create a mask for blending this patch
            patch_mask = original_mask[y:y+h, x:x+w]
            patch_mask_float = patch_mask.astype(np.float32) / 255.0
            
            # Blend the inpainted patch back into the result
            if len(patch_mask_float.shape) == 2:
                patch_mask_float = patch_mask_float[:, :, np.newaxis]
            
            # Ensure dimensions match
            if inpainted_patch_cv.shape[:2] != (h, w):
                inpainted_patch_cv = cv2.resize(inpainted_patch_cv, (w, h))
            
            # Blend only where the mask indicates
            result_patch = result_image[y:y+h, x:x+w].astype(np.float32)
            inpainted_patch_float = inpainted_patch_cv.astype(np.float32)
            
            blended_patch = result_patch * (1 - patch_mask_float) + inpainted_patch_float * patch_mask_float
            result_image[y:y+h, x:x+w] = np.clip(blended_patch, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Error applying processed patch {i+1}: {e}")
            continue
    
    # Convert result back to PIL RGB format
    result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    return result_pil


# Update the main inpaint_region function to use patch-based processing for large images
def inpaint_region(image: Union[np.ndarray, Image.Image],
                  mask: Union[np.ndarray, Image.Image],
                  prompt: str = DEFAULT_PROMPT,
                  negative_prompt: str = NEGATIVE_PROMPT,
                  guidance_scale: float = GUIDANCE_SCALE,
                  num_inference_steps: int = NUM_INFERENCE_STEPS,
                  strength: float = 1.0) -> Image.Image:
    """
    Inpaint a region using Stable Diffusion with automatic patch-based processing for large images.
    
    Args:
        image: Input image
        mask: Mask indicating regions to inpaint
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of denoising steps
        strength: How much to transform the image (0-1)
        
    Returns:
        Inpainted PIL Image
    """
    # Preprocess inputs to get dimensions
    pil_image = preprocess_image(image)
    pil_mask = preprocess_mask(mask)
    
    # Check if image is large enough to require patch-based processing
    max_dimension = max(pil_image.size)
    
    if max_dimension > 768:  # Use patch-based processing for large images
        print(f"Large image detected ({pil_image.size}), using patch-based processing...")
        return inpaint_patches(image, mask, prompt, negative_prompt, guidance_scale, num_inference_steps, strength)
    else:
        # Use original method for smaller images
        # Load pipeline if not already loaded
        pipeline = load_pipeline()
        
        # Ensure consistent sizes
        if pil_image.size != pil_mask.size:
            pil_mask = pil_mask.resize(pil_image.size, Image.LANCZOS)
        
        try:
            with torch.no_grad():
                # Generate
                result = pipeline(
                    prompt=prompt,
                    image=pil_image,
                    mask_image=pil_mask,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=strength
                )
                
                return result.images[0]
                
        except Exception as e:
            print(f"Error during inpainting: {e}")
            # Return original image if inpainting fails
            return pil_image


def batch_inpaint(images: list, 
                 masks: list,
                 prompts: Union[str, list],
                 **kwargs) -> list:
    """
    Batch inpainting for multiple images.
    
    Args:
        images: List of input images
        masks: List of masks
        prompts: Single prompt or list of prompts
        **kwargs: Additional arguments for inpaint_region
        
    Returns:
        List of inpainted PIL Images
    """
    if len(images) != len(masks):
        raise ValueError("Number of images and masks must match")
    
    # Handle single prompt for all images
    if isinstance(prompts, str):
        prompts = [prompts] * len(images)
    elif len(prompts) != len(images):
        raise ValueError("Number of prompts must match number of images")
    
    results = []
    for image, mask, prompt in zip(images, masks, prompts):
        result = inpaint_region(image, mask, prompt, **kwargs)
        results.append(result)
    
    return results


def generate_artistic_prompt(base_prompt: str = DEFAULT_PROMPT,
                           style_modifiers: Optional[list] = None) -> str:
    """
    Generate artistic prompts with style variations.
    
    Args:
        base_prompt: Base prompt text
        style_modifiers: List of style modifiers to randomly choose from
        
    Returns:
        Enhanced prompt string
    """
    if style_modifiers is None:
        style_modifiers = [
            "dreamy watercolor",
            "oil painting texture",
            "impressionist brushstrokes",
            "abstract expressionist",
            "psychedelic colors",
            "pastel art",
            "ink wash painting",
            "digital art",
            "surreal artistic style"
        ]
    
    import random
    style = random.choice(style_modifiers)
    return f"{style} {base_prompt}"


def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image
        
    Returns:
        OpenCV image (BGR format)
    """
    # Convert to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    opencv_image = np.array(pil_image)
    
    # Convert RGB to BGR
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    
    return opencv_image


def optimize_pipeline_memory():
    """
    Optimize pipeline memory usage.
    """
    global _pipeline
    
    if _pipeline is not None:
        # Clear cache
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Enable sequential CPU offload if available
        if hasattr(_pipeline, 'enable_sequential_cpu_offload'):
            _pipeline.enable_sequential_cpu_offload()


def unload_pipeline():
    """
    Unload the pipeline to free memory.
    """
    global _pipeline
    
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


def get_pipeline_info() -> dict:
    """
    Get information about the loaded pipeline.
    
    Returns:
        Dictionary with pipeline information
    """
    global _pipeline
    
    info = {
        'loaded': _pipeline is not None,
        'model_name': MODEL_NAME,
        'device': DEVICE,
        'torch_dtype': str(TORCH_DTYPE)
    }
    
    if _pipeline is not None:
        info['memory_usage'] = 'Available' if DEVICE == 'cuda' else 'N/A'
    
    return info 


def merge_nearby_patches(patches, merge_distance=150, max_merged_size=1024):
    """
    Merge nearby patches to reduce the number of diffusion calls.
    
    Args:
        patches: List of patch dictionaries
        merge_distance: Maximum distance between patches to merge
        max_merged_size: Maximum size of merged patch
        
    Returns:
        List of merged patch dictionaries
    """
    if len(patches) <= 1:
        return patches
    
    merged_patches = []
    used_indices = set()
    
    for i, patch in enumerate(patches):
        if i in used_indices:
            continue
            
        # Start a new merged group with this patch
        group_patches = [patch]
        group_indices = [i]
        
        # Find nearby patches to merge
        x1, y1, w1, h1 = patch['bbox']
        center1 = (x1 + w1//2, y1 + h1//2)
        
        for j, other_patch in enumerate(patches):
            if j <= i or j in used_indices:
                continue
                
            x2, y2, w2, h2 = other_patch['bbox']
            center2 = (x2 + w2//2, y2 + h2//2)
            
            # Calculate distance between patch centers
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if distance <= merge_distance:
                # Check if merged patch would be too large
                min_x = min(x1, x2)
                min_y = min(y1, y2)
                max_x = max(x1 + w1, x2 + w2)
                max_y = max(y1 + h1, y2 + h2)
                
                merged_w = max_x - min_x
                merged_h = max_y - min_y
                
                if merged_w <= max_merged_size and merged_h <= max_merged_size:
                    group_patches.append(other_patch)
                    group_indices.append(j)
                    
                    # Update bounding box for the group
                    x1, y1 = min_x, min_y
                    w1, h1 = merged_w, merged_h
                    center1 = (x1 + w1//2, y1 + h1//2)
        
        # Mark all patches in this group as used
        used_indices.update(group_indices)
        
        if len(group_patches) == 1:
            # Single patch, keep as is
            merged_patches.append(patch)
        else:
            # Create merged patch
            merged_patch = create_merged_patch(group_patches)
            merged_patches.append(merged_patch)
    
    print(f"Merged {len(patches)} patches into {len(merged_patches)} groups")
    return merged_patches


def create_merged_patch(patch_group):
    """
    Create a single merged patch from a group of patches.
    
    Args:
        patch_group: List of patch dictionaries to merge
        
    Returns:
        Merged patch dictionary
    """
    # Calculate bounding box that encompasses all patches
    min_x = min(p['bbox'][0] for p in patch_group)
    min_y = min(p['bbox'][1] for p in patch_group)
    max_x = max(p['bbox'][0] + p['bbox'][2] for p in patch_group)
    max_y = max(p['bbox'][1] + p['bbox'][3] for p in patch_group)
    
    merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    
    # Use the largest original size as reference
    max_area = 0
    reference_patch = patch_group[0]
    for patch in patch_group:
        area = patch['original_size'][0] * patch['original_size'][1]
        if area > max_area:
            max_area = area
            reference_patch = patch
    
    return {
        'patch_image': reference_patch['patch_image'],  # Will be recreated from merged bbox
        'patch_mask': reference_patch['patch_mask'],    # Will be recreated from merged bbox
        'bbox': merged_bbox,
        'original_size': (merged_bbox[2], merged_bbox[3]),
        'merged_from': len(patch_group)  # Track how many patches were merged
    }


def batch_process_patches(patches, prompt, negative_prompt, guidance_scale, num_inference_steps, strength, batch_size=4):
    """
    Process multiple patches in batches for efficiency.
    
    Args:
        patches: List of patch dictionaries
        batch_size: Number of patches to process simultaneously
        
    Returns:
        List of processed patches
    """
    pipeline = load_pipeline()
    processed_patches = []
    
    # Process patches in batches
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]
        
        try:
            # Prepare batch inputs
            batch_images = []
            batch_masks = []
            
            # Pad all images to the same size for batching
            max_size = 512
            for patch_info in batch:
                # Resize to consistent size
                img = patch_info['patch_image'].resize((max_size, max_size), Image.LANCZOS)
                mask = patch_info['patch_mask'].resize((max_size, max_size), Image.LANCZOS)
                batch_images.append(img)
                batch_masks.append(mask)
            
            # Process batch
            with torch.no_grad():
                # Create batch prompts
                batch_prompts = [prompt] * len(batch)
                batch_negative_prompts = [negative_prompt] * len(batch)
                
                # Process all patches in the batch simultaneously
                results = pipeline(
                    prompt=batch_prompts,
                    image=batch_images,
                    mask_image=batch_masks,
                    negative_prompt=batch_negative_prompts,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=strength
                )
                
                # Store results
                for j, result_image in enumerate(results.images):
                    patch_info = batch[j].copy()
                    patch_info['processed_image'] = result_image
                    processed_patches.append(patch_info)
                    
            print(f"Processed batch {i//batch_size + 1}/{(len(patches) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Fallback to individual processing for this batch
            for patch_info in batch:
                try:
                    result = pipeline(
                        prompt=prompt,
                        image=patch_info['patch_image'],
                        mask_image=patch_info['patch_mask'],
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        strength=strength
                    )
                    patch_info['processed_image'] = result.images[0]
                    processed_patches.append(patch_info)
                except Exception as e2:
                    print(f"Error processing individual patch: {e2}")
                    continue
    
    return processed_patches


def filter_important_patches(patches, min_size=64, min_mask_coverage=0.05):
    """
    Filter out patches that are too small or have minimal mask coverage.
    
    Args:
        patches: List of patch dictionaries
        min_size: Minimum patch dimension
        min_mask_coverage: Minimum mask coverage ratio
        
    Returns:
        Filtered list of important patches
    """
    important_patches = []
    
    for patch in patches:
        w, h = patch['original_size']
        
        # Skip tiny patches
        if w < min_size or h < min_size:
            continue
            
        # Calculate mask coverage
        mask_array = np.array(patch['patch_mask'])
        total_pixels = mask_array.shape[0] * mask_array.shape[1]
        mask_pixels = np.sum(mask_array > 128)  # Count white pixels
        coverage = mask_pixels / total_pixels
        
        # Skip patches with minimal mask coverage
        if coverage < min_mask_coverage:
            continue
            
        important_patches.append(patch)
    
    print(f"Filtered {len(patches)} patches down to {len(important_patches)} important patches")
    return important_patches 
