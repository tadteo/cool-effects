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


def inpaint_region(image: Union[np.ndarray, Image.Image],
                  mask: Union[np.ndarray, Image.Image],
                  prompt: str = DEFAULT_PROMPT,
                  negative_prompt: str = NEGATIVE_PROMPT,
                  guidance_scale: float = GUIDANCE_SCALE,
                  num_inference_steps: int = NUM_INFERENCE_STEPS,
                  strength: float = 1.0) -> Image.Image:
    """
    Inpaint a region using Stable Diffusion.
    
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
    # Load pipeline if not already loaded
    pipeline = load_pipeline()
    
    # Preprocess inputs
    pil_image = preprocess_image(image)
    pil_mask = preprocess_mask(mask)
    
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
