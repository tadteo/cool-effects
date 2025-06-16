"""
Configuration file for BlobTrace Art
Contains all constants and paths for the application
"""
import os
import torch

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Debug settings
DEBUG_MODE = False  # Set to True to enable debug mode
DEBUG_SAVE_INTERMEDIATES = True  # Save intermediate results to files
DEBUG_SHOW_VISUALIZATIONS = True  # Show debug visualizations (auto-disabled in GUI mode)
DEBUG_STEP_BY_STEP = False  # Pause between steps for inspection
DEBUG_MAX_FRAMES = 5  # Limit frames in debug mode
DEBUG_VERBOSE = True  # Enable verbose logging

# Video settings
DEFAULT_OUTPUT_FILENAME = "processed.mp4"
DEFAULT_FPS = 30
VIDEO_CODEC = "mp4v"  # Fallback codec
# Better codec options for different platforms
import platform
if platform.system() == "Darwin":  # macOS
    VIDEO_CODEC = "avc1"  # H.264 codec, better compatibility on macOS
elif platform.system() == "Windows":
    VIDEO_CODEC = "XVID"  # XviD codec, good for Windows
else:  # Linux
    VIDEO_CODEC = "XVID"  # XviD codec, good for Linux

# Blob detection settings
BLOB_THRESHOLD = 127
MIN_BLOB_AREA = 100
MAX_BLOB_AREA = 50000
BLOB_EXPANSION_FACTOR = 1.5  # How much to expand blobs
NOISE_INTENSITY = 0.1  # Gaussian noise intensity for blob distortion

# Diffusion model settings
MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEFAULT_PROMPT = "dreamy watercolor painting"
NEGATIVE_PROMPT = "blurry, low quality, distorted"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 20

# Blending settings
BLUR_KERNEL_SIZE = 21  # For Gaussian blur on mask
BLUR_SIGMA = 5.0

# Outward noise settings for mask overflow
ENABLE_OUTWARD_NOISE = True  # Enable outward-flowing noise on masks
NOISE_STRENGTH = 0.3  # Strength of outward noise (0-1)
NOISE_RADIUS = 50  # How far noise extends outward (pixels)
NOISE_DECAY = 0.7  # How quickly noise decays with distance (0-1)

# GUI settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
PREVIEW_WIDTH = 400
PREVIEW_HEIGHT = 300 
