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
DEFAULT_PROMPT = "stylized watercolor painting espressionism style with minimal color palette"
NEGATIVE_PROMPT = "high quality try to refill the content"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 20

# Patch processing optimizations
ENABLE_PATCH_OPTIMIZATION = True  # Enable smart patch processing
PATCH_BATCH_SIZE = 4  # Number of patches to process simultaneously
PATCH_MERGE_DISTANCE = 150  # Distance threshold for merging nearby patches
PATCH_MIN_SIZE = 64  # Minimum patch size to process
PATCH_MIN_MASK_COVERAGE = 0.05  # Minimum mask coverage ratio (5%)

# Flowing mask connection settings
ENABLE_FLOWING_MASKS = True  # Enable flowing, wavy mask connections
FLOWING_MASK_STYLE = "flow_field"  # "flow_field", "convex_hull", or "disabled"
FLOW_CONNECTION_DISTANCE = 120  # Maximum distance to connect masks
FLOW_STRENGTH = 0.6  # Strength of flowing effect (0-1)
FLOW_WAVE_FREQUENCY = 0.05  # Frequency of wavy patterns
FLOW_SMOOTHING_ITERATIONS = 3  # Number of organic smoothing passes

# Blending settings
BLUR_KERNEL_SIZE = 21  # For Gaussian blur on mask
BLUR_SIGMA = 5.0

# Mask smoothing settings (NEW!)
ENABLE_MASK_SMOOTHING = True  # Enable advanced mask smoothing
SMOOTHING_INTENSITY = 0.5  # Overall smoothing intensity (0-1) - reduced from 0.8
GAUSSIAN_BLUR_SIZE = 11  # Gaussian blur kernel size for smoothness - reduced from 15
GAUSSIAN_BLUR_SIGMA = 3.0  # Gaussian blur strength - reduced from 4.0
MORPHOLOGY_KERNEL_SIZE = 7  # Morphological operations kernel size - reduced from 11
MORPHOLOGY_ITERATIONS = 1  # Number of morphological smoothing passes - reduced from 2
ENABLE_CONVEX_HULL_SMOOTHING = False  # Use convex hull for ultra-smooth shapes - disabled by default
CONVEX_HULL_EXPANSION = 1.05  # How much to expand convex hulls - reduced from 1.1

# Mask padding settings (NEW!)
ENABLE_MASK_PADDING = True  # Enable mask padding/expansion
MASK_PADDING_SIZE = 20  # Padding size in pixels
PADDING_INTENSITY = 0.7  # Padding intensity (0-1, affects how much to expand)

# Organic curve settings (NEW!)
ENABLE_ORGANIC_CURVES = True  # Enable organic, sinuous curve generation
CURVE_INTENSITY = 0.8  # How much to distort shapes organically (0-1)
CURVE_FREQUENCY = 0.15  # Frequency of organic waves/curves
CURVE_AMPLITUDE = 15  # Amplitude of curve distortions in pixels
CURVE_OCTAVES = 3  # Number of noise octaves for organic complexity
ENABLE_SPLINE_SMOOTHING = True  # Use spline interpolation for ultra-smooth curves

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
