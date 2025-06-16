# BlobTrace Art - Artistic Video Reinterpreter

🎨 **Transform your videos into artistic masterpieces using AI-powered blob detection and diffusion models!**

BlobTrace Art is a Python-based digital art tool that processes videos frame-by-frame by detecting blob-like regions, expanding them artistically, and reinterpreting them using Stable Diffusion Inpainting to create stunning artistic effects.

## ✨ Features

- **Intelligent Blob Detection**: Automatically detects interesting regions in video frames
- **Artistic Expansion**: Expands blobs with random distortion for organic, splat-like effects  
- **AI-Powered Reinterpretation**: Uses Stable Diffusion Inpainting to reimagine blob regions
- **Seamless Blending**: Smoothly blends AI-generated content with original footage
- **Interactive GUI**: Drag-and-drop interface with real-time progress tracking
- **Customizable Prompts**: Fine-tune the artistic style with custom text prompts
- **GPU Acceleration**: Optimized for CUDA when available
- **Batch Processing**: Process entire videos efficiently

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd blobtrace-art
   ```

2. **Create and activate conda environment:**
   ```bash
   # Create conda environment with Python 3.11
   conda create -n cool-effects python=3.11
   
   # Activate the environment
   conda activate cool-effects
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python gui.py
   ```

### Alternative Installation (without conda)

If you prefer not to use conda, you can install directly with pip (Python 3.11+ required):

```bash
# Make sure you have Python 3.11 or higher
python --version

# Install dependencies
pip install -r requirements.txt

# Run the application
python gui.py
```

### First Use

1. **Launch the application** and wait for the AI models to load
2. **Drag and drop** a video file (or click to browse)
3. **Customize the prompt** (e.g., "dreamy watercolor painting")
4. **Click "Process Video"** and watch the magic happen!

## 🎬 Usage

### GUI Mode (Recommended)

The graphical interface provides the easiest way to use BlobTrace Art:

- **Drag & Drop**: Simply drag your video file into the application
- **Live Preview**: See original and processed frame previews
- **Progress Tracking**: Real-time progress updates and logging
- **Settings**: Customize AI prompts and processing options

### Command Line Mode

For batch processing or automation:

```bash
# Basic usage
python main.py input_video.mp4

# With custom prompt
python main.py input_video.mp4 -p "impressionist oil painting"

# With artistic style variations
python main.py input_video.mp4 --artistic

# Limit frames for testing
python main.py input_video.mp4 --max-frames 50

# Custom output path
python main.py input_video.mp4 -o processed_video.mp4

# Enable debug mode
python main.py input_video.mp4 --debug

# Step-by-step debug mode (pauses between steps)
python main.py input_video.mp4 --debug-step-by-step
```

### 🔍 Debug Mode

BlobTrace Art includes a comprehensive debug mode to help you understand and troubleshoot the processing pipeline:

#### Quick Debug Testing

```bash
# Fast debug (1 frame, no visualizations)
python debug_runner.py your_video.mp4 --fast

# Step-by-step analysis (3 frames with visualizations)
python debug_runner.py your_video.mp4 --step-by-step

# Save debug files only (no pop-ups)
python debug_runner.py your_video.mp4 --no-viz --frames 5
```

#### Debug Features

- **📊 Step-by-Step Visualization**: See blob detection, AI generation, and blending results
- **💾 Intermediate File Saving**: All processing steps saved as images and data files
- **📝 Detailed Logging**: Comprehensive logs with timestamps and performance metrics
- **🔬 Blob Analysis**: Statistical analysis of detected blob characteristics
- **⏸️ Interactive Mode**: Pause between steps for detailed inspection
- **📁 Organized Output**: Debug results neatly organized in timestamped folders

#### What Debug Mode Shows You

1. **Original Frame**: Input frame before processing
2. **Blob Detection**: Original and expanded blob contours
3. **Mask Generation**: Binary and soft masks for blending
4. **AI Generation**: Original vs AI-generated content comparison
5. **Final Blending**: Step-by-step blending visualization
6. **Statistical Analysis**: Blob characteristics and processing metrics

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Settings**: Change AI model or device preferences
- **Blob Detection**: Adjust threshold and size parameters
- **Artistic Effects**: Modify expansion and noise parameters
- **Quality Settings**: Control inference steps and guidance

## 📁 Project Structure

```
blobtrace-art/
├── main.py           # Main orchestration script
├── gui.py            # GUI application
├── config.py         # Configuration settings
├── video_utils.py    # Video processing utilities
├── blob_utils.py     # Blob detection and manipulation
├── diffusion.py      # AI model integration
├── requirements.txt  # Python dependencies
├── input/           # Place input videos here
├── output/          # Processed videos saved here
├── assets/          # Additional resources
└── README.md        # This file
```

## 🎨 Artistic Styles

BlobTrace Art supports various artistic styles through text prompts:

- **"dreamy watercolor"** - Soft, flowing watercolor effects
- **"oil painting texture"** - Rich, textured oil painting style
- **"impressionist brushstrokes"** - Classic impressionist technique
- **"psychedelic colors"** - Vibrant, surreal color palettes
- **"ink wash painting"** - Traditional Asian ink painting style

Enable **"Use random artistic styles"** for automatic style variation!

## ⚙️ Requirements

### System Requirements
- **Python 3.11+** (recommended)
- **8GB+ RAM** (16GB+ recommended for large videos)
- **NVIDIA GPU** with CUDA for optimal performance (CPU fallback available)
- **4GB+ free disk space** (for AI models and debug files)

### Software Dependencies
- **OpenCV** for video processing
- **PyTorch** for AI model inference
- **Diffusers** for Stable Diffusion integration
- **Tkinter** for GUI (usually included with Python)
- **Conda** (recommended for environment management)

### Supported Platforms
- **Windows** 10/11
- **macOS** 10.15+ (including Apple Silicon M1/M2)
- **Linux** (Ubuntu 20.04+, other distributions)

## 🎯 Processing Pipeline

1. **Frame Extraction**: Extract individual frames from input video
2. **Blob Detection**: Identify interesting regions using contour detection
3. **Artistic Expansion**: Expand blobs with noise-based distortion
4. **AI Generation**: Use Stable Diffusion Inpainting to reimagine regions
5. **Seamless Blending**: Combine original and AI content with soft masks
6. **Video Reconstruction**: Compile processed frames into final video

## 🚨 Troubleshooting

### Installation Issues

**"conda: command not found"**
- Install Anaconda or Miniconda from https://docs.conda.io/en/latest/miniconda.html
- Or use the alternative pip installation method

**"Package installation failed"**
```bash
# Try updating pip first
pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v
```

**PyTorch installation issues**
- Visit https://pytorch.org/ for platform-specific installation commands
- For CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

**macOS Apple Silicon (M1/M2) issues**
```bash
# Use conda-forge for better ARM64 support
conda install -c conda-forge pytorch torchvision

# Install remaining packages with pip
pip install diffusers transformers accelerate opencv-python matplotlib tqdm scipy pillow
```

### Runtime Issues

**"CUDA out of memory"**
- Reduce video resolution or process fewer frames
- Enable CPU processing in config.py

**"Model loading failed"**
- Check internet connection for initial model download (first run only)
- Ensure sufficient disk space (~4GB for models)
- Clear cache: delete `~/.cache/huggingface/` and retry

**"No blobs detected"**
- Adjust BLOB_THRESHOLD in config.py
- Try different videos with more contrast
- Use debug mode to visualize detection: `python debug_runner.py video.mp4 --fast`

### Performance Tips

- **Use GPU**: Ensure CUDA is properly installed for GPU acceleration
- **Reduce Resolution**: Process smaller videos for faster results
- **Test First**: Use max-frames option to test on short clips
- **Close Other Apps**: Free up memory for better performance

## 🔮 Future Enhancements

- **Real-time Processing**: Live video stream processing
- **Multiple Blob Types**: Motion-based and color-based blob detection
- **Style Transfer**: Additional AI models for style variation
- **Batch Processing**: Process multiple videos simultaneously
- **Web Interface**: Browser-based version for easier access

## 📖 Technical Details

### Blob Detection Algorithm

1. Convert frame to grayscale
2. Apply Gaussian blur for noise reduction
3. Binary thresholding to create mask
4. Contour detection with area filtering
5. Centroid-based expansion with Gaussian noise

### AI Integration

- **Model**: RunwayML Stable Diffusion Inpainting
- **Optimization**: FP16 precision, attention slicing
- **Memory Management**: Sequential CPU offload when needed
- **Batching**: Efficient processing of multiple regions

### Blending Technique

- Gaussian blur on binary masks for soft edges
- Alpha blending with feathered transitions
- Color space preservation for natural results

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Bug reports and feature requests
- Code contributions and improvements
- Documentation updates
- Example videos and artistic styles
