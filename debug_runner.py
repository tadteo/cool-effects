#!/usr/bin/env python3
"""
Debug runner for BlobTrace Art
Easy way to run debug mode with preconfigured settings
"""
import os
import sys
import argparse

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure debug mode
import config
config.DEBUG_MODE = True
config.DEBUG_SAVE_INTERMEDIATES = True
config.DEBUG_SHOW_VISUALIZATIONS = True
config.DEBUG_STEP_BY_STEP = False
config.DEBUG_MAX_FRAMES = 3  # Process only 3 frames by default
config.DEBUG_VERBOSE = True

from main import process_video
from config import INPUT_DIR, OUTPUT_DIR


def main():
    """
    Debug runner main function.
    """
    parser = argparse.ArgumentParser(description="BlobTrace Art - Debug Runner")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-p", "--prompt", default="stylized watercolor painting espressionism style with minimal color palette", 
                       help="Text prompt for AI generation")
    parser.add_argument("--frames", type=int, default=3, 
                       help="Number of frames to process (default: 3)")
    parser.add_argument("--step-by-step", action="store_true", 
                       help="Enable step-by-step mode (pauses between steps)")
    parser.add_argument("--no-viz", action="store_true", 
                       help="Disable visualizations (save files only)")
    parser.add_argument("--fast", action="store_true", 
                       help="Fast mode: process 1 frame only, no visualizations")
    
    args = parser.parse_args()
    
    # Configure debug settings based on arguments
    if args.step_by_step:
        config.DEBUG_STEP_BY_STEP = True
        config.DEBUG_SHOW_VISUALIZATIONS = True
        print("🔍 Step-by-step debug mode enabled")
        print("   Press Enter at each step to continue...")
    
    if args.no_viz:
        config.DEBUG_SHOW_VISUALIZATIONS = False
        print("💾 Visualization disabled - saving files only")
    
    if args.fast:
        config.DEBUG_MAX_FRAMES = 1
        config.DEBUG_SHOW_VISUALIZATIONS = False
        config.DEBUG_STEP_BY_STEP = False
        args.frames = 1
        print("⚡ Fast debug mode: 1 frame, no visualizations")
    
    # Set paths
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(INPUT_DIR, input_path)
    
    input_filename = os.path.basename(input_path)
    name, ext = os.path.splitext(input_filename)
    output_path = os.path.join(OUTPUT_DIR, f"{name}_debug{ext}")
    
    # Print debug configuration
    print("\n🎬 BlobTrace Art - Debug Mode")
    print("=" * 40)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Max frames: {args.frames}")
    print(f"Save intermediates: {config.DEBUG_SAVE_INTERMEDIATES}")
    print(f"Show visualizations: {config.DEBUG_SHOW_VISUALIZATIONS}")
    print(f"Step-by-step: {config.DEBUG_STEP_BY_STEP}")
    print(f"Debug directory: {config.DEBUG_DIR}")
    print("=" * 40)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        print("\n💡 Tip: Place your video file in the 'input' folder or provide full path")
        return
    
    # Start processing
    print("\n🚀 Starting debug processing...")
    
    try:
        success = process_video(
            input_path=input_path,
            output_path=output_path,
            prompt=args.prompt,
            max_frames=args.frames,
            enable_debug=True
        )
        
        if success:
            print("\n✅ Debug processing completed successfully!")
            print(f"📹 Output video: {output_path}")
            print(f"🔍 Debug results: {config.DEBUG_DIR}")
            
            # Open debug folder automatically
            import subprocess, platform
            try:
                if platform.system() == "Windows":
                    subprocess.run(["explorer", config.DEBUG_DIR])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", config.DEBUG_DIR])
                else:  # Linux
                    subprocess.run(["xdg-open", config.DEBUG_DIR])
                print("📂 Debug folder opened automatically")
            except Exception as e:
                print(f"⚠️ Could not open debug folder automatically: {e}")
                
        else:
            print("\n❌ Debug processing failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
    except Exception as e:
        print(f"\n💥 Error during processing: {e}")
        import traceback
        traceback.print_exc()


def list_example_commands():
    """Print example commands for different debug scenarios."""
    print("\n🎯 Example Debug Commands:")
    print("-" * 30)
    print("# Quick test (1 frame, no visualizations)")
    print("python debug_runner.py my_video.mp4 --fast")
    print()
    print("# Step-by-step analysis (3 frames)")
    print("python debug_runner.py my_video.mp4 --step-by-step")
    print()
    print("# Save intermediates only (no pop-up windows)")
    print("python debug_runner.py my_video.mp4 --no-viz --frames 5")
    print()
    print("# Custom prompt and frame count")
    print('python debug_runner.py my_video.mp4 -p "oil painting" --frames 10')
    print()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🔍 BlobTrace Art - Debug Runner")
        print("Usage: python debug_runner.py <video_file> [options]")
        list_example_commands()
    else:
        main() 
