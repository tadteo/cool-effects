"""
Main orchestration module for BlobTrace Art
Coordinates the entire video processing pipeline
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable
import argparse
import time

from config import (
    INPUT_DIR, OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME, DEFAULT_PROMPT,
    DEBUG_MODE, DEBUG_MAX_FRAMES, ENABLE_FLOWING_MASKS, FLOWING_MASK_STYLE,
    FLOW_CONNECTION_DISTANCE, FLOW_STRENGTH, FLOW_WAVE_FREQUENCY, FLOW_SMOOTHING_ITERATIONS
)
from video_utils import extract_frames, get_video_info, VideoWriter
from blob_utils import (
    process_frame_blobs, create_soft_mask, blend_with_mask,
    detect_blobs, expand_blob_splat, create_blob_mask,
    create_flowing_mask_connections, create_convex_hull_flowing_mask,
    create_ultra_smooth_mask, create_padded_mask, create_kiss_organic_blob
)
from diffusion import inpaint_region, pil_to_opencv, generate_artistic_prompt
from debug_utils import (
    DebugSession, visualize_blob_detection, visualize_masks,
    visualize_ai_generation, visualize_blending, create_processing_summary,
    analyze_blob_characteristics
)


def process_single_frame(frame: np.ndarray, 
                        prompt: str = DEFAULT_PROMPT,
                        progress_callback: Optional[Callable] = None,
                        debug_session: Optional[DebugSession] = None,
                        frame_index: int = 0,
                        save_frames: bool = False,
                        enable_outward_noise: bool = True) -> np.ndarray:
    """
    Process a single frame through the complete pipeline.
    
    Args:
        frame: Input frame
        prompt: Text prompt for AI generation
        progress_callback: Optional callback for progress updates
        debug_session: Optional debug session for detailed logging
        frame_index: Frame index for debugging
        
    Returns:
        Processed frame
    """
    if debug_session:
        debug_session.next_frame(frame_index)
        debug_session.save_image(frame, f"frame_{frame_index:04d}_original.jpg")
    
    if progress_callback:
        progress_callback("Detecting blobs...")
    
    # Step 1: Detect blobs
    if debug_session:
        debug_session.next_step("Blob Detection")
    
    original_contours = detect_blobs(frame)
    
    if not original_contours:
        if debug_session:
            debug_session.log("No blobs detected in frame")
        return frame
    
    # Step 2: Expand blobs
    if debug_session:
        debug_session.next_step("Blob Expansion")
    
    expanded_contours = []
    for contour in original_contours:
        expanded = expand_blob_splat(contour)
        expanded_contours.append(expanded)
    
    # Create masks
    binary_mask = create_blob_mask(frame.shape, expanded_contours)
    
    # Apply flowing mask connections if enabled
    if ENABLE_FLOWING_MASKS and FLOWING_MASK_STYLE != "disabled":
        if debug_session:
            debug_session.log(f"Applying flowing mask style: {FLOWING_MASK_STYLE}")
            # Save original mask for comparison
            debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_original_mask.jpg")
        
        if FLOWING_MASK_STYLE == "flow_field":
            binary_mask = create_flowing_mask_connections(
                binary_mask,
                connection_distance=FLOW_CONNECTION_DISTANCE,
                flow_strength=FLOW_STRENGTH,
                wave_frequency=FLOW_WAVE_FREQUENCY,
                smoothing_iterations=FLOW_SMOOTHING_ITERATIONS
            )
        elif FLOWING_MASK_STYLE == "convex_hull":
            binary_mask = create_convex_hull_flowing_mask(
                binary_mask,
                hull_expansion=1.2 + FLOW_STRENGTH * 0.3,
                smoothing_strength=FLOW_STRENGTH
            )
        
        if debug_session:
            # Save flowing mask for comparison
            debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_flowing_mask.jpg")
    
    # Apply ultra-smooth mask processing
    if debug_session:
        debug_session.log("Applying ultra-smooth mask processing...")
        # Save pre-smoothing mask for comparison
        debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_before_smoothing.jpg")
    
    # Create ultra-smooth version of the mask
    binary_mask = create_ultra_smooth_mask(binary_mask)
    
    if debug_session:
        # Save ultra-smooth mask for comparison
        debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_ultra_smooth_mask.jpg")
    
    # Apply padding to expand the mask
    if debug_session:
        debug_session.log("Applying mask padding...")
        # Save pre-padding mask for comparison
        debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_before_padding.jpg")
    
    # Create padded version of the mask
    binary_mask = create_padded_mask(binary_mask)
    
    if debug_session:
        # Save padded mask for comparison
        debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_padded_mask.jpg")
    
    # Apply ultra-organic curves for sinuous, flowing boundaries
    if debug_session:
        debug_session.log("Applying organic curves for sinuous boundaries...")
        # Save pre-organic mask for comparison
        debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_before_organic.jpg")
    
    # Create KISS organic blob with flowing curves  
    binary_mask = create_kiss_organic_blob(binary_mask)
    
    if debug_session:
        # Save organic curves mask for comparison
        debug_session.save_image(binary_mask, f"frame_{frame_index:04d}_organic_curves_mask.jpg")
        
        # Analyze blob characteristics
        blob_analysis = analyze_blob_characteristics(original_contours)
        debug_session.save_data(blob_analysis, f"frame_{frame_index:04d}_blob_analysis.json")
        
        # Visualize blob detection
        visualize_blob_detection(frame, original_contours, expanded_contours, debug_session)
    
    if np.sum(binary_mask) == 0:  # No valid mask generated
        if debug_session:
            debug_session.log("No valid mask generated")
        return frame
    
    if progress_callback:
        progress_callback("Creating soft mask...")
    
    # Step 3: Create soft mask for blending
    if debug_session:
        debug_session.next_step("Mask Generation")
    
    # Save enhanced mask for debug visualization
    enhanced_mask = None
    if enable_outward_noise and debug_session:
        from blob_utils import create_outward_noise_mask
        enhanced_mask = create_outward_noise_mask(binary_mask)
    
    soft_mask = create_soft_mask(binary_mask, enable_outward_noise=enable_outward_noise)
    
    if debug_session:
        visualize_masks(binary_mask, soft_mask, debug_session, enhanced_mask)
    
    if progress_callback:
        progress_callback("Generating AI content...")
    
    # Step 4: Generate artistic content using diffusion model
    if debug_session:
        debug_session.next_step("AI Content Generation")
    
    ai_generated = inpaint_region(
        image=frame,
        mask=binary_mask,
        prompt=prompt
    )
    
    # Convert PIL to OpenCV format
    ai_frame = pil_to_opencv(ai_generated)
    
    # Save individual diffusion result
    if debug_session:
        visualize_ai_generation(frame, ai_frame, binary_mask, prompt, debug_session)
        # Save the pure AI generated frame
        debug_session.save_image(ai_frame, f"frame_{frame_index:04d}_ai_generated.jpg")
    elif save_frames:
        # Create a simple output folder for diffusion frames
        import os
        diffusion_frames_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diffusion_frames")
        os.makedirs(diffusion_frames_dir, exist_ok=True)
        
        # Save the AI generated frame
        ai_output_path = os.path.join(diffusion_frames_dir, f"frame_{frame_index:04d}_ai_generated.jpg")
        cv2.imwrite(ai_output_path, ai_frame)
        
        # Also save a comparison image
        h, w = frame.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = frame  # Original on left
        comparison[:, w:] = ai_frame  # AI generated on right
        
        comparison_path = os.path.join(diffusion_frames_dir, f"frame_{frame_index:04d}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        
        if frame_index % 10 == 0:  # Log every 10th frame to avoid spam
            print(f"Saved diffusion results for frame {frame_index} to: {diffusion_frames_dir}")
    
    if progress_callback:
        progress_callback("Blending results...")
    
    # Step 5: Blend original and AI-generated content
    if debug_session:
        debug_session.next_step("Final Blending")
    
    result_frame = blend_with_mask(frame, ai_frame, soft_mask)
    
    if debug_session:
        visualize_blending(frame, ai_frame, soft_mask, result_frame, debug_session)
        debug_session.save_image(result_frame, f"frame_{frame_index:04d}_final.jpg")
    
    return result_frame


def process_video(input_path: str,
                 output_path: str,
                 prompt: str = DEFAULT_PROMPT,
                 max_frames: Optional[int] = None,
                 progress_callback: Optional[Callable] = None,
                 enable_debug: bool = DEBUG_MODE,
                 save_frames: bool = False,
                 enable_outward_noise: bool = True) -> bool:
    """
    Process an entire video through the BlobTrace Art pipeline.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        prompt: Text prompt for AI generation
        max_frames: Maximum number of frames to process (for testing)
        progress_callback: Optional callback for progress updates
        enable_debug: Enable debug mode with detailed logging and visualization
        
    Returns:
        True if successful, False otherwise
    """
    debug_session = None
    start_time = time.time()
    
    try:
        # Initialize debug session if enabled
        if enable_debug:
            session_name = f"video_{os.path.basename(input_path).split('.')[0]}_{int(start_time)}"
            debug_session = DebugSession(session_name)
            debug_session.log(f"Debug mode enabled for video: {input_path}")
            
            # Limit frames in debug mode
            if max_frames is None:
                max_frames = DEBUG_MAX_FRAMES
                debug_session.log(f"Debug mode: limiting to {max_frames} frames")
        
        # Get video information
        video_info = get_video_info(input_path)
        total_frames = min(video_info['total_frames'], max_frames) if max_frames else video_info['total_frames']
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"FPS: {video_info['fps']}")
        print(f"Total frames: {total_frames}")
        print(f"Prompt: {prompt}")
        print(f"Debug mode: {'ON' if enable_debug else 'OFF'}")
        
        if debug_session:
            debug_session.save_data(video_info, "video_info.json")
            debug_session.save_data({"prompt": prompt, "max_frames": max_frames}, "processing_params.json")
        
        # Initialize video writer
        with VideoWriter(
            output_path, 
            video_info['width'], 
            video_info['height'], 
            video_info['fps']
        ) as writer:
            
            # Process frames
            frame_count = 0
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                for frame, frame_idx in extract_frames(input_path):
                    if max_frames and frame_count >= max_frames:
                        break
                    
                    # Update progress
                    if progress_callback:
                        progress = frame_count / total_frames
                        progress_callback(f"Processing frame {frame_count + 1}/{total_frames}", progress)
                    
                    # Process frame with debug support
                    processed_frame = process_single_frame(
                        frame, 
                        prompt,
                        lambda msg: None,  # Disable sub-progress for batch processing
                        debug_session,
                        frame_count,
                        save_frames,
                        enable_outward_noise
                    )
                    
                    # Write to output video
                    writer.write_frame(processed_frame)
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # In debug mode, process fewer frames by default
                    if enable_debug and frame_count >= total_frames:
                        break
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Video processing completed! Output saved to: {output_path}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {frame_count/processing_time:.2f}")
        
        # Finalize debug session
        if debug_session:
            create_processing_summary(debug_session, video_info, frame_count, processing_time)
            debug_session.finalize()
            print(f"Debug results saved to: {debug_session.session_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        if progress_callback:
            progress_callback(f"Error: {str(e)}", 0)
        if debug_session:
            debug_session.log(f"Processing failed: {str(e)}", "ERROR")
            debug_session.finalize()
        return False


def main():
    """
    Command-line interface for BlobTrace Art.
    """
    parser = argparse.ArgumentParser(description="BlobTrace Art - Artistic Video Reinterpreter")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output video file path")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, 
                       help=f"Text prompt for AI generation (default: '{DEFAULT_PROMPT}')")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process (for testing)")
    parser.add_argument("--artistic", action="store_true", 
                       help="Use random artistic style modifiers")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode with step-by-step visualization")
    parser.add_argument("--debug-step-by-step", action="store_true", 
                       help="Enable step-by-step debug mode (pauses between steps)")
    parser.add_argument("--debug-no-viz", action="store_true", 
                       help="Disable debug visualizations (save files only)")
    
    args = parser.parse_args()
    
    # Set input path
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(INPUT_DIR, input_path)
    
    # Set output path
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(OUTPUT_DIR, args.output)
    else:
        output_path = os.path.join(OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME)
    
    # Set prompt
    prompt = args.prompt
    if args.artistic:
        prompt = generate_artistic_prompt(prompt)
        print(f"Using artistic prompt: {prompt}")
    
    # Configure debug mode
    enable_debug = args.debug or DEBUG_MODE
    if args.debug_step_by_step:
        enable_debug = True
        # Temporarily modify debug settings
        import config
        config.DEBUG_STEP_BY_STEP = True
        
    if args.debug_no_viz:
        import config
        config.DEBUG_SHOW_VISUALIZATIONS = False
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Process video
    success = process_video(
        input_path=input_path,
        output_path=output_path,
        prompt=prompt,
        max_frames=args.max_frames,
        enable_debug=enable_debug
    )
    
    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")


if __name__ == "__main__":
    main() 
