"""
Video utilities for BlobTrace Art
Handles video reading and writing operations
"""
import cv2
import numpy as np
from typing import Generator, Tuple, Optional
import os
from config import DEFAULT_FPS, VIDEO_CODEC


def extract_frames(video_path: str) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to input video file
        
    Yields:
        Tuple of (frame, frame_number)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, frame_count
        frame_count += 1
    
    cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video information (fps, width, height, total frames).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'fps': fps if fps > 0 else DEFAULT_FPS,
        'width': width,
        'height': height,
        'total_frames': total_frames
    }


class VideoWriter:
    """
    Video writer class for saving processed frames to video file.
    """
    
    def __init__(self, output_path: str, width: int, height: int, fps: float = DEFAULT_FPS):
        """
        Initialize video writer.
        
        Args:
            output_path: Path for output video file
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize video writer with fallback codecs
        codecs_to_try = [VIDEO_CODEC, "avc1", "mp4v", "XVID"]
        self.writer = None
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if writer.isOpened():
                    self.writer = writer
                    print(f"Using video codec: {codec}")
                    break
                else:
                    writer.release()
            except Exception as e:
                print(f"Codec {codec} failed: {e}")
                continue
        
        if not self.writer or not self.writer.isOpened():
            raise ValueError(f"Could not open video writer for: {output_path}. Tried codecs: {codecs_to_try}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write (BGR format)
        """
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
    
    def release(self):
        """Release the video writer."""
        if self.writer:
            self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def create_video_from_frames(frames: list, output_path: str, fps: float = DEFAULT_FPS):
    """
    Create video from list of frames.
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Path for output video
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    
    with VideoWriter(output_path, width, height, fps) as writer:
        for frame in frames:
            writer.write_frame(frame)


def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize frame to target dimensions.
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, (target_width, target_height))


def create_video_from_frames_with_audio(frames: list, output_path: str, original_video_path: str, fps: float = DEFAULT_FPS):
    """
    Create video from list of frames and preserve audio from original video.
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Path for output video
        original_video_path: Path to original video (for audio extraction)
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # First create video without audio
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    create_video_from_frames(frames, temp_video_path, fps)
    
    # Then combine with audio using ffmpeg
    try:
        import subprocess
        
        # Use ffmpeg to combine video and audio
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', temp_video_path,  # Input video (no audio)
            '-i', original_video_path,  # Input audio source
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'aac',   # Encode audio as AAC
            '-map', '0:v:0',  # Map video from first input
            '-map', '1:a:0',  # Map audio from second input
            '-shortest',      # End when shortest stream ends
            output_path
        ]
        
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Success - remove temp file
            os.remove(temp_video_path)
            return True
        else:
            # Failed - keep temp file as fallback
            print(f"FFmpeg failed: {result.stderr}")
            print("Keeping video without audio as fallback")
            os.rename(temp_video_path, output_path)
            return False
            
    except (ImportError, FileNotFoundError):
        # ffmpeg not available - keep video without audio
        print("FFmpeg not available - saving video without audio")
        os.rename(temp_video_path, output_path)
        return False 
