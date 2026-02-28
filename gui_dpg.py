"""
🌀 KISS Edition: Minimal Dear PyGui Interface for BlobTrace Art
Ultra-reliable, modern GUI with drag-and-drop and real-time preview
"""
import dearpygui.dearpygui as dpg
import threading
import os
import time
import cv2
import numpy as np
from typing import Optional

from config import INPUT_DIR, OUTPUT_DIR, DEFAULT_PROMPT, DEBUG_DIR
from main import process_video
from video_utils import get_video_info
from diffusion import get_pipeline_info, generate_artistic_prompt

# ===== Desktop UI/UX Standards - Readable Sizing =====
WINDOW_WIDTH = 2400
WINDOW_HEIGHT = 1800
PREVIEW_WIDTH = 750
PREVIEW_HEIGHT = 550
SETTINGS_WIDTH = 700
BUTTON_HEIGHT = 45
SLIDER_WIDTH = 350
LOG_HEIGHT = 200
SPACING = 20
PADDING = 25

class BlobTraceGUI:
    """🌀 Minimal Dear PyGui interface for artistic video processing"""
    
    def __init__(self):
        # State
        self.current_video_path = None
        self.is_processing = False
        self.processing_thread = None
        
        # Initialize Dear PyGui with proper desktop sizing
        dpg.create_context()
        dpg.create_viewport(
            title="🌀 BlobTrace Art - Professional Edition", 
            width=WINDOW_WIDTH, 
            height=WINDOW_HEIGHT,
            min_width=1400,
            min_height=900
        )
        
        dpg.setup_dearpygui()
        
        # Set up readable font scaling for desktop use
        self._setup_font_scaling()
        
        self.create_interface()
        self.check_dependencies()
    
    def _setup_font_scaling(self):
        """Set up readable font scaling for desktop interface"""
        # Use Dear PyGui's global font scaling for better readability
        dpg.set_global_font_scale(2.0)  # 100% larger fonts for proper desktop readability
        
    def create_interface(self):
        """Create professional desktop interface following HCI/UX standards"""
        
        # Main window with proper desktop sizing
        with dpg.window(label="BlobTrace Art", tag="main_window", 
                       width=WINDOW_WIDTH-20, height=WINDOW_HEIGHT-40):
            
            self._create_header()
            self._create_file_section()
            self._create_settings_section()
            self._create_processing_section()
            self._create_preview_section()
            self._create_log_section()
        
        # Set main window as primary
        dpg.set_primary_window("main_window", True)
    
    def _create_header(self):
        """Create application header"""
        dpg.add_text("🌀 BlobTrace Art - Professional Video Processor", 
                    color=(80, 140, 255))
        dpg.add_text("Transform your videos with AI-powered artistic effects", 
                    color=(150, 150, 150))
        dpg.add_separator()
        dpg.add_spacer(height=SPACING)
    
    def _create_file_section(self):
        """Create file selection section"""
        dpg.add_text("📁 Video File", color=(200, 200, 200))
        dpg.add_spacer(height=5)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select Video File", callback=self.select_video, 
                          width=180, height=BUTTON_HEIGHT)
            dpg.add_spacer(width=SPACING)
            dpg.add_text("No video selected", tag="video_status", color=(150, 150, 150))
        
        dpg.add_spacer(height=SPACING * 2)
    
    def _create_settings_section(self):
        """Create settings section with proper organization"""
        with dpg.collapsing_header(label="⚙️ Processing Settings", default_open=True):
            dpg.add_spacer(height=SPACING)
            
            # AI Prompt Section
            dpg.add_text("🎨 AI Generation Prompt", color=(200, 200, 200))
            dpg.add_input_text(tag="prompt", default_value=DEFAULT_PROMPT, 
                             width=SETTINGS_WIDTH, height=60, multiline=True)
            dpg.add_spacer(height=SPACING)
            
            # Processing Options
            dpg.add_text("🔧 Processing Options", color=(200, 200, 200))
            dpg.add_spacer(height=10)
            
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_checkbox(label="Random artistic styles", tag="artistic", default_value=False)
                    dpg.add_spacer(height=5)
                    dpg.add_checkbox(label="Save individual frames", tag="save_frames", default_value=False)
                    dpg.add_spacer(height=5)
                    dpg.add_checkbox(label="Outward flowing noise", tag="outward_noise", default_value=True)
                
                dpg.add_spacer(width=SPACING * 4)
                
                with dpg.group():
                    dpg.add_checkbox(label="Debug mode", tag="debug", default_value=False)
                    dpg.add_spacer(height=5)
                    dpg.add_checkbox(label="Flowing connections", tag="flowing_masks", default_value=True)
                    dpg.add_spacer(height=5)
                    dpg.add_checkbox(label="Convex hull smoothing", tag="convex_hull", default_value=False)
            
            dpg.add_spacer(height=SPACING)
            
            # Effect Intensity Controls
            dpg.add_text("🎛️ Effect Intensity", color=(200, 200, 200))
            dpg.add_spacer(height=5)
            
            self._create_slider_control("Mask Smoothing", "smoothing", 0.5, self.update_smoothing_label)
            self._create_slider_control("Mask Padding", "padding", 0.7, self.update_padding_label)
            self._create_slider_control("Organic Curves", "organic", 0.8, self.update_organic_label)
            
            # Testing Options
            dpg.add_spacer(height=SPACING)
            dpg.add_text("🧪 Testing", color=(200, 200, 200))
            dpg.add_spacer(height=5)
            
            with dpg.group(horizontal=True):
                dpg.add_text("Max frames (0 = all):")
                dpg.add_spacer(width=SPACING)
                dpg.add_input_int(tag="max_frames", width=100, default_value=0)
            
            dpg.add_spacer(height=SPACING)
    
    def _create_slider_control(self, label, tag, default_value, callback):
        """Create a standardized slider control with label and percentage"""
        with dpg.group(horizontal=True):
            dpg.add_text(f"{label}:")
            dpg.add_spacer(width=10)
            dpg.add_slider_float(tag=tag, default_value=default_value, min_value=0.0, max_value=1.0, 
                               width=SLIDER_WIDTH, callback=callback)
            dpg.add_spacer(width=SPACING)
            dpg.add_text(f"{int(default_value * 100)}%", tag=f"{tag}_label")
        dpg.add_spacer(height=8)
    
    def _create_processing_section(self):
        """Create processing control section"""
        dpg.add_separator()
        dpg.add_spacer(height=SPACING)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="🚀 Process Video", callback=self.start_processing, 
                          width=200, height=50, tag="process_btn")
            
            dpg.add_spacer(width=SPACING * 2)
            
            # Status and progress
            with dpg.group():
                dpg.add_progress_bar(tag="progress", width=500, height=25)
                dpg.add_spacer(height=5)
                dpg.add_text("Ready to process", tag="status", color=(100, 200, 100))
        
        dpg.add_spacer(height=SPACING * 2)
    
    def _create_preview_section(self):
        """Create video preview section"""
        dpg.add_text("👁️ Video Preview", color=(200, 200, 200))
        dpg.add_spacer(height=SPACING)
        
        with dpg.group(horizontal=True):
            # Original preview
            with dpg.child_window(label="Original Video", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT,
                                border=True):
                dpg.add_text("No video loaded", tag="preview_original")
            
            dpg.add_spacer(width=SPACING * 2)
            
            # Processed preview
            with dpg.child_window(label="Processed Result", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT,
                                border=True):
                dpg.add_text("Processing not started", tag="preview_processed")
        
        dpg.add_spacer(height=SPACING * 2)
    
    def _create_log_section(self):
        """Create application log section"""
        dpg.add_separator()
        dpg.add_text("📋 Application Log", color=(200, 200, 200))
        dpg.add_spacer(height=5)
        
        with dpg.child_window(height=LOG_HEIGHT, horizontal_scrollbar=True, border=True):
            dpg.add_text("", tag="log_text")
    
    def update_smoothing_label(self):
        """Update smoothing percentage display"""
        self._update_slider_label("smoothing")
    
    def update_padding_label(self):
        """Update padding percentage display"""
        self._update_slider_label("padding")
    
    def update_organic_label(self):
        """Update organic curves percentage display"""
        self._update_slider_label("organic")
    
    def _update_slider_label(self, slider_tag):
        """Helper method to update slider percentage labels"""
        value = dpg.get_value(slider_tag)
        dpg.set_value(f"{slider_tag}_label", f"{int(value * 100)}%")
    
    def select_video(self):
        """File selection dialog"""
        def callback(sender, app_data):
            if app_data and 'file_path_name' in app_data:
                file_path = app_data['file_path_name']
                # Clean up any weird extensions that might be added
                if file_path.endswith('.*'):
                    file_path = file_path[:-2]
                self.log(f"🔍 Selected file path: {file_path}")
                self.on_video_selected(file_path)
        
        with dpg.file_dialog(
            directory_selector=False, show=True, callback=callback,
            file_count=1, width=700, height=500,
            default_path=INPUT_DIR
        ):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
            dpg.add_file_extension(".avi", color=(0, 255, 0, 255))
            dpg.add_file_extension(".mov", color=(0, 255, 0, 255))
            dpg.add_file_extension(".mkv", color=(0, 255, 0, 255))
            dpg.add_file_extension(".ts", color=(0, 255, 0, 255))  # Transport Stream
            dpg.add_file_extension(".m4v", color=(0, 255, 0, 255))
            dpg.add_file_extension(".webm", color=(0, 255, 0, 255))
    
    def on_video_selected(self, filepath):
        """Handle video selection"""
        # Verify file exists
        if not os.path.exists(filepath):
            self.log(f"❌ File not found: {filepath}")
            dpg.set_value("video_status", "File not found!")
            dpg.configure_item("video_status", color=(255, 100, 100))
            return
        
        # Verify it's a readable file
        if not os.path.isfile(filepath):
            self.log(f"❌ Not a valid file: {filepath}")
            dpg.set_value("video_status", "Invalid file!")
            dpg.configure_item("video_status", color=(255, 100, 100))
            return
        
        self.current_video_path = filepath
        filename = os.path.basename(filepath)
        
        dpg.set_value("video_status", f"Selected: {filename}")
        dpg.configure_item("video_status", color=(100, 200, 100))
        
        # Load and display preview
        self._load_preview(filepath, "original")
        
        # Log video info
        try:
            video_info = get_video_info(filepath)
            self.log(f"✅ Video loaded: {filename}")
            self.log(f"📐 Resolution: {video_info['width']}x{video_info['height']}")
            self.log(f"🎬 FPS: {video_info['fps']}, Frames: {video_info['total_frames']}")
        except Exception as e:
            self.log(f"⚠️ Could not read video info: {e}")
            self.log(f"🔍 File path used: {filepath}")
            self.log(f"📁 File exists: {os.path.exists(filepath)}")
            self.log(f"📄 File size: {os.path.getsize(filepath) if os.path.exists(filepath) else 'N/A'} bytes")
    
    def _load_preview(self, video_path, preview_type):
        """Load video frame and display in preview"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.log(f"⚠️ Could not read frame from {preview_type} video")
                return
            
            # Calculate proper aspect ratio for preview
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio > 1:  # Landscape
                preview_w = min(PREVIEW_WIDTH - 40, width)
                preview_h = int(preview_w / aspect_ratio)
            else:  # Portrait or square
                preview_h = min(PREVIEW_HEIGHT - 40, height)
                preview_w = int(preview_h * aspect_ratio)
            
            # Resize and convert for Dear PyGui
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (preview_w, preview_h))
            texture_data = frame_resized.flatten() / 255.0
            
            # Create or update texture
            texture_tag = f"{preview_type}_texture"
            try:
                dpg.delete_item(texture_tag)
            except:
                pass
            
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(width=preview_w, height=preview_h, 
                                  default_value=texture_data, format=dpg.mvFormat_Float_rgb,
                                  tag=texture_tag)
            
            # Update preview display
            preview_tag = f"preview_{preview_type}"
            try:
                dpg.delete_item(preview_tag)
            except:
                pass
            
            # Find the correct parent window
            parent_windows = dpg.get_item_children("main_window", slot=1)
            if preview_type == "original":
                parent = parent_windows[4][0] if len(parent_windows) > 4 else "main_window"  # First preview window
            else:
                parent = parent_windows[4][2] if len(parent_windows) > 4 else "main_window"  # Second preview window
            
            dpg.add_image(texture_tag, tag=preview_tag, parent=parent)
            
        except Exception as e:
            self.log(f"❌ Error loading {preview_type} preview: {e}")
    
    def start_processing(self):
        """Start video processing"""
        if self.is_processing:
            self.log("⚠️ Processing already in progress!")
            return
        
        if not self.current_video_path:
            self.log("❌ Please select a video file first!")
            return
        
        # Get settings
        prompt = dpg.get_value("prompt").strip() or DEFAULT_PROMPT
        
        if dpg.get_value("artistic"):
            prompt = generate_artistic_prompt(prompt)
            self.log(f"🎨 Using artistic prompt: {prompt}")
        
        max_frames = dpg.get_value("max_frames") or None
        
        # Output path
        input_filename = os.path.basename(self.current_video_path)
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_processed{ext}")
        
        # Start processing
        self.is_processing = True
        dpg.configure_item("process_btn", enabled=False)
        dpg.set_value("progress", 0.0)
        dpg.set_value("status", "Starting processing...")
        dpg.configure_item("status", color=(255, 200, 100))
        
        self.processing_thread = threading.Thread(
            target=self.process_video_thread,
            args=(self.current_video_path, output_path, prompt, max_frames)
        )
        self.processing_thread.start()
    
    def process_video_thread(self, input_path, output_path, prompt, max_frames):
        """Process video in separate thread"""
        try:
            def progress_callback(message, progress=None):
                dpg.set_value("status", message)
                if progress is not None:
                    dpg.set_value("progress", progress)
            
            # Update config
            import config
            config.ENABLE_FLOWING_MASKS = dpg.get_value("flowing_masks")
            config.SMOOTHING_INTENSITY = dpg.get_value("smoothing")
            config.ENABLE_CONVEX_HULL_SMOOTHING = dpg.get_value("convex_hull")
            config.PADDING_INTENSITY = dpg.get_value("padding")
            config.CURVE_INTENSITY = dpg.get_value("organic")
            
            # Process
            success = process_video(
                input_path=input_path,
                output_path=output_path,
                prompt=prompt,
                max_frames=max_frames,
                progress_callback=progress_callback,
                enable_debug=dpg.get_value("debug"),
                save_frames=dpg.get_value("save_frames"),
                enable_outward_noise=dpg.get_value("outward_noise")
            )
            
            # Completion
            if success:
                dpg.set_value("progress", 1.0)
                dpg.set_value("status", "✅ Processing completed!")
                dpg.configure_item("status", color=(100, 255, 100))
                self.log(f"✅ Output saved: {output_path}")
                
                # Load processed preview
                self._load_preview(output_path, "processed")
                    
            else:
                dpg.set_value("status", "❌ Processing failed!")
                dpg.configure_item("status", color=(255, 100, 100))
                self.log("❌ Processing failed!")
            
        except Exception as e:
            dpg.set_value("status", f"❌ Error: {str(e)}")
            dpg.configure_item("status", color=(255, 100, 100))
            self.log(f"❌ Error: {str(e)}")
        
        finally:
            self.is_processing = False
            dpg.configure_item("process_btn", enabled=True)
    
    def check_dependencies(self):
        """Check AI dependencies"""
        try:
            self.log("🔍 Checking AI dependencies...")
            pipeline_info = get_pipeline_info()
            self.log(f"✅ AI Model: {pipeline_info['model_name']}")
            self.log(f"✅ Device: {pipeline_info['device']}")
            self.log("✅ All dependencies ready!")
        except Exception as e:
            self.log(f"⚠️ Warning: {e}")
            self.log("AI models will be downloaded on first use")
    
    def log(self, message):
        """Add message to log"""
        current = dpg.get_value("log_text")
        new_log = f"{current}\n{message}" if current else message
        dpg.set_value("log_text", new_log)
    
    def run(self):
        """Start the GUI"""
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    """Launch the KISS GUI"""
    app = BlobTraceGUI()
    app.run()


if __name__ == "__main__":
    main() 
