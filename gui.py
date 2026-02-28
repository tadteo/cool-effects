"""
LEGACY tkinter GUI module for BlobTrace Art (replaced by gui_dpg.py)
This is kept for reference but the new Dear PyGui version is recommended
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import time
import traceback
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, PREVIEW_WIDTH, PREVIEW_HEIGHT,
    INPUT_DIR, OUTPUT_DIR, DEFAULT_PROMPT
)
from main import process_video
from video_utils import get_video_info
from diffusion import get_pipeline_info, generate_artistic_prompt

# ===== FONT SIZE CONSTANTS (EASY TO CHANGE) =====
TITLE_FONT_SIZE = 56        # Main title
LARGE_TEXT_SIZE = 18        # Drop area text, important labels
NORMAL_TEXT_SIZE = 16       # Regular labels, buttons
SMALL_TEXT_SIZE = 14        # Log text, detailed info
FONT_FAMILY = 'Arial'       # Font family for all text

# Widget dimension constants
DROP_AREA_HEIGHT = 120      # Height of drag-drop area
LOG_HEIGHT = 10             # Height of log text area in lines
# ==================================================


class DragDropFrame(tk.Frame):
    """
    Frame that supports drag and drop functionality for video files.
    """
    
    def __init__(self, parent, callback=None):
        super().__init__(parent)
        self.callback = callback
        self.configure(bg='lightgray', relief='raised', bd=2)
        
        # Create drop area
        self.drop_label = tk.Label(
            self, 
            text="Drag and drop video file here\n(or click to browse)",
            font=(FONT_FAMILY, LARGE_TEXT_SIZE),
            bg='lightgray',
            fg='darkgray',
            justify='center'
        )
        self.drop_label.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Bind events
        self.drop_label.bind('<Button-1>', self.browse_file)
        self.bind('<Button-1>', self.browse_file)
        
        # Configure drag and drop (simplified - tkinter doesn't have native DnD)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
    def on_enter(self, event):
        """Handle mouse enter event."""
        self.configure(bg='lightblue')
        self.drop_label.configure(bg='lightblue')
        
    def on_leave(self, event):
        """Handle mouse leave event."""
        self.configure(bg='lightgray')
        self.drop_label.configure(bg='lightgray')
        
    def browse_file(self, event=None):
        """Open file browser dialog."""
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select video file",
            filetypes=filetypes,
            initialdir=INPUT_DIR
        )
        
        if filename and self.callback:
            self.callback(filename)
            
    def set_file(self, filepath):
        """Set the current file and update display."""
        filename = os.path.basename(filepath)
        self.drop_label.configure(
            text=f"Selected: {filename}\n\nClick to change file",
            fg='darkgreen'
        )


class VideoPreview(tk.Frame):
    """
    Frame for displaying video preview thumbnails.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Original preview
        self.original_frame = tk.LabelFrame(self, text="Original", padx=5, pady=5)
        self.original_frame.pack(side='left', padx=5, pady=5)
        
        self.original_label = tk.Label(
            self.original_frame,
            text="No video loaded",
            width=40, height=20,
            bg='white',
            font=(FONT_FAMILY, NORMAL_TEXT_SIZE)
        )
        self.original_label.pack()
        
        # Processed preview
        self.processed_frame = tk.LabelFrame(self, text="Processed", padx=5, pady=5)
        self.processed_frame.pack(side='right', padx=5, pady=5)
        
        self.processed_label = tk.Label(
            self.processed_frame,
            text="Processing not started",
            width=40, height=20,
            bg='white',
            font=(FONT_FAMILY, NORMAL_TEXT_SIZE)
        )
        self.processed_label.pack()
        
    def set_original_preview(self, image_path_or_array):
        """Set the original video preview."""
        try:
            if isinstance(image_path_or_array, str):
                image = Image.open(image_path_or_array)
            else:
                # Convert numpy array to PIL Image
                if len(image_path_or_array.shape) == 3:
                    image_rgb = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image_rgb)
                else:
                    image = Image.fromarray(image_path_or_array)
            
            # Resize to fit preview
            image.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.original_label.configure(image=photo, text="")
            self.original_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error setting original preview: {e}")
            
    def set_processed_preview(self, image_path_or_array):
        """Set the processed video preview."""
        try:
            if isinstance(image_path_or_array, str):
                image = Image.open(image_path_or_array)
            else:
                # Convert numpy array to PIL Image
                if len(image_path_or_array.shape) == 3:
                    image_rgb = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image_rgb)
                else:
                    image = Image.fromarray(image_path_or_array)
            
            # Resize to fit preview
            image.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.processed_label.configure(image=photo, text="")
            self.processed_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error setting processed preview: {e}")


class BlobtraceArtGUI:
    """
    Main GUI application for BlobTrace Art.
    """
    
    def __init__(self):
        print("Starting BlobTrace Art GUI...")
        self.root = tk.Tk()
        self.root.title("BlobTrace Art - Artistic Video Reinterpreter")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Font zoom system (offset applied to constants)
        self.font_zoom_offset = 0  # Offset to add to all font constants
        self.min_zoom_offset = -6
        self.max_zoom_offset = 12
        
        # Variables
        self.current_video_path = None
        self.processing_thread = None
        self.is_processing = False
        
        print("Setting up interface...")
        self.setup_ui()
        
        # Setup keyboard shortcuts for font zoom
        self.setup_keyboard_shortcuts()
        
        print("Checking dependencies...")
        self.check_dependencies()
        print("GUI ready!")
        
    def setup_ui(self):
        """Set up the user interface."""
        # Configure ttk styles for larger fonts
        style = ttk.Style()
        style.configure('Large.TCheckbutton', font=self.get_font(NORMAL_TEXT_SIZE))
        style.configure('Large.TButton', font=self.get_font(NORMAL_TEXT_SIZE))
        style.configure('Large.TLabel', font=self.get_font(NORMAL_TEXT_SIZE))
        # For LabelFrame, we need to configure the Label part specifically
        style.configure('TLabelFrame.Label', font=self.get_font(NORMAL_TEXT_SIZE))
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        self.title_label = tk.Label(
            main_frame, 
            text="BlobTrace Art", 
            font=self.get_font(TITLE_FONT_SIZE, 'bold'),
            fg='darkblue'
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Drag and drop area
        self.drag_drop = DragDropFrame(main_frame, self.on_video_selected)
        self.drag_drop.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.drag_drop.configure(height=DROP_AREA_HEIGHT)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Prompt input
        ttk.Label(settings_frame, text="AI Prompt:", font=self.get_font(NORMAL_TEXT_SIZE)).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        self.prompt_entry = ttk.Entry(settings_frame, textvariable=self.prompt_var, width=50, font=self.get_font(NORMAL_TEXT_SIZE))
        self.prompt_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # Artistic style checkbox
        self.artistic_var = tk.BooleanVar()
        self.artistic_check = ttk.Checkbutton(
            settings_frame, 
            text="Use random artistic styles", 
            variable=self.artistic_var
        )
        self.artistic_check.configure(style='Large.TCheckbutton')
        self.artistic_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Debug mode checkbox
        self.debug_var = tk.BooleanVar()
        self.debug_check = ttk.Checkbutton(
            settings_frame, 
            text="Enable debug mode (saves intermediate results)", 
            variable=self.debug_var
        )
        self.debug_check.configure(style='Large.TCheckbutton')
        self.debug_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Save frames checkbox
        self.save_frames_var = tk.BooleanVar()
        self.save_frames_check = ttk.Checkbutton(
            settings_frame, 
            text="Save individual AI-generated frames during processing", 
            variable=self.save_frames_var
        )
        self.save_frames_check.configure(style='Large.TCheckbutton')
        self.save_frames_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Outward noise checkbox
        self.outward_noise_var = tk.BooleanVar(value=True)
        self.outward_noise_check = ttk.Checkbutton(
            settings_frame, 
            text="Enable outward flowing noise on masks (makes masks overflow)", 
            variable=self.outward_noise_var
        )
        self.outward_noise_check.configure(style='Large.TCheckbutton')
        self.outward_noise_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Flowing masks checkbox
        self.flowing_masks_var = tk.BooleanVar(value=True)
        self.flowing_masks_check = ttk.Checkbutton(
            settings_frame, 
            text="Enable flowing, wavy connections between nearby masks", 
            variable=self.flowing_masks_var
        )
        self.flowing_masks_check.configure(style='Large.TCheckbutton')
        self.flowing_masks_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # NEW: Mask smoothing controls
        # Smoothing intensity slider with percentage display
        smoothing_frame = ttk.Frame(settings_frame)
        smoothing_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(smoothing_frame, text="Mask smoothing intensity:", font=self.get_font(NORMAL_TEXT_SIZE)).pack(side=tk.LEFT)
        
        self.smoothing_var = tk.DoubleVar(value=0.5)
        self.smoothing_scale = ttk.Scale(smoothing_frame, from_=0.0, to=1.0, variable=self.smoothing_var, orient=tk.HORIZONTAL, length=200)
        self.smoothing_scale.pack(side=tk.LEFT, padx=(10, 5))
        
        self.smoothing_label = ttk.Label(smoothing_frame, text="50%", font=self.get_font(NORMAL_TEXT_SIZE))
        self.smoothing_label.pack(side=tk.LEFT)
        
        # Update percentage display when slider changes
        def update_smoothing_label(*args):
            percentage = int(self.smoothing_var.get() * 100)
            self.smoothing_label.configure(text=f"{percentage}%")
        self.smoothing_var.trace('w', update_smoothing_label)
        
        # Convex hull checkbox
        self.convex_hull_var = tk.BooleanVar(value=False)
        self.convex_hull_check = ttk.Checkbutton(
            settings_frame, 
            text="Enable convex hull smoothing (ultra-smooth shapes)", 
            variable=self.convex_hull_var
        )
        self.convex_hull_check.configure(style='Large.TCheckbutton')
        self.convex_hull_check.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # NEW: Mask padding controls
        # Padding intensity slider with percentage display
        padding_frame = ttk.Frame(settings_frame)
        padding_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(padding_frame, text="Mask padding (enlargement):", font=self.get_font(NORMAL_TEXT_SIZE)).pack(side=tk.LEFT)
        
        self.padding_var = tk.DoubleVar(value=0.7)
        self.padding_scale = ttk.Scale(padding_frame, from_=0.0, to=1.0, variable=self.padding_var, orient=tk.HORIZONTAL, length=200)
        self.padding_scale.pack(side=tk.LEFT, padx=(10, 5))
        
        self.padding_label = ttk.Label(padding_frame, text="70%", font=self.get_font(NORMAL_TEXT_SIZE))
        self.padding_label.pack(side=tk.LEFT)
        
        # Update percentage display when slider changes
        def update_padding_label(*args):
            percentage = int(self.padding_var.get() * 100)
            self.padding_label.configure(text=f"{percentage}%")
        self.padding_var.trace('w', update_padding_label)
        
        # NEW: Organic curves controls
        # Organic curve intensity slider with percentage display
        organic_frame = ttk.Frame(settings_frame)
        organic_frame.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(organic_frame, text="Organic curves (sinuous):", font=self.get_font(NORMAL_TEXT_SIZE)).pack(side=tk.LEFT)
        
        self.organic_var = tk.DoubleVar(value=0.8)
        self.organic_scale = ttk.Scale(organic_frame, from_=0.0, to=1.0, variable=self.organic_var, orient=tk.HORIZONTAL, length=200)
        self.organic_scale.pack(side=tk.LEFT, padx=(10, 5))
        
        self.organic_label = ttk.Label(organic_frame, text="80%", font=self.get_font(NORMAL_TEXT_SIZE))
        self.organic_label.pack(side=tk.LEFT)
        
        # Update percentage display when slider changes
        def update_organic_label(*args):
            percentage = int(self.organic_var.get() * 100)
            self.organic_label.configure(text=f"{percentage}%")
        self.organic_var.trace('w', update_organic_label)
        
        # Max frames for testing
        ttk.Label(settings_frame, text="Max frames (for testing):", font=self.get_font(NORMAL_TEXT_SIZE)).grid(row=10, column=0, sticky=tk.W, pady=2)
        self.max_frames_var = tk.StringVar()
        self.max_frames_entry = ttk.Entry(settings_frame, textvariable=self.max_frames_var, width=10, font=self.get_font(NORMAL_TEXT_SIZE))
        self.max_frames_entry.grid(row=10, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # Process button
        self.process_button = ttk.Button(
            main_frame, 
            text="Process Video", 
            command=self.start_processing,
            style='Large.TButton'
        )
        self.process_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, style='Large.TLabel')
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        progress_frame.columnconfigure(0, weight=1)
        
        # Preview frame
        self.preview = VideoPreview(main_frame)
        self.preview.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=LOG_HEIGHT, width=80, font=self.get_font(SMALL_TEXT_SIZE))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Configure main frame row weights
        main_frame.rowconfigure(6, weight=1)
        
    def check_dependencies(self):
        """Check if all dependencies are available."""
        try:
            self.log("Checking AI dependencies (may take a moment on first run)...")
            pipeline_info = get_pipeline_info()
            self.log(f"✓ AI Model: {pipeline_info['model_name']}")
            self.log(f"✓ Device: {pipeline_info['device']}")
            self.log("✓ All dependencies ready!")
        except Exception as e:
            self.log(f"⚠ Warning: {e}")
            self.log("AI models will be downloaded on first use")
            
    def log(self, message):
        """Add message to log."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def on_video_selected(self, filepath):
        """Handle video file selection."""
        self.current_video_path = filepath
        self.drag_drop.set_file(filepath)
        
        # Update preview with first frame
        try:
            cap = cv2.VideoCapture(filepath)
            ret, frame = cap.read()
            if ret:
                self.preview.set_original_preview(frame)
            cap.release()
            
            # Log video info
            video_info = get_video_info(filepath)
            self.log(f"Video loaded: {os.path.basename(filepath)}")
            self.log(f"Resolution: {video_info['width']}x{video_info['height']}")
            self.log(f"FPS: {video_info['fps']}")
            self.log(f"Total frames: {video_info['total_frames']}")
            
        except Exception as e:
            self.log(f"Error loading video preview: {e}")
            
    def start_processing(self):
        """Start video processing in a separate thread."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress!")
            return
            
        if not self.current_video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
            
        # Get settings
        prompt = self.prompt_var.get().strip()
        if not prompt:
            prompt = DEFAULT_PROMPT
            
        if self.artistic_var.get():
            prompt = generate_artistic_prompt(prompt)
            self.log(f"Using artistic prompt: {prompt}")
            
        max_frames = None
        if self.max_frames_var.get().strip():
            try:
                max_frames = int(self.max_frames_var.get())
            except ValueError:
                messagebox.showerror("Error", "Max frames must be a number!")
                return
                
        # Set output path
        input_filename = os.path.basename(self.current_video_path)
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_processed{ext}")
        
        # Start processing thread
        self.is_processing = True
        self.process_button.configure(state='disabled')
        self.progress_var.set(0)
        self.status_var.set("Starting processing...")
        
        self.processing_thread = threading.Thread(
            target=self.process_video_thread,
            args=(self.current_video_path, output_path, prompt, max_frames)
        )
        self.processing_thread.start()
        
    def process_video_thread(self, input_path, output_path, prompt, max_frames):
        """Process video in separate thread."""
        try:
            def progress_callback(message, progress=None):
                self.root.after(0, lambda: self.status_var.set(message))
                if progress is not None:
                    self.root.after(0, lambda: self.progress_var.set(progress * 100))
                    
            # Get debug mode setting
            enable_debug = self.debug_var.get()
            
            if enable_debug:
                self.root.after(0, lambda: self.log("Debug mode enabled - intermediate results will be saved"))
                    
            # Update config for flowing masks, smoothing, padding, and organic curves
            import config
            config.ENABLE_FLOWING_MASKS = self.flowing_masks_var.get()
            config.SMOOTHING_INTENSITY = self.smoothing_var.get()
            config.ENABLE_CONVEX_HULL_SMOOTHING = self.convex_hull_var.get()
            config.PADDING_INTENSITY = self.padding_var.get()
            config.CURVE_INTENSITY = self.organic_var.get()
            
            # Process video
            success = process_video(
                input_path=input_path,
                output_path=output_path,
                prompt=prompt,
                max_frames=max_frames,
                progress_callback=progress_callback,
                enable_debug=enable_debug,
                save_frames=self.save_frames_var.get(),
                enable_outward_noise=self.outward_noise_var.get()
            )
            
            # Update UI on completion
            self.root.after(0, lambda: self.on_processing_complete(success, output_path, enable_debug))
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.root.after(0, lambda: self.on_processing_error(error_msg))
            
    def on_processing_complete(self, success, output_path, debug_enabled=False):
        """Handle processing completion."""
        self.is_processing = False
        self.process_button.configure(state='normal')
        
        if success:
            self.progress_var.set(100)
            self.status_var.set("Processing completed successfully!")
            self.log(f"Output saved to: {output_path}")
            
            # Update processed preview
            try:
                cap = cv2.VideoCapture(output_path)
                ret, frame = cap.read()
                if ret:
                    self.preview.set_processed_preview(frame)
                cap.release()
            except Exception as e:
                self.log(f"Error loading processed preview: {e}")
            
            # Create completion message
            message = f"Video processing completed!\n\nOutput saved to:\n{output_path}"
            if debug_enabled:
                from config import DEBUG_DIR
                message += f"\n\nDebug results saved to:\n{DEBUG_DIR}"
                
            # Show completion message
            response = messagebox.askyesno(
                "Processing Complete", 
                message + "\n\nWould you like to open the output folder?"
            )
            
            if response:
                # Open output folder
                import subprocess, platform
                folder_to_open = DEBUG_DIR if debug_enabled else OUTPUT_DIR
                if platform.system() == "Windows":
                    subprocess.run(["explorer", folder_to_open])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", folder_to_open])
                else:  # Linux
                    subprocess.run(["xdg-open", folder_to_open])
        else:
            self.status_var.set("Processing failed!")
            self.log("Processing failed! Check the log for details.")
            
    def on_processing_error(self, error_msg):
        """Handle processing error."""
        self.is_processing = False
        self.process_button.configure(state='normal')
        self.status_var.set("Processing failed!")
        self.log(error_msg)
        messagebox.showerror("Error", error_msg)
        
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for font zoom."""
        # Bind zoom shortcuts (works on both Linux and other platforms)
        self.root.bind('<Control-plus>', self.increase_font_size)
        self.root.bind('<Control-equal>', self.increase_font_size)  # For keyboards where + requires Shift
        self.root.bind('<Control-minus>', self.decrease_font_size)
        self.root.bind('<Control-0>', self.reset_font_size)
        
        # Also bind Cmd shortcuts for Mac compatibility
        self.root.bind('<Command-plus>', self.increase_font_size)
        self.root.bind('<Command-equal>', self.increase_font_size)
        self.root.bind('<Command-minus>', self.decrease_font_size)
        self.root.bind('<Command-0>', self.reset_font_size)
        
        # Make sure the window can receive focus for shortcuts
        self.root.focus_set()
        
        # Log the shortcuts
        self.root.after(1000, lambda: self.log("💡 Font zoom shortcuts: Ctrl+/Ctrl- to zoom, Ctrl+0 to reset"))
        
    def increase_font_size(self, event=None):
        """Increase font size."""
        if self.font_zoom_offset < self.max_zoom_offset:
            self.font_zoom_offset += 1
            self.log(f"Font zoom increased (offset: +{self.font_zoom_offset})")
            self.update_all_fonts()
        
    def decrease_font_size(self, event=None):
        """Decrease font size."""
        if self.font_zoom_offset > self.min_zoom_offset:
            self.font_zoom_offset -= 1
            self.log(f"Font zoom decreased (offset: {self.font_zoom_offset:+d})")
            self.update_all_fonts()
            
    def reset_font_size(self, event=None):
        """Reset font size to default."""
        self.font_zoom_offset = 0
        self.log("Font zoom reset to default")
        self.update_all_fonts()
        
    def get_font(self, base_size=NORMAL_TEXT_SIZE, weight='normal'):
        """Get font tuple with specified base size plus zoom offset."""
        final_size = base_size + self.font_zoom_offset
        return (FONT_FAMILY, final_size, weight)
        
    def update_all_fonts(self):
        """Update fonts for all widgets."""
        try:
            # Update TTK styles first
            style = ttk.Style()
            style.configure('Large.TCheckbutton', font=self.get_font(NORMAL_TEXT_SIZE))
            style.configure('Large.TButton', font=self.get_font(NORMAL_TEXT_SIZE))
            style.configure('Large.TLabel', font=self.get_font(NORMAL_TEXT_SIZE))
            style.configure('TLabelFrame.Label', font=self.get_font(NORMAL_TEXT_SIZE))
            
            # Update title
            if hasattr(self, 'title_label'):
                self.title_label.configure(font=self.get_font(TITLE_FONT_SIZE, 'bold'))
            
            # Update drag/drop area
            if hasattr(self, 'drag_drop') and hasattr(self.drag_drop, 'drop_label'):
                self.drag_drop.drop_label.configure(font=self.get_font(LARGE_TEXT_SIZE))
            
            # Update specific widgets we know about
            widget_updates = [
                ('prompt_entry', self.get_font(NORMAL_TEXT_SIZE)),
                ('max_frames_entry', self.get_font(NORMAL_TEXT_SIZE)),
                ('log_text', self.get_font(SMALL_TEXT_SIZE)),
            ]
            
            for widget_name, font in widget_updates:
                if hasattr(self, widget_name):
                    try:
                        getattr(self, widget_name).configure(font=font)
                    except Exception as e:
                        print(f"Could not update font for {widget_name}: {e}")
            
            # Force a refresh of the entire window
            self.root.update_idletasks()
            
        except Exception as e:
            print(f"Error updating fonts: {e}")
            import traceback
            traceback.print_exc()
        
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Run the LEGACY tkinter GUI application (use gui_dpg.py instead!)."""
    print("⚠️  LEGACY GUI: Please use the new Dear PyGui version instead!")
    print("   Run: python gui_dpg.py")
    print("   This tkinter version is kept for reference only.\n")
    
    # Disable matplotlib GUI when running from GUI to prevent threading issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    
    # Disable debug visualizations in GUI mode to prevent threading issues
    import config
    config.DEBUG_SHOW_VISUALIZATIONS = False
    
    try:
        app = BlobtraceArtGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 
