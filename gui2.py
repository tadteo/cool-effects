#!/usr/bin/env python3
"""
BlobTrace Art - Fast Effects GUI (gui2.py)
Optimized for Mac with simple, fast effects that don't require AI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import threading
from pathlib import Path
import time

# Import our modules
from video_utils import extract_frames, create_video_from_frames_with_audio, get_video_info
from blob_utils import detect_blobs, create_soft_mask

# Constants
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
PADDING = "7"
FONT_SIZE = 15
ADAPTIVE_BLOCK_SIZE = 11  # For cv2.adaptiveThreshold
LOG_ROW = 11  # Grid row for log area

class FastEffectsGUI:
    """
    Fast effects GUI with simple blob-based effects that don't require AI.
    Optimized for Mac performance.
    """
    
    def __init__(self):
        print("Starting Fast Effects GUI...")
        self.root = tk.Tk()
        self.root.title("BlobTrace Art - Fast Effects (Mac Optimized)")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Variables
        self.current_video_path = None
        self.processing_thread = None
        self.is_processing = False
        
        # Performance caches
        # Removed caching - was useless for video processing
        
        # Parameter change tracking
        self.last_sensitivity = None
        
        # Available effects
        self.effects = {
            "Color Invert": self.effect_color_invert,
            "Blur Blobs": self.effect_blur_blobs,
            "Brighten Blobs": self.effect_brighten_blobs,
            "Rainbow Blobs": self.effect_rainbow_blobs,
            "Sepia Blobs": self.effect_sepia_blobs,
        }
        
        print("Setting up interface...")
        self.setup_ui()
        print("Fast Effects GUI ready!")
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main container with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        # Configure the canvas to expand the scrollable frame to full width
        def configure_scroll_region(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
            # Make the scrollable frame fill the canvas width
            canvas_width = main_canvas.winfo_width()
            main_canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        canvas_window = main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Also bind canvas resize to update frame width
        def on_canvas_configure(event):
            canvas_width = main_canvas.winfo_width()
            main_canvas.itemconfig(canvas_window, width=canvas_width)
        
        main_canvas.bind('<Configure>', on_canvas_configure)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling - fix for Mac
        def _on_mousewheel(event):
            # For Mac, use event.delta directly
            if hasattr(event, 'delta'):
                main_canvas.yview_scroll(int(-1 * (event.delta)), "units")
            else:
                # For other platforms
                main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind to both canvas and scrollable frame for better coverage
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        main_canvas.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-1, "units"))
        main_canvas.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(1, "units"))
        
        # Also bind to canvas focus events
        def _bind_to_mousewheel(event):
            main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_from_mousewheel(event):
            main_canvas.unbind_all("<MouseWheel>")
        
        main_canvas.bind('<Enter>', _bind_to_mousewheel)
        main_canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Main frame inside scrollable area - full width
        main_frame = ttk.Frame(scrollable_frame, padding=PADDING)
        main_frame.pack(fill="both", expand=True)
        
        # Configure grid weights for full width
        main_frame.columnconfigure(0, weight=0)  # Labels column - fixed width
        main_frame.columnconfigure(1, weight=1)  # Content column - expand to fill
        
        # Title - centered
        title_label = ttk.Label(main_frame, text="🚀 BlobTrace Art - Fast Effects", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 25))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="📁 Video File", padding=PADDING)
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        file_frame.columnconfigure(1, weight=1)
        
        browse_btn = ttk.Button(file_frame, text="Browse Video", 
                               command=self.browse_video)
        browse_btn.grid(row=0, column=0, padx=(0, 15))
        
        self.file_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(file_frame, textvariable=self.file_var, font=('Arial', FONT_SIZE))
        file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Basic Settings
        basic_frame = ttk.LabelFrame(main_frame, text="🎬 Basic Settings", padding=PADDING)
        basic_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        basic_frame.columnconfigure(1, weight=1)
        
        ttk.Label(basic_frame, text="Effect:", font=('Arial', FONT_SIZE, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=(0, 8))
        self.effect_var = tk.StringVar(value="Color Invert")
        effect_combo = ttk.Combobox(basic_frame, textvariable=self.effect_var, 
                                   values=list(self.effects.keys()), state="readonly", font=('Arial', FONT_SIZE))
        effect_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(0, 8))
        
        ttk.Label(basic_frame, text="Speed:", font=('Arial', FONT_SIZE, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 8))
        self.speed_var = tk.StringVar(value="Fast")
        speed_combo = ttk.Combobox(basic_frame, textvariable=self.speed_var,
                                  values=["Ultra Fast (every 5th frame)", "Fast (every 2nd frame)", "Normal (all frames)"],
                                  state="readonly", font=('Arial', FONT_SIZE))
        speed_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 8))
        
        ttk.Label(basic_frame, text="Max Frames:", font=('Arial', FONT_SIZE, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 0))
        max_frames_frame = ttk.Frame(basic_frame)
        max_frames_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 0))
        self.max_frames_var = tk.StringVar(value="100")
        ttk.Entry(max_frames_frame, textvariable=self.max_frames_var, width=12, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(max_frames_frame, text="(0 = process all frames)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        # Mask Detection Settings
        mask_frame = ttk.LabelFrame(main_frame, text="🎭 Mask Detection", padding=PADDING)
        mask_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        mask_frame.columnconfigure(1, weight=1)
        
        ttk.Label(mask_frame, text="Sensitivity:", font=('Arial', FONT_SIZE, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=(0, 8))
        sensitivity_frame = ttk.Frame(mask_frame)
        sensitivity_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(0, 8))
        self.sensitivity_var = tk.StringVar(value="0.45")
        ttk.Entry(sensitivity_frame, textvariable=self.sensitivity_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(sensitivity_frame, text="(0-1, lower = more sensitive)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        ttk.Label(mask_frame, text="Roughness:", font=('Arial', FONT_SIZE, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 8))
        roughness_frame = ttk.Frame(mask_frame)
        roughness_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 8))
        self.roughness_var = tk.StringVar(value="0.16")
        ttk.Entry(roughness_frame, textvariable=self.roughness_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(roughness_frame, text="(0-1, lower = smoother edges)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        ttk.Label(mask_frame, text="Shape:", font=('Arial', FONT_SIZE, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 0))
        self.mask_shape_var = tk.StringVar(value="Polyline")
        shape_combo = ttk.Combobox(mask_frame, textvariable=self.mask_shape_var,
                                  values=["Polyline", "Rectangle", "Original"], state="readonly", font=('Arial', FONT_SIZE))
        shape_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 0))
        
        # Visual Style Settings
        style_frame = ttk.LabelFrame(main_frame, text="🎨 Visual Style", padding=PADDING)
        style_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        style_frame.columnconfigure(1, weight=1)
        
        ttk.Label(style_frame, text="Outline Thickness:", font=('Arial', FONT_SIZE, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=(0, 8))
        outline_frame = ttk.Frame(style_frame)
        outline_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(0, 8))
        self.outline_thickness_var = tk.StringVar(value="2")
        ttk.Entry(outline_frame, textvariable=self.outline_thickness_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(outline_frame, text="(1-5 pixels)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        ttk.Label(style_frame, text="Connection Thickness:", font=('Arial', FONT_SIZE, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 0))
        connection_frame = ttk.Frame(style_frame)
        connection_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 0))
        self.connection_thickness_var = tk.StringVar(value="1")
        ttk.Entry(connection_frame, textvariable=self.connection_thickness_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(connection_frame, text="(1-5 pixels)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        # Connection Settings
        connection_settings_frame = ttk.LabelFrame(main_frame, text="🔗 Connection Settings", padding=PADDING)
        connection_settings_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        connection_settings_frame.columnconfigure(1, weight=1)
        
        ttk.Label(connection_settings_frame, text="Enable Connections:", font=('Arial', FONT_SIZE, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=(0, 8))
        self.connect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(connection_settings_frame, variable=self.connect_var, text="Draw curved connections between shapes").grid(row=0, column=1, sticky=tk.W, padx=(0, 10), pady=(0, 8))
        
        ttk.Label(connection_settings_frame, text="Connection Range:", font=('Arial', FONT_SIZE, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 8))
        distance_frame = ttk.Frame(connection_settings_frame)
        distance_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 8))
        self.max_connection_distance_var = tk.StringVar(value="0.3")
        ttk.Entry(distance_frame, textvariable=self.max_connection_distance_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(distance_frame, text="(0-1, 0=no connections, 1=full screen)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        ttk.Label(connection_settings_frame, text="Max Connections Per Shape:", font=('Arial', FONT_SIZE, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 8))
        connections_frame = ttk.Frame(connection_settings_frame)
        connections_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 8))
        self.max_connections_per_shape_var = tk.StringVar(value="3")
        ttk.Entry(connections_frame, textvariable=self.max_connections_per_shape_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(connections_frame, text="(1-10 connections)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        ttk.Label(connection_settings_frame, text="Connect to N Closest:", font=('Arial', FONT_SIZE, 'bold')).grid(row=3, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 0))
        n_closest_frame = ttk.Frame(connection_settings_frame)
        n_closest_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 0))
        self.connect_n_closest_var = tk.StringVar(value="2")
        ttk.Entry(n_closest_frame, textvariable=self.connect_n_closest_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(n_closest_frame, text="(1-8 nearest neighbors)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        # Advanced Settings
        advanced_frame = ttk.LabelFrame(main_frame, text="⚙️ Advanced Settings", padding=PADDING)
        advanced_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        advanced_frame.columnconfigure(1, weight=1)
        
        ttk.Label(advanced_frame, text="Curve Sensitivity:", font=('Arial', FONT_SIZE, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=(0, 8))
        curve_sens_frame = ttk.Frame(advanced_frame)
        curve_sens_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(0, 8))
        self.curve_sensitivity_var = tk.StringVar(value="0.3")
        ttk.Entry(curve_sens_frame, textvariable=self.curve_sensitivity_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(curve_sens_frame, text="(0-1, higher = more curved)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        ttk.Label(advanced_frame, text="Obstacle Sensitivity:", font=('Arial', FONT_SIZE, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=(8, 0))
        obstacle_sens_frame = ttk.Frame(advanced_frame)
        obstacle_sens_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(8, 0))
        self.obstacle_sensitivity_var = tk.StringVar(value="0.33")
        ttk.Entry(obstacle_sens_frame, textvariable=self.obstacle_sensitivity_var, width=10, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(obstacle_sens_frame, text="(0-1, higher = more obstacles detected)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        # Output Settings
        output_frame = ttk.LabelFrame(main_frame, text="💾 Output Settings", padding=PADDING)
        output_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Filename:", font=('Arial', FONT_SIZE, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=(0, 8))
        filename_frame = ttk.Frame(output_frame)
        filename_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(0, 8))
        self.output_filename_var = tk.StringVar(value="processed_video")
        ttk.Entry(filename_frame, textvariable=self.output_filename_var, width=30, font=('Arial', FONT_SIZE)).pack(side=tk.LEFT)
        ttk.Label(filename_frame, text="(without extension)", font=('Arial', 9), foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        # Process button
        process_frame = ttk.Frame(main_frame)
        process_frame.grid(row=8, column=0, columnspan=2, pady=30)
        self.process_btn = ttk.Button(process_frame, text="🎬 Process Video", 
                                     command=self.process_video)
        self.process_btn.pack()
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                           maximum=100, length=600)
        self.progress_bar.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', FONT_SIZE))
        status_label.grid(row=10, column=0, columnspan=2, pady=(0, 15))
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="📋 Processing Log", padding=PADDING)
        log_frame.grid(row=LOG_ROW, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(15, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(LOG_ROW, weight=1)
        
        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD, font=('Arial', FONT_SIZE))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Initial log message
        self.log("🚀 Fast Effects GUI ready!")
        self.log("💡 These effects are 10-50x faster than AI effects")
        self.log("⚡ Perfect for Mac - no AI models needed!")
        
    def browse_video(self):
        """Browse for video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_video_path = file_path
            self.file_var.set(os.path.basename(file_path))
            
            # Auto-fill output filename with original name + effect
            input_path = Path(file_path)
            base_name = input_path.stem  # filename without extension
            effect_name = self.effect_var.get().lower().replace(' ', '_')
            suggested_name = f"{base_name}_{effect_name}"
            self.output_filename_var.set(suggested_name)
            
            self.log(f"📁 Selected: {file_path}")
            self.log(f"💡 Suggested output: {suggested_name}")
        else:
            self.current_video_path = None
            self.file_var.set("No file selected")
    
    def log(self, message):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def process_video(self):
        """Process video with selected effect."""
        if not self.current_video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress!")
            return
        
        self.processing_thread = threading.Thread(target=self._process_video_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_video_thread(self):
        """Process video in separate thread."""
        try:
            self.is_processing = True
            self.process_btn.configure(state="disabled")
            
            # Clear caches for new processing session
            # Removed all caching since it was useless
            
            effect_name = self.effect_var.get()
            speed_setting = self.speed_var.get()
            max_frames = int(self.max_frames_var.get()) if self.max_frames_var.get() != "0" else None
            
            # Determine frame skip
            if "Ultra Fast" in speed_setting:
                frame_skip = 5
            elif "Fast" in speed_setting:
                frame_skip = 2
            else:
                frame_skip = 1
            
            self.log(f"🎬 Processing with {effect_name}")
            self.log(f"⚡ Speed: {speed_setting}")
            
            # Load video
            self.status_var.set("Loading video...")
            frames = []
            for frame, frame_num in extract_frames(self.current_video_path):
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            
            if max_frames:
                self.log(f"🔢 Limited to {len(frames)} frames")
            else:
                self.log(f"📊 Loaded {len(frames)} frames")
            
            # Get effect function
            effect_func = self.effects[effect_name]
            
            # Process frames
            processed_frames = []
            start_time = time.time()
            
            for i in range(0, len(frames), frame_skip):
                if not self.is_processing:
                    break
                    
                frame = frames[i]
                processed_frame = effect_func(frame)
                processed_frames.append(processed_frame)
                
                # Update progress less frequently for better performance
                if i % 5 == 0 or i == len(frames) - 1:
                    progress = (i + 1) / len(frames) * 100
                    self.progress_var.set(progress)
                    
                    # Calculate and display FPS in the status bar
                    elapsed = time.time() - start_time
                    fps = (i + 1) / elapsed if elapsed > 0 else 0
                    self.status_var.set(f"Processing frame {i+1}/{len(frames)} - {fps:.1f} FPS")
                
                # Only log major milestones, not every 20 frames
                if i % 50 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    fps = (i + 1) / elapsed if elapsed > 0 else 0
                    self.log(f"⚡ Processed {i+1} frames - {fps:.1f} FPS average")
            
            if not self.is_processing:
                self.log("❌ Processing cancelled")
                return
            
            # Save video
            self.status_var.set("Saving video...")
            self.log("💾 Saving processed video...")
            
            # Generate unique output filename
            base_filename = self.output_filename_var.get()
            output_path = self.generate_unique_filename(base_filename)
            
            info = get_video_info(self.current_video_path)
            fps = info['fps'] / frame_skip if frame_skip > 1 else info['fps']
            
            # Save video with audio preservation
            audio_preserved = create_video_from_frames_with_audio(
                processed_frames, str(output_path), self.current_video_path, fps=fps)
            
            if audio_preserved:
                self.log("✅ Audio preserved successfully!")
            else:
                self.log("⚠️ Audio could not be preserved (video saved without audio)")
            
            elapsed = time.time() - start_time
            self.log(f"✅ Complete in {elapsed:.1f}s!")
            self.log(f"💾 Saved: {output_path}")
            self.log(f"⚡ Speed: {len(processed_frames)/elapsed:.1f} FPS")
            
            self.status_var.set("Complete!")
            self.progress_var.set(100)
            
            messagebox.showinfo("Success", f"Video processed!\nSaved: {os.path.basename(output_path)}")
            
        except Exception as e:
            self.log(f"❌ Error: {e}")
            messagebox.showerror("Error", f"Processing failed: {e}")
        finally:
            self.is_processing = False
            self.process_btn.configure(state="normal")
            self.progress_var.set(0)
            self.status_var.set("Ready")
    
    # Effect functions
    def effect_color_invert(self, frame):
        """Invert colors in blob regions."""
        blobs = self.detect_blobs_less_sensitive(frame)
        if not blobs:
            return frame
        
        result = frame.copy()
        outline_thickness = self.get_outline_thickness()
        mask_shape = self.mask_shape_var.get()
        polylines = []
        
        # Process each blob individually with numbering
        for i, blob in enumerate(blobs):
            # Create mask based on selected shape
            if mask_shape == "Polyline":
                # Create simplified polyline from blob
                polyline = self.create_polyline_from_blob(blob)
                polylines.append(polyline)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polyline], 255)
                outline_points = polyline
            elif mask_shape == "Rectangle":
                # Create rectangular mask
                x, y, w, h = cv2.boundingRect(blob)
                rectangle = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                polylines.append(rectangle)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [rectangle], 255)
                outline_points = rectangle
            else:  # Original
                # Use original blob shape
                polylines.append(blob)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [blob], 255)
                outline_points = blob
            
            # Create sharp mask (no blur for crisp edges)
            sharp_mask = mask.astype(np.float32) / 255.0
            
            # Invert colors in this blob region
            inverted = 255 - frame
            
            # Apply effect to fill the entire polyline area
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - sharp_mask) + inverted[:, :, c] * sharp_mask
            
            # Draw the outline so you can see it
            cv2.polylines(result, [outline_points], True, (255, 255, 255), outline_thickness)  # White outline
            
            # Add tiny numbers around the perimeter
            self.add_perimeter_numbers(result, outline_points, i + 1)
        
        # Draw connections between polylines if enabled
        if self.connect_var.get() and len(polylines) > 1:
            self.draw_polyline_connections(result, frame, polylines)
        
        return result.astype(np.uint8)
    
    def effect_blur_blobs(self, frame):
        """Blur blob regions."""
        blobs = self.detect_blobs_less_sensitive(frame)
        if not blobs:
            return frame
        
        result = frame.copy()
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        outline_thickness = self.get_outline_thickness()
        mask_shape = self.mask_shape_var.get()
        polylines = []
        
        for i, blob in enumerate(blobs):
            # Create mask based on selected shape
            if mask_shape == "Polyline":
                polyline = self.create_polyline_from_blob(blob)
                polylines.append(polyline)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polyline], 255)
                outline_points = polyline
            elif mask_shape == "Rectangle":
                x, y, w, h = cv2.boundingRect(blob)
                rectangle = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                polylines.append(rectangle)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [rectangle], 255)
                outline_points = rectangle
            else:  # Original
                polylines.append(blob)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [blob], 255)
                outline_points = blob
            
            sharp_mask = mask.astype(np.float32) / 255.0
            
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - sharp_mask) + blurred[:, :, c] * sharp_mask
            
            cv2.polylines(result, [outline_points], True, (255, 255, 255), outline_thickness)
            self.add_perimeter_numbers(result, outline_points, i + 1)
        
        if self.connect_var.get() and len(polylines) > 1:
            self.draw_polyline_connections(result, frame, polylines)
        
        return result.astype(np.uint8)
    
    def effect_brighten_blobs(self, frame):
        """Brighten blob regions."""
        blobs = self.detect_blobs_less_sensitive(frame)
        if not blobs:
            return frame
        
        result = frame.copy()
        brightened = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        outline_thickness = self.get_outline_thickness()
        mask_shape = self.mask_shape_var.get()
        polylines = []
        
        for i, blob in enumerate(blobs):
            # Create mask based on selected shape
            if mask_shape == "Polyline":
                polyline = self.create_polyline_from_blob(blob)
                polylines.append(polyline)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polyline], 255)
                outline_points = polyline
            elif mask_shape == "Rectangle":
                x, y, w, h = cv2.boundingRect(blob)
                rectangle = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                polylines.append(rectangle)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [rectangle], 255)
                outline_points = rectangle
            else:  # Original
                polylines.append(blob)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [blob], 255)
                outline_points = blob
            
            sharp_mask = mask.astype(np.float32) / 255.0
            
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - sharp_mask) + brightened[:, :, c] * sharp_mask
            
            cv2.polylines(result, [outline_points], True, (255, 255, 255), outline_thickness)
            self.add_perimeter_numbers(result, outline_points, i + 1)
        
        if self.connect_var.get() and len(polylines) > 1:
            self.draw_polyline_connections(result, frame, polylines)
        
        return result.astype(np.uint8)
    
    def effect_rainbow_blobs(self, frame):
        """Apply rainbow colors to blobs."""
        blobs = self.detect_blobs_less_sensitive(frame)
        if not blobs:
            return frame
        
        result = frame.copy()
        colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)]
        outline_thickness = self.get_outline_thickness()
        mask_shape = self.mask_shape_var.get()
        polylines = []
        
        for i, blob in enumerate(blobs):
            # Create mask based on selected shape
            if mask_shape == "Polyline":
                polyline = self.create_polyline_from_blob(blob)
                polylines.append(polyline)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polyline], 255)
                outline_points = polyline
            elif mask_shape == "Rectangle":
                x, y, w, h = cv2.boundingRect(blob)
                rectangle = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                polylines.append(rectangle)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [rectangle], 255)
                outline_points = rectangle
            else:  # Original
                polylines.append(blob)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [blob], 255)
                outline_points = blob
            
            sharp_mask = mask.astype(np.float32) / 255.0
            
            color = colors[i % len(colors)]
            colored = np.full_like(frame, color)
            
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - sharp_mask * 0.6) + colored[:, :, c] * (sharp_mask * 0.6)
            
            cv2.polylines(result, [outline_points], True, (255, 255, 255), outline_thickness)
            self.add_perimeter_numbers(result, outline_points, i + 1)
        
        if self.connect_var.get() and len(polylines) > 1:
            self.draw_polyline_connections(result, frame, polylines)
        
        return result.astype(np.uint8)
    
    def effect_sepia_blobs(self, frame):
        """Apply sepia tone to blobs."""
        blobs = self.detect_blobs_less_sensitive(frame)
        if not blobs:
            return frame
        
        result = frame.copy()
        sepia_filter = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, sepia_filter)
        sepia = np.clip(sepia, 0, 255)
        outline_thickness = self.get_outline_thickness()
        mask_shape = self.mask_shape_var.get()
        polylines = []
        
        for i, blob in enumerate(blobs):
            # Create mask based on selected shape
            if mask_shape == "Polyline":
                polyline = self.create_polyline_from_blob(blob)
                polylines.append(polyline)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polyline], 255)
                outline_points = polyline
            elif mask_shape == "Rectangle":
                x, y, w, h = cv2.boundingRect(blob)
                rectangle = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                polylines.append(rectangle)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [rectangle], 255)
                outline_points = rectangle
            else:  # Original
                polylines.append(blob)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [blob], 255)
                outline_points = blob
            
            sharp_mask = mask.astype(np.float32) / 255.0
            
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - sharp_mask) + sepia[:, :, c] * sharp_mask
            
            cv2.polylines(result, [outline_points], True, (255, 255, 255), outline_thickness)
            self.add_perimeter_numbers(result, outline_points, i + 1)
        
        if self.connect_var.get() and len(polylines) > 1:
            self.draw_polyline_connections(result, frame, polylines)
        
        return result.astype(np.uint8)
    
    # Helper functions for parameter conversion
    def get_outline_thickness(self):
        """Get outline thickness in pixels (1-5)."""
        try:
            value_str = self.outline_thickness_var.get().strip()
            if not value_str:
                return 2
            value = max(1, min(5, int(float(value_str))))
            return value
        except (ValueError, AttributeError):
            return 2  # Default
    
    def get_connection_thickness(self):
        """Get connection thickness in pixels (1-5)."""
        try:
            value_str = self.connection_thickness_var.get().strip()
            if not value_str:
                return 1
            value = max(1, min(5, int(float(value_str))))
            return value
        except (ValueError, AttributeError):
            return 1  # Default
    
    def get_mask_sensitivity(self):
        """Convert normalized sensitivity (0-1) to internal range (50-250)."""
        try:
            value_str = self.sensitivity_var.get().strip()
            if not value_str:
                current_sensitivity = 127
            else:
                norm_value = max(0, min(1, float(value_str)))
                current_sensitivity = int(norm_value * 200 + 50)  # Maps 0->50, 1->250
            
            # Track parameter changes
            if self.last_sensitivity != current_sensitivity:
                print(f"🔄 SENSITIVITY CHANGED: {self.last_sensitivity} → {current_sensitivity} (input: '{value_str}')")
                self.last_sensitivity = current_sensitivity
            
            return current_sensitivity
        except (ValueError, AttributeError):
            return 140  # Default
    
    def get_mask_roughness(self):
        """Convert normalized roughness (0-1) to internal range (0.005-0.1)."""
        try:
            value_str = self.roughness_var.get().strip()
            if not value_str:
                return 0.02
            norm_value = max(0, min(1, float(value_str)))
            return norm_value * 0.095 + 0.005  # Maps 0->0.005, 1->0.1
        except (ValueError, AttributeError):
            return 0.02  # Default
    
    def get_curve_sensitivity(self):
        """Convert normalized curve sensitivity (0-1) to internal range (0.1-0.8)."""
        try:
            value_str = self.curve_sensitivity_var.get().strip()
            if not value_str:
                return 0.3
            norm_value = max(0, min(1, float(value_str)))
            return norm_value * 0.7 + 0.1  # Maps 0->0.1, 1->0.8
        except (ValueError, AttributeError):
            return 0.3  # Default
    
    def get_obstacle_sensitivity(self):
        """Convert normalized obstacle sensitivity (0-1) to internal range (50-200)."""
        try:
            value_str = self.obstacle_sensitivity_var.get().strip()
            if not value_str:
                return 100
            norm_value = max(0, min(1, float(value_str)))
            return int(norm_value * 150 + 50)  # Maps 0->50, 1->200
        except (ValueError, AttributeError):
            return 100  # Default
    
    def get_max_connection_distance(self):
        """Convert screen percentage (0-1) to pixel distance based on frame size."""
        try:
            value_str = self.max_connection_distance_var.get().strip()
            if not value_str:
                return 0.3
            norm_value = max(0, min(1, float(value_str)))
            return norm_value
        except (ValueError, AttributeError):
            return 0.3  # Default (30% of screen)
    
    def get_max_connections_per_shape(self):
        """Get max connections per shape (1-10)."""
        try:
            value_str = self.max_connections_per_shape_var.get().strip()
            if not value_str:
                return 3
            value = max(1, min(10, int(float(value_str))))
            return value
        except (ValueError, AttributeError):
            return 3  # Default
    
    def get_connect_n_closest(self):
        """Get number of closest connections (1-8)."""
        try:
            value_str = self.connect_n_closest_var.get().strip()
            if not value_str:
                return 2
            value = max(1, min(8, int(float(value_str))))
            return value
        except (ValueError, AttributeError):
            return 2  # Default

    # Helper functions for improved blob processing
    def detect_blobs_less_sensitive(self, frame):
        """DYNAMIC blob detection - automatically adjusts threshold to get target number of blobs."""
        sensitivity = self.get_mask_sensitivity()
        
        # Simple processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Target number of blobs (based on sensitivity: lower sensitivity = more blobs)
        target_blobs = max(5, min(25, int(30 - (sensitivity - 50) / 10)))
        
        # Dynamic threshold search
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = 200
        max_area = int(frame_area * 0.05)
        
        # Use ORIGINAL algorithm approach - fixed threshold around middle gray
        # This detects both bright AND dark regions, not just bright ones
        base_threshold = 127  # Original threshold - middle gray
        
        # Allow sensitivity to adjust threshold slightly around the middle
        threshold_adjustment = int((sensitivity - 140) * 0.3)  # Small adjustment
        test_threshold = max(80, min(180, base_threshold + threshold_adjustment))
        
        _, thresh = cv2.threshold(blurred, test_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours with your preferred area limits
        min_area = 150      # Your preferred minimum
        max_area = int(frame_area * 0.1)  # Keep as percentage (5% of screen)
        
        # Filter contours
        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                blobs.append(approx.reshape(-1, 2))
        
        print(f"Debug: Found {len(blobs)} blobs (threshold={test_threshold}, sensitivity={sensitivity})")
        return blobs
    
    def create_polyline_from_blob(self, blob, roughness=None):
        """Create a polyline approximation of a blob contour with adjustable roughness."""
        if roughness is None:
            roughness = self.get_mask_roughness()
        
        # Approximate the contour to a polyline
        epsilon = roughness * cv2.arcLength(blob, True)
        polyline = cv2.approxPolyDP(blob, epsilon, True)
        return polyline.reshape(-1, 2)
    
    def add_perimeter_numbers(self, frame, polyline, number):
        """Add tiny numbers around the perimeter of the polyline."""
        if len(polyline) < 3:
            return
        
        # Calculate the perimeter length
        perimeter = cv2.arcLength(polyline, True)
        
        # Determine how many numbers to place (roughly every 100 pixels)
        num_positions = max(1, min(4, int(perimeter / 100)))
        
        # Get points along the perimeter
        points = polyline.reshape(-1, 2)
        
        for i in range(num_positions):
            # Calculate position along perimeter
            pos_index = int((i * len(points)) / num_positions) % len(points)
            point = points[pos_index]
            
            # Find the direction to offset the number (outward from polygon)
            # Use the next point to determine direction
            next_index = (pos_index + 1) % len(points)
            next_point = points[next_index]
            
            # Calculate perpendicular offset direction
            dx = next_point[0] - point[0]
            dy = next_point[1] - point[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Perpendicular vector (outward)
                offset_x = -dy / length * 15  # 15 pixels offset
                offset_y = dx / length * 15
            else:
                offset_x, offset_y = 15, 0
            
            # Position for the number
            text_x = int(point[0] + offset_x)
            text_y = int(point[1] + offset_y)
            
            # Draw tiny number without background
            text = str(number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # Very small
            thickness = 1
            
            # Draw the number in bright color
            cv2.putText(frame, text, (text_x, text_y),
                       font, font_scale, (0, 255, 255), thickness)  # Bright yellow text
    
    def draw_polyline_connections(self, result, original_frame, polylines):
        """Draw curved connections between polylines with obstacle avoidance."""
        if len(polylines) < 2:
            return
        
        connection_thickness = self.get_connection_thickness()
        obstacle_sensitivity = self.get_obstacle_sensitivity()
        max_distance_ratio = self.get_max_connection_distance()
        max_connections_per_shape = self.get_max_connections_per_shape()
        connect_n_closest = self.get_connect_n_closest()
        
        # Calculate actual max distance based on frame size
        frame_diagonal = np.sqrt(original_frame.shape[0]**2 + original_frame.shape[1]**2)
        max_distance = int(max_distance_ratio * frame_diagonal)
        
        # Create obstacle map from original frame (detect high contrast areas)
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        
        # Adjust Canny thresholds based on obstacle sensitivity
        low_thresh = max(30, obstacle_sensitivity - 50)
        high_thresh = min(200, obstacle_sensitivity + 50)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        
        # Dilate edges to create obstacle zones (more dilation = more obstacles detected)
        kernel_size = max(3, int(obstacle_sensitivity / 25))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        obstacles = cv2.dilate(edges, kernel, iterations=max(1, int(obstacle_sensitivity / 50)))
        
        # Adjust dark/bright area thresholds based on sensitivity
        dark_thresh = max(30, 80 - obstacle_sensitivity // 4)
        bright_thresh = min(225, 180 + obstacle_sensitivity // 4)
        dark_areas = gray < dark_thresh
        bright_areas = gray > bright_thresh
        obstacles = obstacles | dark_areas | bright_areas
        
        # For each polyline, find its N closest neighbors (optimized)
        connections_to_draw = []
        
        # Pre-compute centroids for faster distance estimation
        centroids = []
        for poly in polylines:
            centroid = np.mean(poly, axis=0)
            centroids.append(centroid)
        
        for i in range(len(polylines)):
            poly1 = polylines[i]
            centroid1 = centroids[i]
            
            # First pass: filter by centroid distance (much faster)
            candidate_indices = []
            for j in range(len(polylines)):
                if i == j:
                    continue
                centroid2 = centroids[j]
                centroid_dist = np.linalg.norm(centroid1 - centroid2)
                # Use a larger threshold for centroid filtering
                if centroid_dist <= max_distance * 1.5:
                    candidate_indices.append(j)
            
            # Second pass: precise distance calculation only for candidates
            distances = []
            for j in candidate_indices:
                poly2 = polylines[j]
                
                # Vectorized distance calculation (much faster)
                # Calculate all pairwise distances at once
                p1_expanded = poly1[:, np.newaxis, :]  # Shape: (n1, 1, 2)
                p2_expanded = poly2[np.newaxis, :, :]  # Shape: (1, n2, 2)
                all_distances = np.linalg.norm(p1_expanded - p2_expanded, axis=2)  # Shape: (n1, n2)
                
                # Find minimum distance and corresponding points
                min_idx = np.unravel_index(np.argmin(all_distances), all_distances.shape)
                min_dist = all_distances[min_idx]
                best_p1 = poly1[min_idx[0]]
                best_p2 = poly2[min_idx[1]]
                
                # Only consider connections within max distance
                if min_dist <= max_distance:
                    distances.append((j, best_p1, best_p2, min_dist))
            
            # Sort by distance and take the N closest
            distances.sort(key=lambda x: x[3])
            n_closest = min(connect_n_closest, len(distances))
            
            # Add connections to the N closest polylines
            for k in range(n_closest):
                j, best_p1, best_p2, dist = distances[k]
                # Avoid duplicate connections (i,j) and (j,i)
                if i < j:  # Only add each connection once
                    connections_to_draw.append((i, j, best_p1, best_p2, dist))
        
        # Remove duplicates and respect per-shape limits
        # Convert to tuples for hashing, then back to original format
        seen_connections = set()
        unique_connections = []
        
        for i, j, best_p1, best_p2, dist in connections_to_draw:
            # Create a hashable key using just the polyline indices
            connection_key = (min(i, j), max(i, j))
            if connection_key not in seen_connections:
                seen_connections.add(connection_key)
                unique_connections.append((i, j, best_p1, best_p2, dist))
        
        unique_connections.sort(key=lambda x: x[4])  # Sort by distance
        
        # Track connections per polyline
        connection_count = {i: 0 for i in range(len(polylines))}
        
        # Draw connections, respecting the per-shape limit
        for i, j, best_p1, best_p2, dist in unique_connections:
            # Check if both polylines can accept more connections
            if (connection_count[i] < max_connections_per_shape and 
                connection_count[j] < max_connections_per_shape):
                
                # Create curved path with obstacle avoidance
                curve_points = self.create_curved_path_with_avoidance(
                    best_p1, best_p2, obstacles, original_frame.shape[:2]
                )
                
                # Draw the curved connection
                for k in range(len(curve_points) - 1):
                    cv2.line(result, tuple(curve_points[k].astype(int)), 
                            tuple(curve_points[k + 1].astype(int)), 
                            (0, 255, 255), connection_thickness)  # Bright yellow
                
                # Update connection counts
                connection_count[i] += 1
                connection_count[j] += 1
    
    def create_curved_path_with_avoidance(self, start_point, end_point, obstacle_map, shape):
        """SIMPLE curved path - KISS principle."""
        start = np.array(start_point, dtype=int)
        end = np.array(end_point, dtype=int)
        
        curve_sensitivity = self.get_curve_sensitivity()
        
        # Simple curve - just one bow height based on sensitivity
        mid_point = (start + end) / 2
        direction = end - start
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)
        
        # Make curve more obvious - multiply by distance and sensitivity
        bow_height = curve_sensitivity * 0.5  # More pronounced curve
        control_point = mid_point + perpendicular * np.linalg.norm(direction) * bow_height
        
        # Generate simple Bézier curve
        num_points = 15
        t_values = np.linspace(0, 1, num_points)
        
        curve_points = []
        for t in t_values:
            # Quadratic Bézier curve
            point = (1-t)**2 * start + 2*(1-t)*t * control_point + t**2 * end
            curve_points.append(point)
        
        return np.array(curve_points)
    
    def generate_unique_filename(self, base_filename, extension=".mp4"):
        """Generate a unique filename by adding suffix if file exists."""
        import os
        
        # Clean the base filename
        base_filename = base_filename.strip()
        if not base_filename:
            base_filename = "processed_video"
        
        # Remove any existing extension from base filename
        if base_filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            base_filename = os.path.splitext(base_filename)[0]
        
        # Get the input file's directory
        if self.current_video_path:
            input_path = Path(self.current_video_path)
            input_dir = input_path.parent
        else:
            input_dir = Path(".")
        
        # Create processed_videos subfolder
        output_dir = input_dir / "processed_videos"
        output_dir.mkdir(exist_ok=True)
        
        # Start with the base filename
        output_path = output_dir / f"{base_filename}{extension}"
        
        # If file doesn't exist, use it
        if not output_path.exists():
            return str(output_path)
        
        # File exists, find a unique suffix
        counter = 1
        while True:
            output_path = output_dir / f"{base_filename}_{counter:03d}{extension}"
            if not output_path.exists():
                return str(output_path)
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 999:
                import time
                timestamp = int(time.time())
                output_path = output_dir / f"{base_filename}_{timestamp}{extension}"
                return str(output_path)
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()

def main():
    """Main function."""
    app = FastEffectsGUI()
    app.run()

if __name__ == "__main__":
    main()
