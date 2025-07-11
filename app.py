#!/usr/bin/env python3
"""
Cinematic Video Generator Backend - Render Optimized

A complete Python backend that generates cinematic storytelling videos from text scripts
using Stable Diffusion for image generation and MoviePy for video creation.
Optimized for deployment on Render.com

RENDER DEPLOYMENT INSTRUCTIONS:
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set Build Command: pip install -r requirements.txt
4. Set Start Command: python app.py
5. Set Environment Variables (see below)
6. Deploy!

ENVIRONMENT VARIABLES FOR RENDER:
- PYTHON_VERSION=3.9.16
- MODEL_ID=runwayml/stable-diffusion-v1-5 (optional)
- MAX_WORKERS=1 (to manage memory)
- RENDER_ENV=production
"""

import os
import re
import json
import time
import random
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import gc

# Flask for API
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Image generation
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import PIL.Image as Image
import numpy as np

# Video creation
from moviepy.editor import (
    ImageClip, CompositeVideoClip, VideoFileClip, 
    concatenate_videoclips, ColorClip, CompositeVideoClip
)
from moviepy.video.fx import resize, fadeout, fadein

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Render-specific optimizations
RENDER_ENV = os.environ.get('RENDER_ENV', 'development')
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '1'))

class CinematicVideoGenerator:
    """Main class for generating cinematic videos from text scripts - Render optimized."""
    
    def __init__(self, model_id: str = None):
        """
        Initialize the video generator.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
        """
        self.model_id = model_id or os.environ.get('MODEL_ID', "runwayml/stable-diffusion-v1-5")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        
        # Video settings - optimized for Render
        self.video_width = 1080
        self.video_height = 1920
        self.fps = 24  # Reduced for better performance
        self.scene_duration = 3.0  # Reduced duration
        self.transition_duration = 0.3
        self.max_scenes = 10  # Limit scenes to prevent memory issues
        
        # Use temporary directories for Render
        self.temp_dir = tempfile.mkdtemp()
        self.scenes_dir = Path(self.temp_dir) / "scenes"
        self.output_dir = Path(self.temp_dir) / "output"
        self.scenes_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline with memory optimization
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize Stable Diffusion pipeline with Render optimizations."""
        try:
            logger.info(f"Initializing Stable Diffusion pipeline on {self.device}")
            
            # Use lower precision for memory efficiency
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Aggressive memory optimization for Render
            self.pipeline.enable_attention_slicing()
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_xformers_memory_efficient_attention()
            else:
                # CPU optimizations
                self.pipeline.enable_model_cpu_offload()
            
            logger.info("Stable Diffusion pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            # Fallback to CPU if CUDA fails
            if self.device == "cuda":
                self.device = "cpu"
                self._initialize_pipeline()
            else:
                raise
    
    def parse_script(self, script_text: str) -> List[str]:
        """
        Parse script text into individual scenes with limits for Render.
        
        Args:
            script_text: Raw script text
            
        Returns:
            List of scene descriptions (limited to max_scenes)
        """
        # Clean the script
        script_text = script_text.strip()
        
        # Split by common scene separators
        scenes = []
        
        # Method 1: Split by double newlines
        potential_scenes = re.split(r'\n\s*\n', script_text)
        
        # Method 2: Split by scene markers
        if len(potential_scenes) == 1:
            potential_scenes = re.split(r'(?i)(?:scene|chapter|part)\s*\d+', script_text)
        
        # Method 3: Split by sentences if still one block
        if len(potential_scenes) == 1:
            sentences = re.split(r'(?<=[.!?])\s+', script_text)
            # Group sentences into scenes (2-3 sentences per scene)
            potential_scenes = []
            for i in range(0, len(sentences), 2):
                scene = ' '.join(sentences[i:i+2])
                if scene.strip():
                    potential_scenes.append(scene)
        
        # Clean and filter scenes
        for scene in potential_scenes:
            scene = scene.strip()
            if scene and len(scene) > 10:  # Minimum scene length
                scenes.append(scene)
        
        # Limit scenes for Render performance
        if len(scenes) > self.max_scenes:
            logger.warning(f"Limiting scenes from {len(scenes)} to {self.max_scenes} for performance")
            scenes = scenes[:self.max_scenes]
        
        # Ensure we have at least one scene
        if not scenes:
            scenes = [script_text]
        
        logger.info(f"Parsed {len(scenes)} scenes from script")
        return scenes
    
    def enhance_prompt(self, scene_text: str) -> str:
        """
        Enhance scene text for better image generation.
        
        Args:
            scene_text: Original scene description
            
        Returns:
            Enhanced prompt for image generation
        """
        # Add cinematic quality keywords
        cinematic_keywords = [
            "cinematic lighting", "high quality", "detailed", "atmospheric",
            "dramatic", "professional photography", "8k resolution"
        ]
        
        # Add style keywords based on scene content
        style_keywords = []
        scene_lower = scene_text.lower()
        
        if any(word in scene_lower for word in ["dark", "night", "shadow", "mystery"]):
            style_keywords.extend(["moody lighting", "dark atmosphere"])
        elif any(word in scene_lower for word in ["bright", "sun", "day", "happy"]):
            style_keywords.extend(["bright lighting", "vibrant colors"])
        elif any(word in scene_lower for word in ["forest", "nature", "outdoor"]):
            style_keywords.extend(["natural lighting", "landscape"])
        elif any(word in scene_lower for word in ["city", "urban", "street"]):
            style_keywords.extend(["urban setting", "architectural"])
        
        # Combine original text with enhancements
        enhanced_prompt = f"{scene_text}, {', '.join(random.sample(cinematic_keywords, 2))}"
        if style_keywords:
            enhanced_prompt += f", {', '.join(random.sample(style_keywords, 1))}"
        
        return enhanced_prompt
    
    def generate_image(self, prompt: str, scene_index: int) -> str:
        """
        Generate image for a scene using Stable Diffusion with memory management.
        
        Args:
            prompt: Text prompt for image generation
            scene_index: Index of the scene for filename
            
        Returns:
            Path to generated image
        """
        try:
            logger.info(f"Generating image for scene {scene_index + 1}")
            
            # Generate image with lower inference steps for speed
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
                    num_inference_steps=20,  # Reduced for speed
                    guidance_scale=7.5,
                    width=self.video_width,
                    height=self.video_height,
                    generator=torch.Generator(device=self.device).manual_seed(random.randint(0, 2**32))
                )
            
            image = result.images[0]
            
            # Save image
            image_path = self.scenes_dir / f"scene_{scene_index:03d}.png"
            image.save(image_path)
            
            # Memory cleanup
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Image saved: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to generate image for scene {scene_index}: {e}")
            # Create a fallback colored image
            return self._create_fallback_image(scene_index)
    
    def _create_fallback_image(self, scene_index: int) -> str:
        """Create a fallback colored image if generation fails."""
        colors = [(64, 64, 64), (32, 32, 64), (64, 32, 32), (32, 64, 32)]
        color = colors[scene_index % len(colors)]
        
        image = Image.new('RGB', (self.video_width, self.video_height), color)
        image_path = self.scenes_dir / f"scene_{scene_index:03d}_fallback.png"
        image.save(image_path)
        
        return str(image_path)
    
    def apply_cinematic_effects(self, image_path: str, scene_index: int) -> ImageClip:
        """
        Apply cinematic effects to an image to create a video clip.
        
        Args:
            image_path: Path to the scene image
            scene_index: Index of the scene
            
        Returns:
            ImageClip with cinematic effects
        """
        # Create image clip
        clip = ImageClip(image_path, duration=self.scene_duration)
        
        # Apply different cinematic effects based on scene
        effect_type = scene_index % 3  # Reduced effects for performance
        
        if effect_type == 0:
            # Zoom in effect
            clip = clip.resize(lambda t: 1 + 0.02 * t)
        elif effect_type == 1:
            # Pan left to right
            clip = clip.set_position(lambda t: (-30 + 60 * t / self.scene_duration, 'center'))
        else:
            # Zoom out effect
            clip = clip.resize(lambda t: 1.05 - 0.01 * t)
        
        # Add fade in/out effects
        clip = clip.fadein(self.transition_duration).fadeout(self.transition_duration)
        
        # Ensure proper dimensions
        clip = clip.resize(height=self.video_height)
        
        return clip
    
    def create_video(self, scenes: List[str], output_filename: str) -> str:
        """
        Create final video from scenes with memory management.
        
        Args:
            scenes: List of scene descriptions
            output_filename: Name of output video file
            
        Returns:
            Path to generated video
        """
        try:
            logger.info(f"Creating video with {len(scenes)} scenes")
            
            video_clips = []
            
            # Process each scene
            for i, scene in enumerate(scenes):
                logger.info(f"Processing scene {i + 1}/{len(scenes)}")
                
                # Generate enhanced prompt
                enhanced_prompt = self.enhance_prompt(scene)
                logger.info(f"Enhanced prompt: {enhanced_prompt}")
                
                # Generate image
                image_path = self.generate_image(enhanced_prompt, i)
                
                # Create video clip with cinematic effects
                clip = self.apply_cinematic_effects(image_path, i)
                video_clips.append(clip)
                
                # Aggressive memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all clips
            logger.info("Concatenating video clips")
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # Set final video properties
            final_video = final_video.set_fps(self.fps)
            
            # Save video with optimized settings for Render
            output_path = self.output_dir / output_filename
            logger.info(f"Saving video to: {output_path}")
            
            final_video.write_videofile(
                str(output_path),
                fps=self.fps,
                codec='libx264',
                preset='ultrafast',  # Fastest encoding
                bitrate='4000k',     # Lower bitrate for speed
                temp_audiofile=None,
                remove_temp=True,
                logger=None,         # Disable moviepy logging
                verbose=False
            )
            
            # Clean up clips
            for clip in video_clips:
                clip.close()
            final_video.close()
            
            # Final cleanup
            gc.collect()
            
            logger.info(f"Video created successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            raise
    
    def generate_video_from_script(self, script_text: str, output_filename: str = None) -> str:
        """
        Main method to generate video from script text.
        
        Args:
            script_text: Input script text
            output_filename: Optional output filename
            
        Returns:
            Path to generated video
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"cinematic_video_{timestamp}.mp4"
        
        # Parse script into scenes
        scenes = self.parse_script(script_text)
        
        # Create video
        video_path = self.create_video(scenes, output_filename)
        
        return video_path
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Failed to cleanup temporary files: {e}")

# Flask API
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global video generator instance
video_generator = None

def initialize_generator():
    """Initialize the video generator when the app starts."""
    global video_generator
    try:
        video_generator = CinematicVideoGenerator()
        logger.info("Video generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize video generator: {e}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation."""
    return jsonify({
        "name": "Cinematic Video Generator API",
        "version": "1.0.0",
        "description": "Generate cinematic videos from text scripts using AI",
        "endpoints": {
            "POST /generate-video": "Generate video from script text",
            "GET /health": "Health check",
            "GET /download/<filename>": "Download generated video",
            "GET /list-videos": "List all generated videos"
        },
        "status": "ready" if video_generator else "initializing"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "device": video_generator.device if video_generator else "unknown",
        "environment": RENDER_ENV,
        "max_workers": MAX_WORKERS,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/generate-video', methods=['POST'])
def generate_video():
    """
    Generate video from script text.
    
    Expected JSON payload:
    {
        "script": "Your story text here...",
        "output_filename": "optional_custom_name.mp4"
    }
    """
    try:
        if not video_generator:
            # Try to initialize if not already done
            initialize_generator()
            if not video_generator:
                return jsonify({"error": "Video generator not initialized"}), 500
        
        # Get request data
        data = request.get_json()
        if not data or 'script' not in data:
            return jsonify({"error": "Missing 'script' in request body"}), 400
        
        script_text = data['script']
        output_filename = data.get('output_filename')
        
        if not script_text.strip():
            return jsonify({"error": "Script text cannot be empty"}), 400
        
        # Validate script length for Render
        if len(script_text) > 5000:
            return jsonify({"error": "Script too long. Maximum 5000 characters."}), 400
        
        # Generate video
        logger.info("Starting video generation")
        video_path = video_generator.generate_video_from_script(script_text, output_filename)
        
        return jsonify({
            "status": "success",
            "video_path": video_path,
            "filename": os.path.basename(video_path),
            "message": "Video generated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download generated video."""
    try:
        if not video_generator:
            return jsonify({"error": "Video generator not initialized"}), 500
            
        secure_name = secure_filename(filename)
        file_path = video_generator.output_dir / secure_name
        
        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/list-videos', methods=['GET'])
def list_videos():
    """List all generated videos."""
    try:
        if not video_generator:
            return jsonify({"videos": []})
            
        videos = []
        
        for file_path in video_generator.output_dir.glob("*.mp4"):
            videos.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
            })
        
        return jsonify({"videos": videos})
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error."""
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({"error": "Internal server error occurred."}), 500

if __name__ == '__main__':
    """
    Main entry point for the application.
    
    For Render deployment, this will be called with: python app.py
    
    API Endpoints:
    - GET / - API documentation
    - GET /health - Health check
    - POST /generate-video - Generate video from script
    - GET /download/<filename> - Download generated video
    - GET /list-videos - List all generated videos
    """
    # Initialize the generator at startup
    initialize_generator()
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    print("=" * 60)
    print("CINEMATIC VIDEO GENERATOR - RENDER OPTIMIZED")
    print("=" * 60)
    print(f"Starting Flask server on port {port}...")
    print(f"Environment: {RENDER_ENV}")
    print(f"Device: {video_generator.device if video_generator else 'unknown'}")
    print(f"Max Workers: {MAX_WORKERS}")
    print("\nAPI Endpoints:")
    print("- GET / - API documentation")
    print("- POST /generate-video - Generate video from script")
    print("- GET /health - Health check")
    print("- GET /download/<filename> - Download video")
    print("- GET /list-videos - List generated videos")
    print("=" * 60)
    
    # Run the app
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=(RENDER_ENV != 'production'),
        threaded=True
    )
