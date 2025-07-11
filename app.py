#!/usr/bin/env python3
"""
Cinematic Video Generator Backend

A complete Python backend that generates cinematic storytelling videos from text scripts
using Stable Diffusion for image generation and OpenCV for video creation.

INSTALLATION INSTRUCTIONS:
1. Install Python 3.8+ and pip
2. Install system dependencies:
   - Linux: sudo apt-get install libopencv-dev python3-opencv
   - macOS: brew install opencv
   - Windows: pip install opencv-python will handle this
3. Install Python dependencies: pip install -r requirements.txt
4. Run the server: python cinematic_video_generator.py
5. Send POST request to http://localhost:5000/generate-video with script text

REQUIREMENTS.txt:
flask==2.3.3
torch>=1.13.0
diffusers>=0.21.0
transformers>=4.21.0
opencv-python>=4.8.0
pillow>=9.0.0
numpy>=1.21.0
accelerate>=0.21.0

USAGE:
- Send a POST request to /generate-video with JSON: {"script": "your story text"}
- The system will automatically break the script into scenes
- Each scene generates a high-quality image using Stable Diffusion
- Images are combined with cinematic effects into a vertical video
- Final video is saved to /output/ folder

FEATURES:
- Automatic scene detection and splitting
- High-quality image generation with Stable Diffusion
- Cinematic camera effects (zoom, pan, fade, crossfade)
- Vertical video format (1080x1920) for YouTube Shorts
- 100% local processing, no external APIs
- Clean, modular, well-commented code
- OpenCV for efficient video processing
"""

import os
import re
import json
import time
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Flask for API
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Image generation
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import PIL.Image as Image
import numpy as np

# Video creation with OpenCV
import cv2

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CinematicVideoGenerator:
    """Main class for generating cinematic videos from text scripts."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initialize the video generator.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        
        # Video settings
        self.video_width = 1080
        self.video_height = 1920
        self.fps = 30
        self.scene_duration = 4.0  # seconds per scene
        self.transition_duration = 0.5  # seconds for transitions
        
        # Directories
        self.scenes_dir = Path("scenes")
        self.output_dir = Path("output")
        self.scenes_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Stable Diffusion
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize Stable Diffusion pipeline."""
        try:
            logger.info(f"Initializing Stable Diffusion pipeline on {self.device}")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # Optimize for memory
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_model_cpu_offload()
            
            logger.info("Stable Diffusion pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def parse_script(self, script_text: str) -> List[str]:
        """
        Parse script text into individual scenes.
        
        Args:
            script_text: Raw script text
            
        Returns:
            List of scene descriptions
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
            "dramatic", "professional photography", "8k resolution",
            "film grain", "depth of field", "bokeh"
        ]
        
        # Add style keywords based on scene content
        style_keywords = []
        scene_lower = scene_text.lower()
        
        if any(word in scene_lower for word in ["dark", "night", "shadow", "mystery"]):
            style_keywords.extend(["moody lighting", "dark atmosphere", "noir style"])
        elif any(word in scene_lower for word in ["bright", "sun", "day", "happy"]):
            style_keywords.extend(["bright lighting", "vibrant colors", "cheerful atmosphere"])
        elif any(word in scene_lower for word in ["forest", "nature", "outdoor"]):
            style_keywords.extend(["natural lighting", "landscape", "organic"])
        elif any(word in scene_lower for word in ["city", "urban", "street"]):
            style_keywords.extend(["urban setting", "architectural", "modern"])
        
        # Combine original text with enhancements
        enhanced_prompt = f"{scene_text}, {', '.join(random.sample(cinematic_keywords, 3))}"
        if style_keywords:
            enhanced_prompt += f", {', '.join(random.sample(style_keywords, 2))}"
        
        return enhanced_prompt
    
    def generate_image(self, prompt: str, scene_index: int) -> str:
        """
        Generate image for a scene using Stable Diffusion.
        
        Args:
            prompt: Text prompt for image generation
            scene_index: Index of the scene for filename
            
        Returns:
            Path to generated image
        """
        try:
            logger.info(f"Generating image for scene {scene_index + 1}")
            
            # Generate image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, ugly, bad anatomy, worst quality",
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=self.video_width,
                    height=self.video_height,
                    generator=torch.Generator(device=self.device).manual_seed(random.randint(0, 2**32))
                )
            
            image = result.images[0]
            
            # Save image
            image_path = self.scenes_dir / f"scene_{scene_index:03d}.png"
            image.save(image_path)
            
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
    
    def apply_fade_transition(self, img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
        """
        Apply fade transition between two images.
        
        Args:
            img1: First image
            img2: Second image
            progress: Transition progress (0.0 to 1.0)
            
        Returns:
            Blended image
        """
        alpha = progress
        beta = 1.0 - alpha
        return cv2.addWeighted(img1, beta, img2, alpha, 0)
    
    def apply_cinematic_effect(self, image: np.ndarray, effect_type: int, progress: float) -> np.ndarray:
        """
        Apply cinematic effect to an image based on progress.
        
        Args:
            image: Input image
            effect_type: Type of effect to apply
            progress: Progress through the effect (0.0 to 1.0)
            
        Returns:
            Processed image
        """
        h, w = image.shape[:2]
        
        if effect_type == 0:  # Zoom in
            scale = 1.0 + (0.2 * progress)
            center_x, center_y = w // 2, h // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            
        elif effect_type == 1:  # Pan left to right
            shift_x = int(-100 + (200 * progress))
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            
        elif effect_type == 2:  # Pan right to left
            shift_x = int(100 - (200 * progress))
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            
        elif effect_type == 3:  # Zoom out
            scale = 1.2 - (0.2 * progress)
            center_x, center_y = w // 2, h // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            
        elif effect_type == 4:  # Diagonal pan with zoom
            scale = 1.0 + (0.1 * progress)
            shift_x = int(-50 + (100 * progress))
            shift_y = int(-30 + (60 * progress))
            center_x, center_y = w // 2, h // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            M[0, 2] += shift_x
            M[1, 2] += shift_y
            result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            
        else:  # Slow zoom with rotation
            scale = 1.0 + (0.05 * progress)
            angle = 2 * progress  # Slight rotation
            center_x, center_y = w // 2, h // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
            result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        return result
    
    def create_video(self, scenes: List[str], output_filename: str) -> str:
        """
        Create final video from scenes using OpenCV.
        
        Args:
            scenes: List of scene descriptions
            output_filename: Name of output video file
            
        Returns:
            Path to generated video
        """
        try:
            logger.info(f"Creating video with {len(scenes)} scenes")
            
            # Generate all images first
            scene_images = []
            for i, scene in enumerate(scenes):
                logger.info(f"Processing scene {i + 1}/{len(scenes)}")
                
                # Generate enhanced prompt
                enhanced_prompt = self.enhance_prompt(scene)
                logger.info(f"Enhanced prompt: {enhanced_prompt}")
                
                # Generate image
                image_path = self.generate_image(enhanced_prompt, i)
                scene_images.append(image_path)
                
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create video
            output_path = self.output_dir / output_filename
            logger.info(f"Creating video file: {output_path}")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                self.fps, 
                (self.video_width, self.video_height)
            )
            
            if not video_writer.isOpened():
                raise Exception("Failed to open video writer")
            
            # Process each scene
            for i, image_path in enumerate(scene_images):
                logger.info(f"Processing video for scene {i + 1}/{len(scene_images)}")
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    continue
                
                # Resize image to match video dimensions
                image = cv2.resize(image, (self.video_width, self.video_height))
                
                # Calculate frame counts
                total_frames = int(self.scene_duration * self.fps)
                fade_frames = int(self.transition_duration * self.fps)
                
                # Generate frames with cinematic effects
                effect_type = i % 6  # 6 different effects
                
                for frame_idx in range(total_frames):
                    # Calculate progress through the scene
                    progress = frame_idx / total_frames
                    
                    # Apply cinematic effect
                    processed_frame = self.apply_cinematic_effect(image, effect_type, progress)
                    
                    # Apply fade in/out
                    if frame_idx < fade_frames:
                        # Fade in
                        alpha = frame_idx / fade_frames
                        black_frame = np.zeros_like(processed_frame)
                        processed_frame = self.apply_fade_transition(black_frame, processed_frame, alpha)
                    elif frame_idx >= total_frames - fade_frames:
                        # Fade out
                        alpha = (total_frames - frame_idx) / fade_frames
                        black_frame = np.zeros_like(processed_frame)
                        processed_frame = self.apply_fade_transition(black_frame, processed_frame, alpha)
                    
                    # Write frame to video
                    video_writer.write(processed_frame)
                
                # Add cross-fade transition between scenes (except for last scene)
                if i < len(scene_images) - 1:
                    # Load next image for transition
                    next_image_path = scene_images[i + 1]
                    next_image = cv2.imread(next_image_path)
                    if next_image is not None:
                        next_image = cv2.resize(next_image, (self.video_width, self.video_height))
                        
                        # Create crossfade transition
                        transition_frames = int(self.transition_duration * self.fps)
                        for t_idx in range(transition_frames):
                            alpha = t_idx / transition_frames
                            
                            # Apply effects to both images
                            current_effect = self.apply_cinematic_effect(image, effect_type, 1.0)
                            next_effect = self.apply_cinematic_effect(next_image, (i + 1) % 6, 0.0)
                            
                            # Blend the images
                            blended_frame = self.apply_fade_transition(current_effect, next_effect, alpha)
                            video_writer.write(blended_frame)
            
            # Release video writer
            video_writer.release()
            
            logger.info(f"Video created successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            if 'video_writer' in locals():
                video_writer.release()
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

# Flask API
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global video generator instance
video_generator = None

@app.before_first_request
def initialize_generator():
    """Initialize the video generator when the app starts."""
    global video_generator
    try:
        video_generator = CinematicVideoGenerator()
        logger.info("Video generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize video generator: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "device": video_generator.device if video_generator else "unknown",
        "opencv_version": cv2.__version__,
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
            return jsonify({"error": "Video generator not initialized"}), 500
        
        # Get request data
        data = request.get_json()
        if not data or 'script' not in data:
            return jsonify({"error": "Missing 'script' in request body"}), 400
        
        script_text = data['script']
        output_filename = data.get('output_filename')
        
        if not script_text.strip():
            return jsonify({"error": "Script text cannot be empty"}), 400
        
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
        secure_name = secure_filename(filename)
        file_path = Path("output") / secure_name
        
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
        output_dir = Path("output")
        videos = []
        
        for file_path in output_dir.glob("*.mp4"):
            videos.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
            })
        
        return jsonify({"videos": videos})
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    """
    Main entry point for the application.
    
    To run: python cinematic_video_generator.py
    
    API Endpoints:
    - GET /health - Health check
    - POST /generate-video - Generate video from script
    - GET /download/<filename> - Download generated video
    - GET /list-videos - List all generated videos
    
    Example usage:
    curl -X POST http://localhost:5000/generate-video \
         -H "Content-Type: application/json" \
         -d '{"script": "A lonely astronaut drifts through space. The Earth grows smaller behind them. They reach for the stars, hoping to find a new home among the cosmos."}'
    """
    print("=" * 60)
    print("CINEMATIC VIDEO GENERATOR (OpenCV Version)")
    print("=" * 60)
    print("Starting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("- POST /generate-video - Generate video from script")
    print("- GET /health - Health check")
    print("- GET /download/<filename> - Download video")
    print("- GET /list-videos - List generated videos")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
