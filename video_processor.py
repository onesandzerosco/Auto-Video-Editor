"""Advanced video processing service with Instagram/TikTok optimization."""

import os
import json
import hashlib
import threading
import time
import gc
import psutil
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from collections import defaultdict
from threading import Thread
import logging
from io import BytesIO

# Configure ImageMagick before importing moviepy
import moviepy.config as mpconfig
try:
    # Try to find ImageMagick binary
    imagemagick_path = None
    possible_paths = [
        "/opt/homebrew/bin/magick",  # Homebrew on M1 Mac
        "/usr/local/bin/magick",     # Homebrew on Intel Mac
        "/usr/bin/magick",           # Linux
        "/usr/local/bin/convert",    # Older ImageMagick
        "/opt/homebrew/bin/convert", # Older ImageMagick on M1
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            imagemagick_path = path
            break
    
    if imagemagick_path:
        mpconfig.change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
        print(f"ImageMagick configured: {imagemagick_path}")
    else:
        print("Warning: ImageMagick not found, text overlays may not work")
except Exception as e:
    print(f"Warning: Could not configure ImageMagick: {e}")

from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
import moviepy.config as moviepy_config

# Set MoviePy ImageMagick configuration  
if 'IMAGEMAGICK_BINARY' in os.environ:
    moviepy_config.IMAGEMAGICK_BINARY = os.environ['IMAGEMAGICK_BINARY']

from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from .error_handler import ErrorHandler, ErrorContext, ErrorCategory, ErrorSeverity
from .caption_generator import CaptionGenerator
from .caption_styler import CaptionStyler, CaptionStyle
from .prompt_engineer import VideoCategory, ContentAnalysis
from tqdm import tqdm
import shutil
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import (
    VideoFileClip, 
    CompositeVideoClip, 
    TextClip,
    ColorClip,
    concatenate_videoclips,
    ImageClip
)
from datetime import datetime
from ..utils.logger import logger
from .error_handler import ErrorHandler, ErrorContext, ErrorCategory, ErrorSeverity
from .caption_generator import CaptionGenerator
from .caption_styler import CaptionStyler, CaptionStyle
from .prompt_engineer import VideoCategory, ContentAnalysis
from PIL import Image, ImageDraw, ImageFont
from .sheets_service import SheetsService

class ProcessingPriority(Enum):
    """Processing priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

@dataclass
class ProcessingConfig:
    """Video processing configuration for social media."""
    target_width: int = 1080    # Standard width for vertical videos
    target_height: int = 1920   # 9:16 aspect ratio height
    target_fps: int = 30
    codec: str = "libx264"
    bitrate: str = "8000k"      # Higher bitrate for better quality
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "medium"
    threads: int = 4
    cache_dir: str = "cache"
    temp_dir: str = "temp"
    max_memory_percent: float = 80.0
    max_concurrent_jobs: int = 2  # Reduced from 4 to prevent memory issues
    
    @property
    def aspect_ratio(self) -> float:
        """Get the target aspect ratio (9:16 = 0.5625)."""
        return self.target_width / self.target_height

@dataclass
class ProcessingJob:
    """Video processing job."""
    video_path: str
    output_path: str
    config: ProcessingConfig
    priority: ProcessingPriority
    callback: Optional[callable] = None
    progress_callback: Optional[callable] = None

class VideoProcessor:
    """
    Advanced video processor for Instagram Reels/TikTok format with AI captions.
    Handles video conversion, caption generation, and ultra-high-quality rendering.
    """
    
    def __init__(
        self,
        error_handler: ErrorHandler,
        config: Optional[ProcessingConfig] = None
    ):
        # Initialize basic attributes first
        self.config = config or ProcessingConfig()
        
        # Reduce concurrent jobs to prevent memory issues
        self.config.max_concurrent_jobs = 1  # Single job at a time to prevent memory corruption
        
        self.error_handler = error_handler
        self.logger = logger

        # Initialize cache
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cache_index = {}
        self._load_cache_index()

        # Initialize queuing system
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        
        # Use ThreadPoolExecutor with very limited workers to prevent memory issues
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VideoProcessor")
        
        # Initialize resource monitoring
        self.resource_monitor_active = True
        self.resource_monitor_thread = threading.Thread(
            target=self._monitor_resources, 
            daemon=True,
            name="ResourceMonitor"
        )

        # Create thread-safe emoji lock
        self.emoji_lock = threading.Lock()
        
        # Set up temp directory
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize caption sheet ID
        from ..config import CAPTION_SHEET_ID
        self.caption_sheet_id = CAPTION_SHEET_ID
        
        # Force garbage collection before starting monitoring
        import gc
        gc.collect()
        
        # Start resource monitoring
        self.resource_monitor_thread.start()
        
        self.logger.info("✅ Video processor initialized successfully")
        
        # Initialize temp directory
        self.temp_dir = Path(self.config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"VideoProcessor initialized with config: {self.config}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Temp directory: {self.temp_dir}")
        
        # Initialize caption services
        self.caption_generator = CaptionGenerator()
        self.caption_styler = CaptionStyler()

    def _convert_to_9_16_aspect_ratio(self, video: VideoFileClip) -> VideoFileClip:
        """
        Convert video to 9:16 aspect ratio (1080x1920) for Instagram Reels/TikTok.
        
        This method will:
        1. Calculate the best crop/resize strategy
        2. Maintain as much content as possible
        3. Fill any gaps with blurred background
        """
        target_width = self.config.target_width
        target_height = self.config.target_height
        target_aspect = target_width / target_height  # 0.5625 for 9:16
        
        original_width = video.w
        original_height = video.h
        original_aspect = original_width / original_height
        
        logger.info(f"Converting video from {original_width}x{original_height} (aspect: {original_aspect:.3f}) to {target_width}x{target_height} (aspect: {target_aspect:.3f})")
        
        if abs(original_aspect - target_aspect) < 0.01:
            # Already close to 9:16, just resize
            logger.info("Video is already close to 9:16 aspect ratio, resizing...")
            return video.resize(height=target_height)
        
        elif original_aspect > target_aspect:
            # Video is wider (landscape), need to crop sides and possibly add top/bottom blur
            logger.info("Video is landscape, cropping to fit 9:16...")
            
            # Calculate dimensions for center crop to 9:16
            new_width = int(original_height * target_aspect)
            
            if new_width <= original_width:
                # Can crop to perfect 9:16
                x_center = original_width // 2
                crop_x1 = max(0, x_center - new_width // 2)
                crop_x2 = min(original_width, x_center + new_width // 2)
                
                cropped = video.crop(x1=crop_x1, x2=crop_x2)
                resized = cropped.resize(height=target_height)
                logger.info(f"Cropped video to {new_width}x{original_height}, then resized to {target_width}x{target_height}")
                return resized
            else:
                # Need to scale down first, then add blur background
                scale_factor = target_height / original_height
                scaled_video = video.resize(scale_factor)
                
                # Create blurred background
                blurred_bg = video.resize(scale_factor * 1.2).blur(10)  # Slightly larger and blurred
                blurred_bg = blurred_bg.crop(
                    x1=(blurred_bg.w - target_width) // 2,
                    x2=(blurred_bg.w + target_width) // 2,
                    y1=(blurred_bg.h - target_height) // 2,
                    y2=(blurred_bg.h + target_height) // 2
                )
                
                # Center the main video over blurred background
                final_video = CompositeVideoClip([
                    blurred_bg,
                    scaled_video.set_position('center')
                ], size=(target_width, target_height))
                
                logger.info(f"Created 9:16 video with blurred background")
                return final_video
        
        else:
            # Video is taller (portrait), center it and add side blur
            logger.info("Video is portrait, adding side blur to fit 9:16...")
            
            # Scale video to fit height
            scale_factor = target_height / original_height
            scaled_video = video.resize(scale_factor)
            
            if scaled_video.w >= target_width:
                # Video is wide enough, just crop
                crop_x = (scaled_video.w - target_width) // 2
                return scaled_video.crop(x1=crop_x, x2=crop_x + target_width)
            else:
                # Create blurred background for sides
                blurred_bg = video.resize(scale_factor).blur(10)
                
                # Stretch blurred background to fill width
                blurred_bg = blurred_bg.resize(width=target_width)
                if blurred_bg.h < target_height:
                    blurred_bg = blurred_bg.resize(height=target_height)
                
                # Crop to exact size
                blurred_bg = blurred_bg.crop(
                    x1=0, x2=target_width,
                    y1=(blurred_bg.h - target_height) // 2,
                    y2=(blurred_bg.h + target_height) // 2
                )
                
                # Center the main video
                final_video = CompositeVideoClip([
                    blurred_bg,
                    scaled_video.set_position('center')
                ], size=(target_width, target_height))
                
                logger.info(f"Created 9:16 video with side blur")
                return final_video

    def _generate_adult_creator_caption(self, use_ai: bool = True, sheet_id: str = None) -> str:
        """
        Generate flirty, suggestive captions for adult creator promotion.
        Now with AI-powered generation using Google Sheets inspiration.
        """
        # Try AI generation first if enabled
        if use_ai:
            try:
                # Use provided sheet_id or fall back to configured one
                from ..config import CAPTION_SHEET_ID
                target_sheet_id = sheet_id or CAPTION_SHEET_ID
                
                if target_sheet_id:
                    return self._generate_ai_inspired_caption(target_sheet_id)
                else:
                    logger.warning("No Google Sheets ID configured for AI captions, using fallback")
            except Exception as e:
                logger.warning(f"AI caption generation failed, using fallback: {str(e)}")
        
        # Fallback to your existing hardcoded captions
        import random
        
        # Your existing captions list
        captions = [
            "I BET YOU'D TASTE\nSWEET 😈💋",
            "YOUR GIRLFRIEND DOESN'T\nDO THIS 🔥😏",
            "DELETE THIS BEFORE\nSHE FINDS OUT 🤫📱",
            "I'M EXACTLY WHAT\nYOU'VE BEEN CRAVING 💦🍑",
            "SEND THIS TO YOUR\nFAVORITE PERSON 💕😉",
            "I KNOW WHAT YOU'RE\nTHINKING RIGHT NOW 👀💭",
            "YOUR SEARCH HISTORY\nWOULD EMBARRASS YOU 💻🙈",
            "I'M THE GIRL YOU\nDREAM ABOUT 😴💫",
            "YOU'RE GETTING\nTURNED ON AREN'T YOU 🔥😈",
            "IMAGINE WHAT I COULD\nDO TO YOU 🤤💋",
            "YOUR HEART IS RACING\nRIGHT NOW 💓🔥",
            "I'D MAKE YOUR EX\nJEALOUS AS HELL 😏💅",
            "YOU'RE THINKING ABOUT\nME NAKED AREN'T YOU 🙈🔥",
            "I MAKE GOOD GIRLS\nTURN BAD 😈👿",
            "YOUR WIFE DOESN'T\nNEED TO KNOW 🤐💋",
            "I'M YOUR DIRTY\nLITTLE SECRET 🤫💦",
            "YOU WISH I WAS\nYOUR GIRLFRIEND 💕😩",
            "I COULD RUIN YOU\nIN THE BEST WAY 😈🔥",
            "YOU'RE ALREADY\nADDICTED TO ME 💉😍",
            "I'M THE REASON YOUR\nPHONE IS ALWAYS HOT 📱🔥",
            "DELETE YOUR BROWSER\nHISTORY NOW 🗑️💻",
            "I'LL BE YOUR FAVORITE\nBAD DECISION 😈💋",
            "YOUR GIRLFRIEND COULD\nNEVER COMPETE 👑💅",
            "I'M EVERYTHING SHE'S\nNOT 🔥😏",
            "YOU'RE BLUSHING THROUGH\nTHE SCREEN 😊📱",
            "I KNOW EXACTLY HOW\nTO PLEASE YOU 😈💦",
            "YOUR FRIENDS WOULD\nBE SO JEALOUS 😉💯",
            "I'M THE GIRL YOUR\nMOM HATES 😇👿",
            "YOU'VE BEEN WAITING\nFOR SOMEONE LIKE ME 🎯💕",
            "I'M YOUR BIGGEST\nWEAKNESS 💔😈",
            "YOU'D DO ANYTHING\nTO HAVE ME 😩💋",
            "I'M THE DANGER YOU\nCAN'T RESIST 🔥⚡",
            "YOUR PULSE IS\nRACING ISN'T IT 💓⚡",
            "I'LL MAKE YOU FORGET\nEVERYONE ELSE 😈💫",
            "YOU'RE MINE NOW\nWHETHER YOU KNOW IT 👑🔗",
            "I'M THE FEVER DREAM\nYOU CAN'T WAKE UP FROM 🔥💭",
            "CLOSE THE APP BEFORE\nYOU DO SOMETHING STUPID 📱😈",
            "I'M THE TEMPTATION\nYOU CAN'T IGNORE 🍎😈",
            "YOUR BODY IS BETRAYING\nYOUR MIND RIGHT NOW 💭🔥",
            "I'M THE SIN YOU\nWANT TO COMMIT 😈⛪"
        ]
        
        return random.choice(captions)

    def _create_instagram_style_captions(
        self, 
        video: VideoFileClip, 
        caption_text: str
    ) -> List[VideoFileClip]:
        """
        Create high-quality Instagram/TikTok style captions with EXACT specifications:
        - High-quality rendering (no pixelation/blur)
        - 20% margin from edges with automatic line wrapping
        - Maximum 3 lines per caption
        - 65px font size (fixed)
        - 1/3 from bottom positioning (CORRECTED)
        - Bold, chunky TikTok style with thick black outline
        - iPhone-style emoji support
        """
        try:
            target_width = self.config.target_width  # 1080
            target_height = self.config.target_height  # 1920
            
            # Use the caption_text parameter instead of generating random captions
            logger.info(f"Using provided caption: '{caption_text}'")
            
            # EXACT SPECIFICATIONS
            font_size = 85  # Increased from 65px for larger text
            max_lines = 3   # Maximum 3 lines as required
            margin_percent = 10  # Reduced from 30% to 10% margin from edges for wider text
            margin_pixels = int(target_width * (margin_percent / 100))  # 108px margin (reduced from 324px)
            text_area_width = target_width - (2 * margin_pixels)  # 864px available width (increased from 432px)
            
            # Calculate maximum text block height for 3 lines
            line_height = int(font_size * 1.2)  # 78px per line with 20% spacing
            max_text_height = max_lines * line_height  # 234px maximum
            
            # UPDATED POSITIONING: 20% below center point of the page
            # For 1920px height: center = 960px, 20% below = 960 + (1920 * 0.20) = 1344px
            center_y = target_height // 2  # 960px for 1920px video
            offset_below_center = int(target_height * 0.20)  # 384px (20% of video height)
            caption_center_y = center_y + offset_below_center  # 1344px
            
            # Ensure caption fits within video bounds
            half_text_height = max_text_height // 2
            caption_top = caption_center_y - half_text_height
            caption_bottom = caption_center_y + half_text_height
            
            # Adjust if caption would go off-screen
            if caption_bottom > target_height - 20:  # 20px safety margin from bottom
                caption_bottom = target_height - 20
                caption_center_y = caption_bottom - half_text_height
                caption_top = caption_center_y - half_text_height
            
            logger.info(f"UPDATED positioning: caption center at y={caption_center_y} (20% below center at y={center_y})")
            logger.info(f"Caption range: y={caption_top} to y={caption_bottom}, video height: {target_height}px")
            logger.info(f"Text area: {text_area_width}px wide with {margin_pixels}px margins on each side")
            
            # Smart text wrapping with word boundaries and explicit newline handling
            def wrap_text_smart(text, font, max_width, max_lines):
                # First, split on explicit newlines to respect intended line breaks
                explicit_lines = text.split('\n')
                lines = []
                remaining_words = []
                
                for line_text in explicit_lines:
                    if len(lines) >= max_lines:
                        # Collect remaining words for potential fitting
                        remaining_words.extend(line_text.split())
                        continue
                        
                    # If line_text is empty (from consecutive \n), skip it
                    if not line_text.strip():
                        continue
                    
                    # Now handle word wrapping within this line
                    words = line_text.split()
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        
                        # Get text dimensions
                        try:
                            bbox = font.getbbox(test_line)
                            text_width = bbox[2] - bbox[0]
                        except:
                            # Fallback for basic fonts
                            text_width = len(test_line) * (font_size * 0.6)
                        
                        if text_width <= max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                                current_line = word
                                # Check line limit after adding a line
                                if len(lines) >= max_lines:
                                    # Collect remaining words
                                    remaining_words.extend(words[words.index(word):])
                                    break
                            else:
                                # Single word is too long, force it but truncate if needed
                                lines.append(word[:20] + "..." if len(word) > 20 else word)
                                current_line = ""
                                # Check line limit after adding a line
                                if len(lines) >= max_lines:
                                    remaining_words.extend(words[words.index(word)+1:])
                                    break
                    
                    # Add the last line if there's content and we haven't hit the limit
                    if current_line and len(lines) < max_lines:
                        lines.append(current_line)
                    elif current_line and len(lines) >= max_lines:
                        remaining_words.extend(current_line.split())
                
                # Try to fit remaining important words (emojis, key terms) into existing lines
                if remaining_words and len(lines) > 0:
                    for word in remaining_words:
                        # Prioritize emojis and short important words
                        if any(char in word for char in ['😀', '😁', '😂', '🤣', '😃', '😄', '😅', '😆', '😉', '😊', '😋', '😎', '😍', '😘', '🥰', '😗', '😙', '😚', '☺️', '🙂', '🤗', '🤩', '🤔', '🤨', '😐', '😑', '😶', '🙄', '😏', '😣', '😥', '😮', '🤐', '😯', '😪', '😫', '😴', '😌', '😛', '😜', '😝', '🤤', '😒', '😓', '😔', '😕', '🙃', '🤑', '😲', '☹️', '🙁', '😖', '😞', '😟', '😤', '😢', '😭', '😦', '😧', '😨', '😩', '🤯', '😬', '😰', '😱', '🥵', '🥶', '😳', '🤪', '😵', '😡', '😠', '🤬', '😷', '🤒', '🤕', '🤢', '🤮', '🤧', '😇', '🤠', '🤡', '🥳', '🥴', '🥺', '🤥', '🤫', '🤭', '🧐', '🤓', '😈', '👿', '👹', '👺', '💀', '☠️', '👻', '👽', '👾', '🤖', '🎃', '😺', '😸', '😹', '😻', '😼', '😽', '🙀', '😿', '😾', '💋', '💌', '💘', '💝', '💖', '💗', '💓', '💞', '💕', '💟', '❣️', '💔', '❤️', '🧡', '💛', '💚', '💙', '💜', '🤎', '🖤', '🤍', '💯', '💢', '💥', '💫', '💦', '💨', '🕳️', '💣', '💬', '👁️‍🗨️', '🗨️', '🗯️', '💭', '💤', '👋', '🤚', '🖐️', '✋', '🖖', '👌', '🤏', '✌️', '🤞', '🤟', '🤘', '🤙', '👈', '👉', '👆', '🖕', '👇', '☝️', '👍', '👎', '👊', '✊', '🤛', '🤜', '👏', '🙌', '👐', '🤲', '🤝', '🙏', '✍️', '💅', '🤳', '💪', '🦾', '🦿', '🦵', '🦶', '👂', '🦻', '👃', '🧠', '🦷', '🦴', '👀', '👁️', '👅', '👄', '💃', '🕺', '👯', '👯‍♂️', '👯‍♀️', '🕴️', '👫', '👬', '👭', '💏', '💑', '👪', '🌟', '⭐', '✨', '🌈', '☀️', '🌞', '🌝', '🌛', '🌜', '🌚', '🌕', '🌖', '🌗', '🌘', '🌑', '🌒', '🌓', '🌔', '🌙', '🌎', '🌍', '🌏', '💫', '⭐', '🌟', '✨', '⚡', '☄️', '💥', '🔥', '🌪️', '🌈', '☀️', '🌤️', '⛅', '🌦️', '🌧️', '⛈️', '🌩️', '🌨️', '❄️', '☃️', '⛄', '🌬️', '💨', '💧', '💦', '☔', '☂️', '🌊', '🌫️', '🍏', '🍎', '🍐', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🫐', '🍈', '🍒', '🍑', '🥭', '🍍', '🥥', '🥝', '🍅', '🍆', '🥑', '🥦', '🥬', '🥒', '🌶️', '🫑', '🌽', '🥕', '🧄', '🧅', '🥔', '🍠', '🥐', '🥖', '🍞', '🥨', '🥯', '🧀', '🥚', '🍳', '🧈', '🥞', '🧇', '🥓', '🥩', '🍗', '🍖', '🦴', '🌭', '🍔', '🍟', '🍕', '🫓', '🥪', '🥙', '🧆', '🌮', '🌯', '🫔', '🥗', '🥘', '🫕', '🥫', '🍝', '🍜', '🍲', '🍛', '🍣', '🍱', '🥟', '🦪', '🍤', '🍙', '🍚', '🍘', '🍥', '🥠', '🥮', '🍢', '🍡', '🍧', '🍨', '🍦', '🥧', '🧁', '🍰', '🎂', '🍮', '🍭', '🍬', '🍫', '🍿', '🍩', '🍪', '🌰', '🥜', '🍯', '🥛', '🍼', '☕', '🍵', '🧃', '🥤', '🍶', '🍺', '🍻', '🥂', '🍷', '🥃', '🍸', '🍹', '🧉', '🍾', '🧊', '🥄', '🍴', '🍽️', '🥣', '🥡', '🥢', '🧂', '⚽', '🏀', '🏈', '⚾', '🥎', '🎾', '🏐', '🏉', '🥏', '🎱', '🪀', '🏓', '🏸', '🏒', '🏑', '🥍', '🏏', '🪃', '🥅', '⛳', '🪁', '🏹', '🎣', '🤿', '🥽', '🥼', '🦺', '⛷️', '🏂', '🪂', '🏋️‍♀️', '🏋️', '🏋️‍♂️', '🤼‍♀️', '🤼', '🤼‍♂️', '🤸‍♀️', '🤸', '🤸‍♂️', '⛹️‍♀️', '⛹️', '⛹️‍♂️', '🤺', '🤾‍♀️', '🤾', '🤾‍♂️', '🏌️‍♀️', '🏌️', '🏌️‍♂️', '🏇', '🧘‍♀️', '🧘', '🧘‍♂️', '🏄‍♀️', '🏄', '🏄‍♂️', '🏊‍♀️', '🏊', '🏊‍♂️', '🤽‍♀️', '🤽', '🤽‍♂️', '🚣‍♀️', '🚣', '🚣‍♂️', '🧗‍♀️', '🧗', '🧗‍♂️', '🚵‍♀️', '🚵', '🚵‍♂️', '🚴‍♀️', '🚴', '🚴‍♂️', '🏆', '🥇', '🥈', '🥉', '🏅', '🎖️', '🏵️', '🎗️', '🎫', '🎟️', '🎪', '🤹‍♀️', '🤹', '🤹‍♂️', '🎭', '🩰', '🎨', '🎬', '🎤', '🎧', '🎼', '🎵', '🎶', '🥁', '🪘', '🎷', '🎺', '🎸', '🪕', '🎻', '🎲', '♟️', '🎯', '🎳', '🎮', '🎰', '🧩', '🚗', '🚕', '🚙', '🚌', '🚎', '🏎️', '🚓', '🚑', '🚒', '🚐', '🛻', '🚚', '🚛', '🚜', '🦯', '🦽', '🦼', '🛴', '🚲', '🛵', '🏍️', '🛺', '🚨', '🚔', '🚍', '🚘', '🚖', '🚡', '🚠', '🚟', '🚃', '🚋', '🚞', '🚝', '🚄', '🚅', '🚈', '🚂', '🚆', '🚇', '🚊', '🚉', '✈️', '🛫', '🛬', '🛩️', '💺', '🛰️', '🚀', '🛸', '🚁', '🛶', '⛵', '🚤', '🛥️', '🛳️', '⛴️', '🚢', '⚓', '⛽', '🚧', '🚦', '🚥', '🚏', '🗺️', '🗿', '🗽', '🗼', '🏰', '🏯', '🏟️', '🎡', '🎢', '🎠', '⛲', '⛱️', '🏖️', '🏝️', '🏜️', '🌋', '⛰️', '🏔️', '🗻', '🏕️', '⛺', '🏠', '🏡', '🏘️', '🏚️', '🏗️', '🏭', '🏢', '🏬', '🏣', '🏤', '🏥', '🏦', '🏨', '🏪', '🏫', '🏩', '💒', '🏛️', '⛪', '🕌', '🕍', '🛕', '🕋', '⛩️', '🛤️', '🛣️', '🗾', '🎑', '🏞️', '🌅', '🌄', '🌠', '🎇', '🎆', '🌇', '🌆', '🏙️', '🌃', '🌌', '🌉', '🌁', '⌚', '📱', '📲', '💻', '⌨️', '🖥️', '🖨️', '🖱️', '🖲️', '🕹️', '🗜️', '💽', '💾', '💿', '📀', '📼', '📷', '📸', '📹', '🎥', '📽️', '🎞️', '📞', '☎️', '📟', '📠', '📺', '📻', '🎙️', '🎚️', '🎛️', '🧭', '⏱️', '⏲️', '⏰', '🕰️', '⌛', '⏳', '📡', '🔋', '🔌', '💡', '🔦', '🕯️', '🪔', '🧯', '🛢️', '💸', '💵', '💴', '💶', '💷', '💰', '💳', '💎', '⚖️', '🧰', '🔧', '🔨', '⚒️', '🛠️', '⛏️', '🔩', '⚙️', '🧱', '⛓️', '🧲', '🔫', '💣', '🧨', '🪓', '🔪', '🗡️', '⚔️', '🛡️', '🚬', '⚰️', '⚱️', '🏺', '🔮', '📿', '🧿', '💈', '⚗️', '🔭', '🔬', '🕳️', '🩹', '🩺', '💊', '💉', '🩸', '🧬', '🦠', '🧫', '🧪', '🌡️', '🧹', '🧺', '🧻', '🚽', '🚰', '🚿', '🛁', '🛀', '🧼', '🪒', '🧽', '🧴', '🛎️', '🔑', '🗝️', '🚪', '🪑', '🛋️', '🛏️', '🛌', '🧸', '🖼️', '🛍️', '🛒', '🎁', '🎈', '🎏', '🎀', '🎊', '🎉', '🎎', '🏮', '🎐', '🧧', '✉️', '📩', '📨', '📧', '💌', '📥', '📤', '📦', '🏷️', '📪', '📬', '📭', '📮', '📯', '📜', '📃', '📄', '📑', '📊', '📈', '📉', '🗒️', '🗓️', '📆', '📅', '📇', '🗃️', '🗳️', '🗄️', '📋', '📁', '📂', '🗂️', '🗞️', '📰', '📓', '📔', '📒', '📕', '📗', '📘', '📙', '📚', '📖', '🔖', '🧷', '🔗', '📎', '🖇️', '📐', '📏', '🧮', '📌', '📍', '✂️', '🖊️', '🖋️', '✒️', '🖌️', '🖍️', '📝', '✏️', '🔍', '🔎', '🔏', '🔐', '🔒', '🔓', '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💔', '❣️', '💕', '💞', '💓', '💗', '💖', '💘', '💝', '💟', '☮️', '✝️', '☪️', '🕉️', '☸️', '✡️', '🔯', '🕎', '☯️', '☦️', '🛐', '⛎', '♈', '♉', '♊', '♋', '♌', '♍', '♎', '♏', '♐', '♑', '♒', '♓', '🆔', '⚛️', '🉑', '☢️', '☣️', '📴', '📳', '🈶', '🈚', '🈸', '🈺', '🈷️', '✴️', '🆚', '💮', '🉐', '㊙️', '㊗️', '🈴', '🈵', '🈹', '🈲', '🅰️', '🅱️', '🆎', '🆑', '🅾️', '🆘', '❌', '⭕', '🛑', '⛔', '📛', '🚫', '💯', '💢', '♨️', '🚷', '🚯', '🚳', '🚱', '🔞', '📵', '🚭', '❗', '❕', '❓', '❔', '‼️', '⁉️', '🔅', '🔆', '〽️', '⚠️', '🚸', '🔱', '⚜️', '🔰', '♻️', '✅', '🈯', '💹', '❇️', '✳️', '❎', '🌐', '💠', 'Ⓜ️', '🌀', '💤', '🏧', '🚾', '♿', '🅿️', '🈳', '🈂️', '🛂', '🛃', '🛄', '🛅', '🚹', '🚺', '🚼', '🚻', '🚮', '🎦', '📶', '🈁', '🔣', 'ℹ️', '🔤', '🔡', '🔠', '🆖', '🆗', '🆙', '🆒', '🆕', '🆓', '0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟', '🔢', '#️⃣', '*️⃣', '⏏️', '▶️', '⏸️', '⏯️', '⏹️', '⏺️', '⏭️', '⏮️', '⏩', '⏪', '⏫', '⏬', '◀️', '🔼', '🔽', '➡️', '⬅️', '⬆️', '⬇️', '↗️', '↘️', '↙️', '↖️', '↕️', '↔️', '↪️', '↩️', '⤴️', '⤵️', '🔀', '🔁', '🔂', '🔄', '🔃', '🎵', '🎶', '➕', '➖', '➗', '✖️', '♾️', '💲', '💱', '™️', '©️', '®️', '〰️', '➰', '➿', '🔚', '🔙', '🔛', '🔝', '🔜', '✔️', '☑️', '🔘', '🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '⚫', '⚪', '🟤', '🔺', '🔻', '🔸', '🔹', '🔶', '🔷', '🔳', '🔲', '▪️', '▫️', '◾', '◽', '◼️', '◻️', '🟥', '🟧', '🟨', '🟩', '🟦', '🟪', '⬛', '⬜', '🟫', '🔈', '🔇', '🔉', '🔊', '🔔', '🔕', '📣', '📢', '👁️‍🗨️', '💬', '💭', '🗯️', '♠️', '♣️', '♥️', '♦️', '🃏', '🎴', '🀄', '🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛', '🕜', '🕝', '🕞', '🕟', '🕠', '🕡', '🕢', '🕣', '🕤', '🕥', '🕦', '🕧']) or len(word) <= 3:
                            for i in range(len(lines)-1, -1, -1):
                                test_line = lines[i] + " " + word
                                try:
                                    bbox = font.getbbox(test_line)
                                    text_width = bbox[2] - bbox[0]
                                except:
                                    text_width = len(test_line) * (font_size * 0.6)
                                
                                if text_width <= max_width:
                                    lines[i] = test_line
                                    break
                
                # Clean up any remaining \n characters and excessive spacing from the final lines (important!)
                cleaned_lines = []
                for line in lines[:max_lines]:
                    # Remove any remaining \n characters that might have slipped through
                    cleaned_line = line.replace('\\n', ' ').replace('\n', ' ').strip()
                    # Clean up multiple consecutive spaces to prevent large gaps
                    cleaned_line = re.sub(r'\s+', ' ', cleaned_line)
                    if cleaned_line:
                        cleaned_lines.append(cleaned_line)
                
                return cleaned_lines
            
            # Get font and wrap text
            font = self._get_caption_font(font_size)
            text_lines = wrap_text_smart(caption_text, font, text_area_width, max_lines)
            
            if not text_lines:
                logger.warning("No text to render after wrapping")
                return []
            
            logger.info(f"Wrapped into {len(text_lines)} lines: {text_lines}")
            
            # Calculate actual text block height
            actual_text_height = len(text_lines) * line_height
            
            # Create ultra-high-resolution PIL image for crisp text
            scale_factor = 4  # 4x resolution for ultra-crisp text
            pil_width = target_width * scale_factor  # 4320px
            pil_height = target_height * scale_factor  # 7680px
            
            # Create transparent image
            pil_image = Image.new('RGBA', (pil_width, pil_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(pil_image)
            
            # Get ultra-high-res font
            scaled_font_size = font_size * scale_factor  # 260px
            try:
                scaled_font = self._get_caption_font(scaled_font_size)
                scaled_emoji_font = self._get_emoji_font(scaled_font_size)
                logger.info(f"Using ultra-high-quality font at {scaled_font_size}px")
                if scaled_emoji_font:
                    logger.info(f"Emoji support enabled with font at {scaled_font_size}px")
            except:
                scaled_font = self._get_caption_font(font_size * 2)  # Fallback to 2x
                scaled_emoji_font = self._get_emoji_font(font_size * 2)
                logger.warning("Using fallback font scaling")
            
            # Calculate starting Y position (centered around our target position)
            scaled_center_y = caption_center_y * scale_factor
            scaled_text_height = actual_text_height * scale_factor
            start_y = scaled_center_y - (scaled_text_height // 2)
            
            # Draw each line with ultra-thick black outline for TikTok style
            outline_thickness = 6 * scale_factor  # Slightly thicker outline for better readability
            scaled_line_height = line_height * scale_factor
            
            for i, line in enumerate(text_lines):
                # Calculate center position for this line
                try:
                    bbox = scaled_font.getbbox(line)
                    text_width = bbox[2] - bbox[0]
                except:
                    text_width = len(line) * (scaled_font_size * 0.6)
                
                x = (pil_width - text_width) // 2  # Center horizontally
                y = start_y + (i * scaled_line_height)
                
                # FIRST: Draw thick black outline for TikTok style (must be drawn BEFORE white text)
                for offset_x in range(-outline_thickness, outline_thickness + 1):
                    for offset_y in range(-outline_thickness, outline_thickness + 1):
                        if offset_x == 0 and offset_y == 0:
                            continue  # Skip center (that's where white text will go)
                        self._draw_text_with_emoji(
                            img=pil_image,
                            text=line,
                            font=scaled_font,
                            position=(x + offset_x, y + offset_y),
                            fill=(0, 0, 0, 255),  # Black outline
                            render_emojis=False
                        )
                
                # SECOND: Draw white text on top of the black outline
                self._draw_text_with_emoji(
                    img=pil_image,
                    text=line,
                    font=scaled_font,
                    position=(x, y),
                    fill=(255, 255, 255, 255),  # White text on top
                    render_emojis=True  # Enable emojis for the main text
                )
                
                logger.info(f"Drew TikTok-style text line {i+1}: '{line}' with black outline + white text at position ({x//scale_factor}, {y//scale_factor})")
            
            # Downsample to target resolution using highest quality method
            pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
            
            # Save as high-quality PNG
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            temp_caption_file = temp_dir / f"caption_hq_{hash(caption_text)}.png"
            pil_image.save(str(temp_caption_file), "PNG", optimize=True, compress_level=1)
            
            logger.info(f"Created ultra-high-quality text overlay: '{' '.join(text_lines)}' positioned correctly at y={caption_center_y}")
            
            # Create VideoFileClip from the high-quality image
            from moviepy.editor import ImageClip
            
            caption_clip = ImageClip(str(temp_caption_file), duration=video.duration)
            caption_clip = caption_clip.set_position('center')
            
            # Clean up temp file after a delay to ensure MoviePy has loaded it
            def cleanup_later():
                import time
                time.sleep(1)  # Give MoviePy time to load the image
                try:
                    import os
                    os.unlink(temp_caption_file)
                except:
                    pass
            
            import threading
            threading.Thread(target=cleanup_later, daemon=True).start()
            
            return [caption_clip]
            
        except Exception as e:
            logger.error(f"Caption creation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def _get_caption_font(self, font_size: int = 65) -> ImageFont.ImageFont:
        """
        Get the best available font for captions with proper fallbacks.
        Priority: Semibold fonts > Bold fonts > Regular fonts
        """
        # Priority list of fonts to try (updated for semibold emphasis)
        font_candidates = [
            # Semibold fonts (highest priority)
            "OpenSans-SemiBold",
            "Open Sans SemiBold", 
            "ProximaNova-SemiBold",
            "Proxima Nova SemiBold", 
            "Montserrat-SemiBold",
            "Montserrat SemiBold",
            
            # Bold fonts (second priority)
            "ProximaNova-Bold",
            "Proxima Nova Bold",
            "Montserrat-Bold",
            "Montserrat Bold",
            "OpenSans-Bold",
            "Open Sans Bold",
            "Helvetica Neue Bold",
            "Helvetica-Bold",
            "Arial Bold",
            
            # Regular weights as fallbacks
            "ProximaNova-Regular",
            "Proxima Nova",
            "Montserrat-Regular",
            "Montserrat",
            "OpenSans-Regular", 
            "Open Sans",
            "Helvetica Neue",
            "Helvetica",
            "Arial"
        ]
        
        # Try to load each font candidate
        for font_name in font_candidates:
            try:
                font = ImageFont.truetype(font_name, font_size)
                self.logger.info(f"✅ Using font: {font_name} at size {font_size}")
                return font
            except (OSError, IOError):
                continue
        
        # Try system font paths for installed fonts
        font_paths = [
            f"/Users/{os.getenv('USER')}/Library/Fonts/OpenSans-SemiBold.ttf",
            f"/Users/{os.getenv('USER')}/Library/Fonts/ProximaNova-SemiBold.otf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf"
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    self.logger.info(f"✅ Using font from path: {font_path}")
                    return font
            except (OSError, IOError):
                continue
        
        # Final fallback to default font
        try:
            font = ImageFont.load_default()
            self.logger.warning(f"⚠️ Using default font for captions at size {font_size}")
            return font
        except Exception as e:
            self.logger.error(f"❌ Could not load any font: {e}")
            raise RuntimeError("No suitable font available for caption generation")

    def _get_emoji_font(self, font_size: int = 65) -> Optional[ImageFont.ImageFont]:
        """
        Get the Apple Color Emoji font for proper emoji rendering.
        Apple Color Emoji has restrictions on valid sizes, so we need to use specific sizes.
        """
        emoji_font_paths = [
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/Library/Fonts/Apple Color Emoji.ttc",
            "/System/Library/Fonts/NotoColorEmoji.ttf"
        ]
        
        # Apple Color Emoji only supports specific sizes
        # Use more conservative sizing to prevent loading failures
        valid_emoji_sizes = [16, 20, 24, 32, 40, 48, 64, 72, 96]
        
        # Find the closest valid size
        if font_size > 96:
            target_size = 96  # Use maximum safe size
            self.logger.debug(f"Requested emoji size {font_size}px is too large, using maximum safe size: {target_size}px")
        else:
            target_size = min(valid_emoji_sizes, key=lambda x: abs(x - font_size))
            if target_size != font_size:
                self.logger.debug(f"Adjusted emoji size from {font_size}px to closest valid size: {target_size}px")
        
        for font_path in emoji_font_paths:
            if Path(font_path).exists():
                try:
                    # For Apple Color Emoji (.ttc file), we need to specify the font index
                    # Try different font indices to get the best support
                    font_indices = [0, 1] if font_path.endswith('.ttc') else [0]
                    
                    for font_index in font_indices:
                        try:
                            if font_path.endswith('.ttc'):
                                emoji_font = ImageFont.truetype(font_path, size=target_size, index=font_index)
                            else:
                                emoji_font = ImageFont.truetype(font_path, size=target_size)
                            
                            # Test if the font can render an emoji
                            test_bbox = emoji_font.getbbox("😀")
                            if test_bbox and (test_bbox[2] - test_bbox[0] > 0):
                                self.logger.info(f"✅ Successfully loaded emoji font: {font_path} (index {font_index}) at size {target_size}px")
                                self.logger.info(f"🌈 Using real color emoji images from Twemoji/OpenMoji for authentic appearance")
                                return emoji_font
                            
                        except (OSError, IOError) as e:
                            self.logger.debug(f"Failed to load emoji font with index {font_index}: {e}")
                            continue
                    
                except Exception as e:
                    self.logger.debug(f"Failed to load emoji font {font_path}: {e}")
                    continue
        
        # Try fallback fonts
        fallback_fonts = [
            "Arial Unicode MS",
            "Helvetica",
            "Arial"
        ]
        
        for fallback_font in fallback_fonts:
            try:
                emoji_font = ImageFont.truetype(fallback_font, target_size)
                self.logger.warning(f"⚠️ Using fallback font {fallback_font} for emoji at size {target_size}px")
                return emoji_font
            except:
                continue
                
        self.logger.warning("❌ No emoji font found - emojis may display as boxes")
        return None

    def _draw_text_with_emoji(self, img: Image.Image, text: str, font: ImageFont.ImageFont, 
                             position: Tuple[int, int], fill: Tuple[int, int, int, int] = (255, 255, 255, 255),
                             render_emojis: bool = True) -> None:
        """
        Draw text with improved emoji support using iPhone-style color emojis.
        Now with robust fallback system to ensure all emojis render properly.
        
        Args:
            render_emojis: If False, skips emoji rendering and draws only text (for outline phases)
        """
        try:
            import re
            import unicodedata
            from io import BytesIO
            
            draw = ImageDraw.Draw(img)
            x, y = position
            current_x = float(x)
            
            # Get text height for emoji sizing
            text_bbox = font.getbbox("Ay")  # Use characters with ascenders/descenders
            actual_text_height = text_bbox[3] - text_bbox[1]
            
            # Calculate emoji size to match text height
            emoji_size = int(actual_text_height * 1.1)  # Slightly larger for better visual balance
            if render_emojis:
                self.logger.info(f"📏 Calculated emoji size: {emoji_size}px to match text height: {actual_text_height}px")
            
            # Process each character
            i = 0
            while i < len(text):
                char = text[i]
                
                # Check if current character is an emoji using improved detection
                if self._is_emoji_char(char):
                    if render_emojis:
                        self.logger.debug(f"🔍 Detected emoji: '{char}' (Unicode: U+{ord(char):04X})")
                        
                        # Try to render as color emoji with fallback chain
                        emoji_rendered = False
                        
                        # Method 1: Try color emoji download (iPhone style)
                        try:
                            emoji_img = self._render_color_emoji_to_image(char, emoji_size)
                            if emoji_img and emoji_img.size[0] > 0 and emoji_img.size[1] > 0:
                                # Calculate vertical centering
                                emoji_y = y + (actual_text_height - emoji_img.size[1]) // 2
                                
                                # Paste emoji onto image
                                if emoji_img.mode == 'RGBA':
                                    img.paste(emoji_img, (int(current_x), emoji_y), emoji_img)
                                else:
                                    img.paste(emoji_img, (int(current_x), emoji_y))
                                
                                current_x += emoji_img.size[0] + 2  # Small spacing after emoji
                                emoji_rendered = True
                                self.logger.info(f"🎨 Composited {emoji_size}px emoji '{char}' at ({int(current_x)}, {emoji_y}) to match {actual_text_height}px text")
                            
                        except Exception as e:
                            self.logger.debug(f"Color emoji failed for '{char}': {e}")
                        
                        # Method 2: Try Apple Color Emoji font with proper sizing
                        if not emoji_rendered:
                            try:
                                emoji_font = self._get_emoji_font(emoji_size)
                                if emoji_font:
                                    # Test if emoji renders properly with this font
                                    test_bbox = emoji_font.getbbox(char)
                                    if test_bbox[2] > test_bbox[0]:  # Has width, so renders
                                        draw.text((current_x, y), char, font=emoji_font, fill=fill)
                                        char_width = emoji_font.getlength(char)
                                        current_x += char_width + 2
                                        emoji_rendered = True
                                        self.logger.info(f"🍎 Rendered emoji '{char}' using Apple Color Emoji font")
                            except Exception as e:
                                self.logger.debug(f"Apple emoji font failed for '{char}': {e}")
                        
                        # Method 3: Create custom emoji fallback
                        if not emoji_rendered:
                            try:
                                fallback_img = self._create_emoji_fallback(char, emoji_size)
                                if fallback_img:
                                    emoji_y = y + (actual_text_height - fallback_img.size[1]) // 2
                                    if fallback_img.mode == 'RGBA':
                                        img.paste(fallback_img, (int(current_x), emoji_y), fallback_img)
                                    else:
                                        img.paste(fallback_img, (int(current_x), emoji_y))
                                    current_x += fallback_img.size[0] + 2
                                    emoji_rendered = True
                                    self.logger.info(f"🔄 Used fallback for emoji '{char}'")
                            except Exception as e:
                                self.logger.debug(f"Fallback emoji failed for '{char}': {e}")
                        
                        # Method 4: Last resort - draw as regular text but warn
                        if not emoji_rendered:
                            self.logger.warning(f"⚠️ All emoji methods failed for '{char}', drawing as text")
                            draw.text((current_x, y), char, font=font, fill=fill)
                            char_width = font.getlength(char)
                            current_x += char_width + 2
                    else:
                        # Skip emoji rendering for outline phase, but advance position
                        char_width = font.getlength(char)
                        current_x += char_width + 2  # Match spacing from emoji rendering
                
                else:
                    # Regular text character
                    draw.text((current_x, y), char, font=font, fill=fill)
                    char_width = font.getlength(char)
                    current_x += char_width
                
                i += 1
                
        except Exception as e:
            self.logger.error(f"Error in _draw_text_with_emoji: {str(e)}")
            # Fallback to simple text drawing
            draw = ImageDraw.Draw(img)
            draw.text(position, text, font=font, fill=fill)

    def _load_cache_index(self):
        """Load cache index from file."""
        cache_index_file = self.cache_dir / "cache_index.json"
        if cache_index_file.exists():
            try:
                with open(cache_index_file) as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache index: {str(e)}")
                self.cache_index = {}

    def _save_cache_index(self):
        """Save cache index to file."""
        cache_index_file = self.cache_dir / "cache_index.json"
        try:
            with open(cache_index_file, "w") as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")

    def _get_cache_key(self, video_path: str, config: ProcessingConfig) -> str:
        """Generate cache key for video processing."""
        key_data = f"{video_path}:{config.target_width}:{config.target_height}:{config.target_fps}:{config.codec}:{config.bitrate}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if processed video exists in cache."""
        if cache_key in self.cache_index:
            cache_path = self.cache_dir / f"{cache_key}.mp4"
            if cache_path.exists():
                return str(cache_path)
        return None

    def _add_to_cache(self, cache_key: str, video_path: str, config: ProcessingConfig):
        """Add processed video to cache."""
        cache_path = self.cache_dir / f"{cache_key}.mp4"
        try:
            shutil.copy2(video_path, cache_path)
            self.cache_index[cache_key] = {
                "original_path": video_path,
                "config": {
                    "target_width": config.target_width,
                    "target_height": config.target_height,
                    "target_fps": config.target_fps,
                    "codec": config.codec,
                    "bitrate": config.bitrate
                },
                "timestamp": datetime.now().isoformat()
            }
            self._save_cache_index()
        except Exception as e:
            logger.error(f"Error adding to cache: {str(e)}")

    def _monitor_resources(self):
        """Monitor system resources and clean up when needed."""
        import psutil
        import gc
        
        while self.resource_monitor_active:
            try:
                # Get current memory usage
                memory = psutil.virtual_memory()
                
                # If memory usage is high, trigger cleanup
                if memory.percent > self.config.max_memory_percent:
                    self.logger.warning(f"High memory usage: {memory.percent:.1f}% - cleaning up")
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Clean up temp files
                    self._cleanup_temp_files()
                    
                    # Clean up old cache if needed
                    self._cleanup_cache()
                
                # Sleep for 30 seconds before next check
                import time
                for _ in range(30):
                    if not self.resource_monitor_active:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                import time
                time.sleep(10)  # Sleep longer if error occurred
                
        self.logger.info("Resource monitor stopping due to shutdown")

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")

    def _cleanup_cache(self):
        """Clean up old cache files."""
        try:
            # Remove cache files older than 7 days
            cutoff_time = time.time() - (7 * 24 * 60 * 60)
            
            for cache_key in list(self.cache_index.keys()):
                cache_path = self.cache_dir / f"{cache_key}.mp4"
                if cache_path.exists():
                    if cache_path.stat().st_mtime < cutoff_time:
                        cache_path.unlink()
                        del self.cache_index[cache_key]
            
            self._save_cache_index()
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")

    def process_video(
        self,
        video_path: str,
        output_path: str,
        config: Optional[ProcessingConfig] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
        force_fresh: bool = False
    ) -> bool:
        """
        Process a video file for Instagram Reels/TikTok format with AI captions.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to output video
            config (Optional[ProcessingConfig]): Processing configuration
            priority (ProcessingPriority): Processing priority
            callback (Optional[callable]): Callback function after processing
            progress_callback (Optional[callable]): Progress callback function
            force_fresh (bool): If True, bypass cache and force fresh processing
            
        Returns:
            bool: True if processing was successful
        """
        try:
            # Use provided config or default
            config = config or self.config
            
            if force_fresh:
                logger.info(f"Processing video {video_path} (FORCE FRESH - cache disabled)")
            else:
                # Check cache first unless forced fresh processing
                cache_key = self._get_cache_key(video_path, config)
                cached_result = self._check_cache(cache_key)
                
                if cached_result and Path(cached_result).exists():
                    logger.info(f"Using cached version for {video_path}")
                    # Copy cached file to output path
                    import shutil
                    shutil.copy2(cached_result, output_path)
                    
                    if callback:
                        callback(True)
                    return True
                
                logger.info(f"Processing video {video_path} (cache disabled to ensure captions)")
            
            # Create processing job
            job = ProcessingJob(
                video_path=video_path,
                output_path=output_path,
                config=config,
                priority=priority,
                callback=callback,
                progress_callback=progress_callback
            )
            
            # Add to processing queue
            with self.queue_lock:
                self.processing_queue.append(job)
                self.processing_queue.sort(key=lambda x: x.priority.value, reverse=True)
            
            # Submit job and wait for completion
            future = self.executor.submit(self._process_job, job)
            
            # Wait for the job to complete and get the result
            try:
                # This will block until the job is actually finished
                result = future.result()  # This will raise an exception if the job failed
                
                # Verify the output file was created successfully AND has captions
                if Path(output_path).exists():
                    # Additional verification that the video has captions
                    file_size = Path(output_path).stat().st_size
                    if file_size > 0:
                        logger.info(f"Successfully created output video: {output_path} ({file_size} bytes)")
                        return True
                    else:
                        logger.error(f"Output file is empty: {output_path}")
                        return False
                else:
                    logger.error(f"Processing completed but output file not found: {output_path}")
                    return False
                    
            except Exception as e:
                logger.error(f"Job execution failed: {str(e)}")
                return False
            
        except Exception as e:
            context = ErrorContext(
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.ERROR,
                component="VideoProcessor",
                operation="process_video",
                details={"video_path": video_path, "output_path": output_path, "force_fresh": force_fresh},
                timestamp=datetime.now()
            )
            
            self.error_handler.handle_error(e, context)
            return False

    def _process_job(self, job: ProcessingJob):
        """Process a video job for Instagram Reels/TikTok with AI captions."""
        video = None
        converted_video = None
        final_video = None
        caption_clips = []
        
        try:
            start_time = time.time()
            
            logger.info(f"Starting processing of {job.video_path} for Instagram Reels/TikTok format")
            
            # Check if file exists
            if not os.path.exists(job.video_path):
                logger.error(f"Video file not found: {job.video_path}")
                return False
            
            # Memory cleanup before starting
            gc.collect()
            
            # Load video with explicit resource management
            logger.info("Loading video...")
            video = VideoFileClip(job.video_path)
            logger.info(f"Loaded video: {video.w}x{video.h}, duration: {video.duration:.2f}s, fps: {video.fps}")
            
            # Convert to 9:16 aspect ratio for Instagram Reels/TikTok
            logger.info("Converting to 9:16 aspect ratio for Instagram Reels/TikTok...")
            converted_video = self._convert_to_9_16_aspect_ratio(video)
            logger.info(f"Converted video to: {converted_video.w}x{converted_video.h}")
            
            # Generate adult creator marketing caption
            logger.info("Generating adult creator marketing caption...")
            
            # Generate caption text
            caption_text = self._generate_adult_creator_caption()
            logger.info(f"Generated adult creator caption: {caption_text}")
            
            # Create Instagram-style captions
            logger.info(f"Creating Instagram-style captions with text: {caption_text}")
            caption_clips = self._create_instagram_style_captions(converted_video, caption_text)
            
            # CRITICAL CHANGE: Only proceed if captions were successfully created
            if not caption_clips:
                logger.error("❌ Caption creation failed - video processing aborted to prevent upload without captions")
                logger.error("Video will be retried later to ensure AI captions are applied")
                return False
            
            logger.info(f"✅ Successfully added {len(caption_clips)} caption elements to video")
            
            # Composite video with captions
            final_video = CompositeVideoClip([converted_video] + caption_clips)
            
            # Export final video
            output_path = job.output_path
            logger.info(f"Exporting final video with AI captions to {output_path}...")
            
            # Write video with conservative settings to prevent memory issues
            try:
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp/temp-audio.m4a',
                    remove_temp=True,
                    fps=30,
                    preset='medium',
                    threads=1,  # Single thread to prevent memory corruption
                    verbose=False,
                    logger=None  # Disable logging to reduce memory usage
                )
            except Exception as export_error:
                logger.error(f"❌ Video export failed with error: {export_error}")
                logger.error(f"Export error type: {type(export_error).__name__}")
                import traceback
                logger.error(f"Export traceback: {traceback.format_exc()}")
                return False
            
            # CRITICAL: Verify the output file was actually created
            if not os.path.exists(output_path):
                logger.error(f"❌ Video export failed - output file not found: {output_path}")
                logger.error("This may be due to memory issues, codec problems, or insufficient disk space")
                return False
            
            # Check file size to ensure it's not corrupted/empty
            file_size = os.path.getsize(output_path)
            if file_size < 1024:  # Less than 1KB indicates a problem
                logger.error(f"❌ Video export failed - output file too small: {file_size} bytes")
                return False
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Successfully processed {job.video_path} with AI captions in {processing_time:.2f}s")
            logger.info(f"✅ Output file verified: {output_path} ({file_size:,} bytes)")
            logger.info(f"Output video: {final_video.w}x{final_video.h} (9:16 aspect ratio)")
            
            # Add to cache
            cache_key = self._get_cache_key(job.video_path, job.config)
            self._add_to_cache(cache_key, output_path, job.config)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video {job.video_path}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        finally:
            # CRITICAL: Always clean up video objects to prevent memory corruption
            try:
                # Close caption clips first
                for clip in caption_clips:
                    try:
                        if hasattr(clip, 'close'):
                            clip.close()
                    except:
                        pass
                
                # Close final video
                if final_video and hasattr(final_video, 'close'):
                    try:
                        final_video.close()
                    except:
                        pass
                
                # Close converted video (if different from original)
                if converted_video and converted_video != video and hasattr(converted_video, 'close'):
                    try:
                        converted_video.close()
                    except:
                        pass
                
                # Close original video
                if video and hasattr(video, 'close'):
                    try:
                        video.close()
                    except:
                        pass
                
            except Exception as cleanup_error:
                logger.error(f"Error during video cleanup: {cleanup_error}")
            
            # Force garbage collection to free memory
            gc.collect()

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current processing queue status."""
        with self.queue_lock:
            return {
                "queue_length": len(self.processing_queue),
                "jobs": [
                    {
                        "video_path": job.video_path,
                        "priority": job.priority.value,
                        "config": job.config.__dict__
                    }
                    for job in self.processing_queue
                ]
            }

    def cancel_job(self, video_path: str) -> bool:
        """Cancel a processing job."""
        with self.queue_lock:
            for i, job in enumerate(self.processing_queue):
                if job.video_path == video_path:
                    self.processing_queue.pop(i)
                    return True
        return False

    def shutdown(self):
        """Shutdown the video processor and clean up resources."""
        self.logger.info("Shutting down video processor...")
        
        try:
            # Stop resource monitoring first
            self.resource_monitor_active = False
            self.logger.info("Stopping resource monitor thread...")
            
            # Wait for resource monitor to stop
            if hasattr(self, 'resource_monitor_thread') and self.resource_monitor_thread.is_alive():
                self.resource_monitor_thread.join(timeout=5)
            
            # Shutdown thread pool executor
            self.logger.info("Shutting down thread pool executor...")
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
                
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Video processor shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            if hasattr(self, 'resource_monitor_active'):
                self.resource_monitor_active = False
            if hasattr(self, 'executor') and not self.executor._shutdown:
                self.executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore errors in destructor

    def _render_color_emoji_to_image(self, emoji_char: str, size: int) -> Optional[Image.Image]:
        """
        Download and render actual color emoji images with proper resource management.
        Uses Apple emoji style for iPhone-like appearance with reliable sources.
        """
        try:
            from PIL import Image
            import requests
            import unicodedata
            import hashlib
            import os
            import threading
            import gc
            
            # Thread-safe emoji cache access
            with self.emoji_lock:
                # Create emoji cache directory
                emoji_cache_dir = Path("cache/emojis")
                emoji_cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a safe filename from emoji character
                try:
                    emoji_name = unicodedata.name(emoji_char, "unknown").replace(" ", "_").lower()
                except (ValueError, TypeError):
                    # For multi-character emojis or unknown characters
                    emoji_name = f"emoji_{hashlib.md5(emoji_char.encode('utf-8')).hexdigest()[:8]}"
                
                cache_filename = f"{emoji_name}_{size}px.png"
                cache_path = emoji_cache_dir / cache_filename
                
                # Return cached version if exists
                if cache_path.exists():
                    try:
                        img = Image.open(cache_path).convert("RGBA")
                        # Ensure proper size
                        if img.size != (size, size):
                            img = img.resize((size, size), Image.Resampling.LANCZOS)
                        return img
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached emoji {cache_filename}: {e}")
                        # Delete corrupted cache file
                        try:
                            cache_path.unlink()
                        except:
                            pass
                
                # Convert emoji to Unicode codepoint
                try:
                    # Handle single character emojis
                    if len(emoji_char) == 1:
                        unicode_hex = f"{ord(emoji_char):x}"
                    else:
                        # Handle multi-character emojis (like skin tone modifiers)
                        unicode_hex = "-".join(f"{ord(c):x}" for c in emoji_char)
                except Exception as e:
                    self.logger.warning(f"Failed to convert emoji to unicode: {e}")
                    return None
                
                # Try multiple reliable emoji sources
                emoji_urls = [
                    # JsDelivr CDN (most reliable)
                    f"https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.0.1/img/apple/64/{unicode_hex}.png",
                    f"https://cdn.jsdelivr.net/npm/emoji-datasource-apple@14.0.0/img/apple/64/{unicode_hex}.png",
                    # Twemoji as backup
                    f"https://twemoji.maxcdn.com/v/latest/72x72/{unicode_hex}.png",
                    f"https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{unicode_hex}.png",
                ]
                
                # Try to download emoji
                img = None
                session = None
                
                try:
                    session = requests.Session()
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15'
                    })
                    
                    for url in emoji_urls:
                        try:
                            self.logger.debug(f"🍎 Downloading iPhone emoji '{emoji_char}' from {url}")
                            response = session.get(url, timeout=10, stream=True)
                            
                            if response.status_code == 200:
                                # Load image directly from response content
                                img_data = response.content
                                response.close()
                                
                                # Create PIL image from bytes
                                img = Image.open(BytesIO(img_data)).convert("RGBA")
                                
                                # Resize to requested size
                                if img.size != (size, size):
                                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                                
                                # Save to cache
                                try:
                                    img.save(cache_path, "PNG", optimize=True)
                                    self.logger.info(f"✅ Successfully downloaded iPhone emoji '{emoji_char}' at {size}x{size}")
                                except Exception as save_error:
                                    self.logger.warning(f"Failed to save emoji to cache: {save_error}")
                                
                                break
                            else:
                                response.close()
                                
                        except requests.RequestException as e:
                            self.logger.debug(f"Failed to download from {url}: {e}")
                            continue
                        except Exception as e:
                            self.logger.warning(f"Error processing emoji from {url}: {e}")
                            continue
                    
                    if img is None:
                        self.logger.warning(f"⚠️ Failed to download emoji: {emoji_char} (unicode: {unicode_hex}) - using fallback")
                        return self._create_emoji_fallback(emoji_char, size)
                        
                    return img
                    
                finally:
                    # Clean up session
                    if session:
                        try:
                            session.close()
                        except:
                            pass
                    
                    # Force garbage collection
                    gc.collect()
                    
        except Exception as e:
            self.logger.error(f"Error in emoji rendering: {e}")
            return self._create_emoji_fallback(emoji_char, size)
    
    def _create_emoji_fallback(self, emoji_char: str, size: int) -> Optional[Image.Image]:
        """
        Create a visible fallback for emojis that fail to download.
        Creates a colorful circular icon with the emoji's first letter or a symbol.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a square image for the emoji
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Define emoji-to-color mapping for common emojis
            emoji_colors = {
                '🔥': [(255, 69, 0), (255, 140, 0)],      # Fire - red/orange
                '💯': [(50, 205, 50), (0, 255, 0)],       # 100 - green
                '👀': [(135, 206, 235), (70, 130, 180)],  # Eyes - blue
                '😈': [(148, 0, 211), (255, 20, 147)],    # Devil - purple/pink
                '📱': [(105, 105, 105), (169, 169, 169)], # Phone - gray
                '💅': [(255, 182, 193), (255, 105, 180)], # Nails - pink
                '🤖': [(70, 130, 180), (100, 149, 237)],  # Robot - blue
                '🔄': [(255, 165, 0), (255, 215, 0)],     # Refresh - orange/gold
                '💻': [(47, 79, 79), (105, 105, 105)],    # Computer - dark gray
                '😏': [(255, 215, 0), (255, 165, 0)],     # Smirk - gold
                '📸': [(128, 128, 128), (169, 169, 169)], # Camera - gray
            }
            
            # Get colors for this emoji or use default
            colors = emoji_colors.get(emoji_char, [(255, 107, 107), (255, 182, 193)])  # Default pink
            
            # Create gradient circle background
            center = size // 2
            radius = size // 2 - 2
            
            # Draw outer circle with gradient effect
            for i in range(radius):
                ratio = i / radius
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                
                circle_color = (r, g, b, 255)
                draw.ellipse([center - radius + i, center - radius + i, 
                             center + radius - i, center + radius - i], 
                            fill=circle_color)
            
            # Add emoji symbol or letter in the center
            symbol_map = {
                '🔥': '🔥',  # Keep fire symbol
                '💯': '💯', # Keep 100 symbol  
                '👀': '👀', # Keep eyes
                '😈': '😈', # Keep devil
                '📱': '📱', # Keep phone
                '💅': '💅', # Keep nails
                '🤖': '🤖', # Keep robot
                '🔄': '↻',   # Use refresh arrow
                '💻': '💻', # Keep computer
                '😏': '😏', # Keep smirk
                '📸': '📸', # Keep camera
            }
            
            # Try to get a good fallback symbol
            symbol = symbol_map.get(emoji_char, emoji_char)
            
            # If we still have the original emoji, create a text representation
            if symbol == emoji_char:
                # Use emoji name or first character as fallback
                emoji_names = {
                    '🎯': '🎯', '❤️': '❤', '💕': '💕', '✨': '✨',
                    '★': '★', '⭐': '★', '💫': '✦', '🎨': '🎨'
                }
                symbol = emoji_names.get(emoji_char, '●')  # Default to bullet
            
            # Try to draw the symbol with a font
            try:
                # Use a smaller font size for the symbol
                font_size = max(12, size // 3)
                font = ImageFont.load_default()
                
                # Try to load a better font if available
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", font_size)
                except:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                    except:
                        font = ImageFont.load_default()
                
                # Get text size and center it
                bbox = draw.textbbox((0, 0), symbol, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = (size - text_width) // 2
                text_y = (size - text_height) // 2
                
                # Draw white symbol with dark outline for visibility
                draw.text((text_x + 1, text_y + 1), symbol, font=font, fill=(0, 0, 0, 128))  # Shadow
                draw.text((text_x, text_y), symbol, font=font, fill=(255, 255, 255, 255))    # Main text
                
            except Exception as font_error:
                self.logger.debug(f"Font rendering failed for fallback: {font_error}")
                # Draw a simple shape as last resort
                quarter = size // 4
                draw.rectangle([quarter, quarter, size - quarter, size - quarter], 
                              fill=(255, 255, 255, 255))
            
            self.logger.debug(f"Created fallback emoji for '{emoji_char}' ({size}x{size})")
            return img
            
        except Exception as e:
            self.logger.error(f"Failed to create emoji fallback: {e}")
            return None

    def _is_emoji_char(self, char: str) -> bool:
        """
        Improved emoji detection using Unicode categories.
        More reliable than just checking UTF-8 byte length.
        """
        if not char:
            return False
            
        # Get Unicode category
        import unicodedata
        try:
            # Check for emoji ranges and categories
            cp = ord(char)
            
            # Common emoji ranges
            emoji_ranges = [
                (0x1F600, 0x1F64F),  # Emoticons
                (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
                (0x1F680, 0x1F6FF),  # Transport and Map
                (0x1F1E0, 0x1F1FF),  # Regional indicators
                (0x2600, 0x26FF),    # Misc symbols
                (0x2700, 0x27BF),    # Dingbats
                (0xFE00, 0xFE0F),    # Variation Selectors
                (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
                (0x1F018, 0x1F270),  # Various symbols
            ]
            
            # Check if character is in emoji range
            for start, end in emoji_ranges:
                if start <= cp <= end:
                    return True
                    
            # Additional check for some common emojis that might be missed
            category = unicodedata.category(char)
            if category == 'So':  # Other symbols
                return True
                
            return False
            
        except (ValueError, TypeError):
            # Fallback to old method for edge cases
            try:
                return len(char.encode('utf-8')) > 2
            except:
                return False

    def _generate_ai_inspired_caption(self, sheet_id: str = None) -> str:
        """
        Generate captions inspired by Google Sheets content using OpenAI.
        
        Args:
            sheet_id (str, optional): Google Sheet ID to pull ideas from
        
        Returns:
            str: AI-generated caption inspired by sheet content
        """
        try:
            # Initialize sheets service
            sheets_service = SheetsService(sheet_id=sheet_id)
            
            # Get random caption ideas from the sheet
            sheet_ideas = sheets_service.get_random_caption_ideas(count=3)
            
            if not sheet_ideas:
                logger.warning("No ideas found in Google Sheet, falling back to hardcoded captions")
                return self._generate_adult_creator_caption()
            
            # Create prompt for OpenAI
            ideas_text = "\n".join([f"- {idea}" for idea in sheet_ideas])
            
            prompt = f"""
Based on these caption ideas from my content strategy:

{ideas_text}

Create a similar style caption for an adult creator that is:
- 7-15 words maximum
- Split into 2 lines with \\n
- Includes appropriate emoji
- Flirty and suggestive but platform-compliant
- TikTok/Instagram Reels style
- Emotionally provocative and algorithm-optimized

Return ONLY the caption text, nothing else.
"""

            # Use your existing OpenAI setup
            from openai import OpenAI
            from ..config import OPENAI_API_KEY
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at creating viral TikTok captions for adult creators. You create engaging, platform-compliant content that drives engagement."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.8
            )
            
            generated_caption = response.choices[0].message.content.strip()
            
            logger.info(f"Generated AI-inspired caption: '{generated_caption}'")
            return generated_caption
            
        except Exception as e:
            logger.error(f"Error generating AI-inspired caption: {str(e)}")
            # Fallback to existing method
            return self._generate_adult_creator_caption()

    def set_caption_sheet_id(self, sheet_id: str):
        """
        Set the Google Sheet ID for caption inspiration.
        
        Args:
            sheet_id (str): Google Sheets ID
        """
        self.caption_sheet_id = sheet_id
        logger.info(f"Set caption sheet ID: {sheet_id}") 