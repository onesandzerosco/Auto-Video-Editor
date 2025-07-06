import time
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set
from ..utils.logger import logger
from .drive_service import DriveService
from .video_processor import VideoProcessor
from .error_handler import ErrorHandler
import json
import hashlib

class VideoMonitor:
    def __init__(self, drive_service: DriveService, poll_interval: int = 30, max_retries: int = 3):
        """
        Initialize the video monitor with queuing and retry capabilities.
        
        Args:
            drive_service (DriveService): Initialized DriveService instance
            poll_interval (int): Interval in seconds between folder checks
            max_retries (int): Maximum number of retry attempts for failed operations
        """
        self.drive_service = drive_service
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.is_running = False
        self.processing_queue = Queue()
        self.workers = []
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Persistent file tracking
        self.state_file = Path("data/processing_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        
        # File tracking sets and mappings
        self.processed_files = set()        # Successfully processed file IDs (persistent)
        self.processed_signatures = {}      # File ID -> signature mapping (persistent)
        self.processing_files = set()       # Currently being processed (memory only)
        self.failed_files = {}              # Failed files with retry count and timestamp (persistent)
        self.permanently_failed = set()     # Files that exceeded max retries (persistent)
        
        # Load persistent state
        self._load_state()
        
        # Initialize error handler first
        self.error_handler = ErrorHandler()
        
        # Initialize video processor with error handler
        self.video_processor = VideoProcessor(self.error_handler)

    def _load_state(self):
        """Load persistent state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.processed_files = set(state.get('processed_files', []))
                    self.processed_signatures = state.get('processed_signatures', {})
                    self.failed_files = state.get('failed_files', {})
                    self.permanently_failed = set(state.get('permanently_failed', []))
                    logger.info(f"Loaded state: {len(self.processed_files)} processed, "
                              f"{len(self.failed_files)} failed, "
                              f"{len(self.permanently_failed)} permanently failed")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            # Initialize empty state
            self.processed_files = set()
            self.processed_signatures = {}
            self.failed_files = {}
            self.permanently_failed = set()

    def _save_state(self):
        """Save persistent state to disk."""
        try:
            # Clean up old failed files (older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            cleaned_failed = {}
            for file_id, data in self.failed_files.items():
                if isinstance(data, dict) and 'last_attempt' in data:
                    last_attempt = datetime.fromisoformat(data['last_attempt'])
                    if last_attempt > cutoff_time:
                        cleaned_failed[file_id] = data
                elif isinstance(data, int):
                    # Old format - keep for now but will be updated
                    cleaned_failed[file_id] = data
            
            self.failed_files = cleaned_failed
            
            state = {
                'processed_files': list(self.processed_files),
                'processed_signatures': self.processed_signatures,
                'failed_files': self.failed_files,
                'permanently_failed': list(self.permanently_failed),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")

    def start(self, num_workers: int = 3):
        """Start monitoring for new videos."""
        if self.is_running:
            logger.warning("Video monitor is already running")
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(num_workers):
            worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            worker_thread.start()
            self.workers.append(worker_thread)
        
        # Initialize monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started video monitor with {num_workers} workers")

    def stop(self):
        """Stop the monitoring and processing system."""
        logger.info("Stopping video monitor...")
        self.is_running = False
        
        # Wait for monitor thread with timeout
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop gracefully")
        
        # Wait for worker threads with timeout
        for i, worker in enumerate(self.workers):
            if worker.is_alive():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    logger.warning(f"Worker thread {i} did not stop gracefully")
        
        # Save state before shutdown
        self._save_state()
        
        # Shutdown video processor
        if hasattr(self, 'video_processor') and self.video_processor:
            try:
                self.video_processor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down video processor: {str(e)}")
        
        logger.info("Stopped video monitor")

    def _create_file_signature(self, video: dict) -> str:
        """
        Create a unique signature for a file based on file attributes including name.
        Uses file name + size + creation time so renamed files are treated as new.
        
        Args:
            video (dict): Video file metadata
            
        Returns:
            str: File signature hash
        """
        # Include filename so renamed files are treated as new videos
        file_name = video.get('name', '')
        file_size = video.get('size', '0')
        created_time = video.get('createdTime', '')
        
        # Include file ID as secondary identifier
        file_id = video.get('id', '')
        
        # Create signature from name + size + creation time + id
        signature_data = f"{file_name}_{file_size}_{created_time}_{file_id}"
        
        signature = hashlib.md5(signature_data.encode()).hexdigest()
        
        logger.debug(f"📝 Created signature for '{file_name}': {signature[:8]}... (includes filename for rename detection)")
        
        return signature

    def _is_file_changed(self, video: dict) -> bool:
        """
        Check if a file has been changed (content modified) since last processing.
        Now resistant to file renaming in Google Drive.
        
        Args:
            video (dict): Video file metadata
            
        Returns:
            bool: True if file content has changed, False otherwise
        """
        video_id = video['id']
        current_signature = self._create_file_signature(video)
        
        # If file ID not processed before, it's new
        if video_id not in self.processed_files:
            return True
            
        # If we have a signature for this file, compare it
        if video_id in self.processed_signatures:
            stored_signature = self.processed_signatures[video_id]
            if stored_signature != current_signature:
                logger.info(f"Detected content change in file '{video['name']}' (ID: {video_id}) - will reprocess")
                return True
            # File signature matches - no change, skip processing even if renamed
            logger.debug(f"File '{video['name']}' (ID: {video_id}) unchanged, skipping")
            return False
        
        # If no signature stored but file was processed, create signature and skip
        # (this handles upgrade from old version without proper signatures)
        logger.info(f"Creating signature for previously processed file '{video['name']}' (ID: {video_id})")
        self.processed_signatures[video_id] = current_signature
        self._save_state()
        return False

    def _monitor_loop(self):
        """Main monitoring loop that checks for new/changed videos."""
        consecutive_errors = 0
        max_consecutive_errors = 3
        last_retry_check = datetime.now()
        retry_check_interval = timedelta(minutes=1)  # Check for retries every minute
        
        logger.info("Starting video monitoring loop")
        
        while self.is_running:
            try:
                # Check for videos ready for retry every minute
                now = datetime.now()
                if now - last_retry_check > retry_check_interval:
                    self._check_failed_files_for_retry()
                    last_retry_check = now
                
                # Reset consecutive error counter on successful iteration
                consecutive_errors = 0
                
                # Get videos from Drive
                videos = self.drive_service.list_new_videos()
                if videos is None:
                    logger.warning("Failed to retrieve video list")
                    time.sleep(self.poll_interval)
                    continue
                
                with self.lock:
                    for video in videos:
                        video_id = video['id']
                        
                        # Skip if already being processed
                        if video_id in self.processing_files:
                            continue
                        
                        # Skip if permanently failed
                        if video_id in self.permanently_failed:
                            continue
                        
                        # Check if we should process this video
                        should_process = False
                        
                        # Process if never seen before
                        if video_id not in self.processed_files:
                            should_process = True
                        
                        # Process if file changed
                        elif self._is_file_changed(video):
                            should_process = True
                            # Remove from processed list if file changed
                            if video_id in self.processed_files:
                                self.processed_files.remove(video_id)
                            if video_id in self.processed_signatures:
                                del self.processed_signatures[video_id]
                            # Clear from failed files since it's a new version
                            if video_id in self.failed_files:
                                del self.failed_files[video_id]
                        
                        # Skip if it's in failed files but not ready for retry
                        if video_id in self.failed_files:
                            if not self._should_retry_file(video_id):
                                continue
                        
                        # Add to processing queue
                        if should_process:
                            self.processing_queue.put(video)
                            self.processing_files.add(video_id)
                            
                            # Log appropriately
                            if video_id in self.failed_files:
                                retry_info = self.failed_files[video_id]
                                retry_count = retry_info['count'] if isinstance(retry_info, dict) else retry_info
                                logger.info(f"Re-queued failed video for retry {retry_count + 1}/{self.max_retries}: {video['name']} (ID: {video_id})")
                            else:
                                change_reason = "new file" if video_id not in self.processed_signatures else "file changed"
                                logger.info(f"Added video to queue ({change_reason}): {video['name']} (ID: {video_id})")
                
                # Save state periodically
                self._save_state()
                
                # Sleep until next check
                time.sleep(self.poll_interval)
                
            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)
                
                # Handle specific error types
                if "SSL" in error_msg or "ssl" in error_msg.lower():
                    logger.warning(f"SSL error in monitor loop (attempt {consecutive_errors}): {error_msg}")
                    sleep_time = min(60, 10 * consecutive_errors)  # Backoff for SSL errors
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    logger.warning(f"Timeout error in monitor loop (attempt {consecutive_errors}): {error_msg}")
                    sleep_time = min(30, 5 * consecutive_errors)  # Shorter backoff for timeouts
                else:
                    logger.error(f"Error in monitor loop (attempt {consecutive_errors}): {error_msg}")
                    sleep_time = min(60, 10 * consecutive_errors)
                
                # If too many consecutive errors, wait longer
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), waiting longer...")
                    sleep_time = 300  # 5 minutes
                    consecutive_errors = 0  # Reset counter
                
                time.sleep(sleep_time)

    def _check_failed_files_for_retry(self):
        """Check failed files and queue those ready for retry."""
        if not self.failed_files:
            return
            
        retry_count = 0
        with self.lock:
            # Create a copy to avoid modification during iteration
            failed_items = list(self.failed_files.items())
            
            for video_id, failed_info in failed_items:
                if self._should_retry_file(video_id):
                    # Need to get video info from Drive to re-queue
                    try:
                        # Try to get the video metadata from Drive
                        videos = self.drive_service.list_new_videos()
                        if videos:
                            for video in videos:
                                if video['id'] == video_id:
                                    # Skip if already being processed
                                    if video_id not in self.processing_files:
                                        self.processing_queue.put(video)
                                        self.processing_files.add(video_id)
                                        retry_count += 1
                                        
                                        retry_info = self.failed_files[video_id]
                                        current_retry = retry_info['count'] if isinstance(retry_info, dict) else retry_info
                                        logger.info(f"🔄 Re-queued failed video for retry {current_retry + 1}/{self.max_retries}: {video['name']} (ID: {video_id})")
                                    break
                    except Exception as e:
                        logger.warning(f"Error while checking failed file {video_id} for retry: {str(e)}")
        
        if retry_count > 0:
            logger.info(f"Re-queued {retry_count} failed videos for retry")

    def _should_retry_file(self, video_id: str) -> bool:
        """Check if a failed file should be retried based on retry count and time."""
        if video_id not in self.failed_files:
            return True
        
        failed_info = self.failed_files[video_id]
        
        # Handle old format (just integer)
        if isinstance(failed_info, int):
            retry_count = failed_info
            # Convert to new format
            self.failed_files[video_id] = {
                'count': retry_count,
                'last_attempt': datetime.now().isoformat()
            }
            return retry_count < self.max_retries
        
        # New format (dict with count and timestamp)
        retry_count = failed_info['count']
        if retry_count >= self.max_retries:
            return False
        
        # Check if enough time has passed for retry (exponential backoff)
        last_attempt = datetime.fromisoformat(failed_info['last_attempt'])
        wait_time = timedelta(minutes=2 ** retry_count)  # 2, 4, 8 minutes
        
        return datetime.now() > last_attempt + wait_time

    def _process_queue(self):
        """Worker thread that processes videos from the queue."""
        while self.is_running:
            try:
                # Get video from queue
                video = self.processing_queue.get(timeout=1)
                video_id = video['id']
                
                try:
                    # Process the video with forced fresh processing
                    success = self._process_video_guaranteed(video)
                    
                    with self.lock:
                        if success:
                            # Mark as successfully processed
                            self.processed_files.add(video_id)
                            # Store file signature for change detection
                            self.processed_signatures[video_id] = self._create_file_signature(video)
                            # Remove from failed files if it was there
                            if video_id in self.failed_files:
                                del self.failed_files[video_id]
                            logger.info(f"✅ Successfully processed video: {video['name']}")
                        else:
                            # Handle failure with improved tracking
                            current_failed = self.failed_files.get(video_id, {'count': 0})
                            if isinstance(current_failed, int):
                                current_failed = {'count': current_failed}
                            
                            retry_count = current_failed['count'] + 1
                            
                            if retry_count <= self.max_retries:
                                self.failed_files[video_id] = {
                                    'count': retry_count,
                                    'last_attempt': datetime.now().isoformat(),
                                    'name': video['name']
                                }
                                logger.warning(f"❌ Failed to process video {video['name']}, will retry ({retry_count}/{self.max_retries})")
                            else:
                                # Move to permanently failed
                                self.permanently_failed.add(video_id)
                                if video_id in self.failed_files:
                                    del self.failed_files[video_id]
                                logger.error(f"🚫 Permanently failed to process video {video['name']} after {self.max_retries} attempts")
                
                finally:
                    # Remove from processing set
                    with self.lock:
                        self.processing_files.discard(video_id)
                    self.processing_queue.task_done()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing thread: {str(e)}")
                # Add a small delay to prevent rapid error loops
                time.sleep(1)

    def _process_video_guaranteed(self, video: dict) -> bool:
        """
        Process a single video with guaranteed fresh processing (no cache).
        
        Args:
            video (dict): Video metadata from Google Drive
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        video_id = video['id']
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Download video
        input_path = temp_dir / f"input_{video['name']}"
        if not self._download_with_retry(video_id, input_path):
            return False
        
        try:
            # Process video with AI captions - FORCE FRESH PROCESSING
            output_path = temp_dir / f"processed_{video['name']}"
            
            # Force bypassing cache by using unique processing parameters
            success = self.video_processor.process_video(
                str(input_path),
                str(output_path),
                force_fresh=True  # This will bypass cache
            )
            
            if not success:
                logger.error(f"Video processing failed for {video['name']}")
                return False
            
            # Verify output file exists
            if not output_path.exists():
                logger.error(f"Output file not created for {video['name']}")
                return False
            
            # Upload processed video
            upload_title = f"Processed_{video['name']}"
            uploaded_id = self._upload_with_retry(output_path, upload_title)
            if not uploaded_id:
                return False
            
            logger.info(f"✅ Uploaded processed video with ID: {uploaded_id}")
            return True
            
        finally:
            # Cleanup - always clean up temporary files
            self._cleanup_files([input_path, output_path])

    def _download_with_retry(self, file_id: str, output_path: Path) -> bool:
        """Download a file with retry logic."""
        for attempt in range(self.max_retries):
            try:
                if self.drive_service.download_video(file_id, str(output_path)):
                    return True
                logger.warning(f"Download attempt {attempt + 1} failed for file {file_id}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        return False

    def _upload_with_retry(self, file_path: Path, title: str) -> str:
        """Upload a file with retry logic."""
        for attempt in range(self.max_retries):
            try:
                uploaded_id = self.drive_service.upload_video(str(file_path), title)
                if uploaded_id:
                    return uploaded_id
                logger.warning(f"Upload attempt {attempt + 1} failed for file {file_path}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Upload attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        return None

    def _cleanup_files(self, file_paths: list):
        """Clean up temporary files."""
        for path in file_paths:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Cleaned up temporary file: {path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {path}: {str(e)}")

    def get_status(self) -> dict:
        """Get current status of the monitoring system."""
        with self.lock:
            return {
                'queue_size': self.processing_queue.qsize(),
                'processing_count': len(self.processing_files),
                'processed_count': len(self.processed_files),
                'failed_count': len(self.failed_files),
                'permanently_failed_count': len(self.permanently_failed),
                'is_running': self.is_running
            } 