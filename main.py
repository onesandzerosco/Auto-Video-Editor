import time
import signal
import sys
from pathlib import Path
from src.services.drive_service import DriveService
from src.services.video_monitor import VideoMonitor
from src.utils.logger import logger
from src.config import PROCESSING_INTERVAL_SECONDS, BASE_DIR

class VideoProcessingApp:
    def __init__(self):
        self.drive_service = DriveService()
        self.monitor = VideoMonitor(
            drive_service=self.drive_service,
            poll_interval=PROCESSING_INTERVAL_SECONDS
        )
        self.temp_dir = BASE_DIR / "temp"
        self.temp_dir.mkdir(exist_ok=True)

    def run(self):
        """Main application loop."""
        logger.info("Starting video processing application")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        try:
            # Start the monitor
            self.monitor.start()
            
            # Keep the main thread alive
            while True:
                # Log status every minute
                status = self.monitor.get_status()
                logger.info(f"System status: {status}")
                time.sleep(60)
                
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            self._handle_shutdown()

    def _handle_shutdown(self, signum=None, frame=None):
        """Handle graceful shutdown of the application."""
        logger.info("Shutting down video processing application...")
        
        try:
            # Stop the monitor first
            if hasattr(self, 'monitor') and self.monitor:
                logger.info("Stopping video monitor...")
                self.monitor.stop()
                
                # Shutdown video processor if available
                if hasattr(self.monitor, 'video_processor') and self.monitor.video_processor:
                    logger.info("Shutting down video processor...")
                    self.monitor.video_processor.shutdown()
            
            # Clean up temporary directory
            try:
                if self.temp_dir.exists():
                    for file in self.temp_dir.glob("*"):
                        if file.is_file():
                            file.unlink()
                    logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        
        logger.info("Application shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    app = VideoProcessingApp()
    app.run() 