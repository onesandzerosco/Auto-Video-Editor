from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaFileUpload
import io
import os
import time
from pathlib import Path
from ratelimit import limits, sleep_and_retry
from ..utils.logger import logger
from ..config import (
    GOOGLE_DRIVE_CREDENTIALS_FILE,
    SOURCE_FOLDER_ID,
    DESTINATION_FOLDER_ID,
    SUPPORTED_FORMATS
)
from typing import Optional
from google.oauth2.service_account import Credentials
from googleapiclient.errors import HttpError

# Rate limiting constants
CALLS_PER_MINUTE = 60
CALLS_PER_DAY = 10000

class DriveService:
    def __init__(self, credentials_path=None):
        """
        Initialize the DriveService with service account credentials.
        
        Args:
            credentials_path (str, optional): Path to service account credentials JSON file.
                                             Defaults to GOOGLE_DRIVE_CREDENTIALS_FILE from config.
        """
        self.credentials_path = credentials_path or GOOGLE_DRIVE_CREDENTIALS_FILE
        self.service = None
        self._authenticate()
        self.processed_files = set()  # Track processed files to avoid duplicates

    def _authenticate(self):
        """Authenticate with Google Drive API using service account credentials."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("Successfully authenticated with Google Drive using service account")
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Drive: {str(e)}")
            raise

    def list_new_videos(self):
        """
        List new videos in the source folder that haven't been processed yet.
        Detects new files, renamed files, and modified files.
        
        Returns:
            list: List of video file metadata dictionaries
        """
        try:
            # Build query for video files
            mime_types = [f"mimeType='video/{fmt}'" for fmt in SUPPORTED_FORMATS]
            mime_query = " or ".join(mime_types)
            query = f"'{SOURCE_FOLDER_ID}' in parents and ({mime_query})"
            
            results = self.service.files().list(
                q=query,
                pageSize=50,
                fields="nextPageToken, files(id, name, mimeType, createdTime, modifiedTime, size)"
            ).execute()
            
            files = results.get('files', [])
            
            # Note: The filtering logic is now handled in VideoMonitor
            # which has access to the persistent state
            return files
        except Exception as e:
            logger.error(f"Error listing videos: {str(e)}")
            return []

    def _get_file_size(self, file_id: str) -> int:
        """Get file size in bytes."""
        try:
            file_metadata = self.service.files().get(fileId=file_id, fields="size").execute()
            return int(file_metadata.get('size', 0))
        except Exception as e:
            logger.warning(f"Could not get file size for {file_id}: {e}")
            return 0

    def download_video(self, file_id: str, filename: str) -> bool:
        """
        Download a video from Google Drive.
        
        Args:
            file_id (str): Google Drive file ID
            filename (str): Local filename to save the video
            
        Returns:
            bool: True if download successful, False otherwise
        """
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                request = self.service.files().get_media(fileId=file_id)
                
                with open(filename, 'wb') as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        if status:
                            progress = int(status.progress() * 100)
                            logger.info(f"Download progress: {progress}%")
                
                logger.info(f"Successfully downloaded video to {filename}")
                return True
                
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                error_msg = str(e)
                
                if attempt < max_retries - 1:
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {delay}s: {error_msg}")
                    time.sleep(delay)
                else:
                    logger.error(f"Error downloading video: {error_msg}")
                    return False
        
        return False

    def upload_video(self, file_path: str, filename: str) -> Optional[str]:
        """
        Upload a video file to Google Drive.
        
        Args:
            file_path (str): Local path to the video file
            filename (str): Name for the file in Google Drive
            
        Returns:
            Optional[str]: File ID if upload successful, None otherwise
        """
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                file_metadata = {
                    'name': filename,
                    'parents': [DESTINATION_FOLDER_ID]
                }
                
                media = MediaFileUpload(file_path, resumable=True)
                
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                file_id = file.get('id')
                logger.info(f"Successfully uploaded video with ID: {file_id}")
                return file_id
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a network/SSL error that we should retry
                if any(error_type in error_str for error_type in [
                    'ssl', 'timeout', 'connection', 'network', 'read operation timed out',
                    'record layer failure', 'remote end closed', 'broken pipe'
                ]):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Network error uploading video (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                
                logger.error(f"Error uploading video: {e}")
                return None
        
        logger.error(f"Failed to upload video after {max_retries} attempts")
        return None

    def mark_as_processed(self, file_id):
        """
        Mark a file as processed to avoid reprocessing.
        
        Args:
            file_id (str): Google Drive file ID
        """
        self.processed_files.add(file_id)

    def cleanup_local_files(self, directory):
        """
        Clean up local files in the specified directory.
        
        Args:
            directory (str): Path to directory containing files to clean up
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return

            for file_path in dir_path.glob("*"):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        logger.info(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_file_metadata(self, file_id):
        """
        Get metadata for a specific file.
        
        Args:
            file_id (str): Google Drive file ID
            
        Returns:
            dict: File metadata or None if not found
        """
        try:
            file = self.service.files().get(
                fileId=file_id,
                fields='id, name, mimeType, size, createdTime'
            ).execute()
            return file
        except Exception as e:
            logger.error(f"Error getting file metadata: {str(e)}")
            return None 