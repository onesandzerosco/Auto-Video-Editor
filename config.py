import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Google Drive settings
GOOGLE_DRIVE_CREDENTIALS_FILE = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE", "credentials.json")
GOOGLE_DRIVE_TOKEN_FILE = os.getenv("GOOGLE_DRIVE_TOKEN_FILE", "token.json")
SOURCE_FOLDER_ID = os.getenv("SOURCE_FOLDER_ID")
DESTINATION_FOLDER_ID = os.getenv("DESTINATION_FOLDER_ID")

# Google Sheets settings
CAPTION_SHEET_ID = os.getenv("CAPTION_SHEET_ID")

# Processing settings
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "1000"))
SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "mp4,mov,avi").split(",")
PROCESSING_INTERVAL_SECONDS = int(os.getenv("PROCESSING_INTERVAL_SECONDS", "60"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "video_processor.log"))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
 