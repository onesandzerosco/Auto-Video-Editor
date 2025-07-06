import sys
from loguru import logger
from ..config import LOG_LEVEL, LOG_FILE

def setup_logger():
    """Configure the logger with appropriate settings."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL
    )
    
    # Add file handler
    logger.add(
        LOG_FILE,
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=LOG_LEVEL
    )
    
    return logger

# Initialize logger
logger = setup_logger() 