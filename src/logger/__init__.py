import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5

log_dir_path = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def configure_logger():
    """
    Configures the logger to write logs to a file with rotation.
    Prevents duplicate handlers.
    """
    logger = logging.getLogger()
    
    # Check if handlers are already added to prevent duplicates
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
