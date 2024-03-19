import os

from src.util.logger import create_logger

# logger
current_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(current_dir, '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, 'log.log')
logger = create_logger(log_file)
