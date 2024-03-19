import logging

from logging.handlers import RotatingFileHandler


def create_logger(log_file: str, level: int=logging.INFO) -> logging.Logger:
    """
    Create a logger.

    :param log_file: file where logs are saved
    :param level: logging level
    :return: logger
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=3)  # 1 MB limit, keep 3 old log files
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
