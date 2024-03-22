from logging.handlers import RotatingFileHandler
import logging
import os

LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "DEBUG").upper(), None)  # Log level
LOG_FILE = os.getenv('LOG_FILE', './data/logs/scraper.log')  # Log file


def get_logger(name: str):
    """
    Creates and returns a configured logger.
    """
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(LOG_LEVEL))

    # Create file handler with log level 'LOG_LEVEL'
    if not os.path.exists(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE))
    fh = RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024 * 5, backupCount=5)
    fh.setLevel(logging.getLevelName(LOG_LEVEL))

    # Create stream handler with a higher log level
    sh = logging.StreamHandler()
    sh.setLevel(logging.ERROR)

    # Create formatter, add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
