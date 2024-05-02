from dotenv import load_dotenv
import logging
import os

# Environment variables
load_dotenv(dotenv_path='../../environment.env')

db_uri = os.getenv('DB_URI')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

log_level_stream = os.getenv('LOG_LEVEL_STREAM', 'DEBUG')
log_level_file = os.getenv('LOG_LEVEL_FILE', 'DEBUG')
log_file_path = os.getenv('LOG_FILE_PATH', '../../logs/program.log')


# Logging
def get_logger(
    name: str,
):
    """
    Creates and returns a logger with both StreamHandler and FileHandler.

    :param name: Name of the logger.
    :return: Configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level_stream)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not os.path.exists('../../logs'):
        os.makedirs('../../logs')
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level_stream)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
