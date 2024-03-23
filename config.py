import logging
import os

# Environment variables
db_user = os.getenv('DB_USER', 'neo4j')
db_password = os.getenv('DB_PASSWORD', 'password')
db_url = os.getenv('DB_URL', 'bolt://localhost:7687')


# Logging
def get_logger(
    name,
    file_path='logs/program.log',
    stream_level=logging.DEBUG,
    file_level=logging.DEBUG
):
    """
    Creates and returns a logger with both StreamHandler and FileHandler.

    :param name: Name of the logger.
    :param file_path: Path to the log file.
    :param stream_level: Logging level of the stream handler.
    :param file_level: Logging level of the file handler.
    :return: Configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(stream_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
