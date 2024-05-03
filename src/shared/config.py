from dotenv import load_dotenv
import logging
import os

# Environment variables
success = load_dotenv(dotenv_path='environment.env')

db_uri = os.getenv('DB_URI')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

data_dir = os.getenv('DATA_DIR', '/data')

log_level_stream = os.getenv('LOG_LEVEL_STREAM', 'DEBUG')
log_level_file = os.getenv('LOG_LEVEL_FILE', 'DEBUG')
log_file_path = os.getenv('LOG_FILE_PATH', '../../logs/program.log')

print(f"\n=== Configuration =================================\n")

print(f"Loading environment variables: {'Success' if success else 'Failed'}")

print(f"\n___ LOGGING _______________________________________\n")
print(f"    - Stream: {log_level_stream}")
print(f"    - File: {log_level_file}")

print(f"\n___ DATA __________________________________________\n")
print(f"    - DATA_DIR: {data_dir}")

print(f"\n___ DATABASE ______________________________________\n")
print(f"    - DB_URI: {db_uri}")
print(f"    - DB_USER: {db_user}")

print(f"\n=== End Configuration =============================\n")


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
