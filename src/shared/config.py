from dotenv import load_dotenv
import logging
import json
import os

# Environment variables
env = load_dotenv(dotenv_path='environment.env')
secret_env = load_dotenv(dotenv_path='secrets.env')

if not secret_env:
    print("File 'secrets.env' missing. Exiting...")
    exit(1)

data_dir = os.getenv('DATA_DIR', './data')
model_dir = os.getenv('MODEL_DIR', './data/models')
log_dir = os.getenv('LOG_DIR', './logs')

db_uri = os.getenv('DB_URI')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

model_config = os.getenv('MODEL_CONFIG', 'model_config.json')

log_level_stream = os.getenv('LOG_LEVEL_STREAM', 'DEBUG')
log_level_file = os.getenv('LOG_LEVEL_FILE', 'DEBUG')

print(f"\n=== Configuration =================================\n")

print(f"Loading environment variables: {'Success' if (env & secret_env) else 'Failed'}")

print(f"\n___ LOGGING _______________________________________\n")
print(f"    - Stream: {log_level_stream}")
print(f"    - File: {log_level_file}")

print(f"\n___ DIRS __________________________________________\n")
print(f"    - DATA_DIR: {data_dir}")
print(f"    - MODEL_DIR: {model_dir}")
print(f"    - LOG_DIR: {log_dir}")

print(f"\n___ DATABASE ______________________________________\n")
print(f"    - DB_URI: {db_uri}")
print(f"    - DB_USER: {db_user}")

print(f"\n=== End Configuration =============================\n")


def get_params():
    with open(model_config, 'r') as file:
        params = json.load(file)
        return params


# Logging
def get_logger(name: str):
    """
    Creates and returns a logger with both StreamHandler and FileHandler.

    :param name: Name of the logger.
    :return: Configured logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
    file_handler = logging.FileHandler(os.path.join(log_dir, 'name_disambiguation.log'))
    file_handler.setLevel(log_level_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_dirs():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
