import os
import json
import torch
import logging
from datetime import datetime
from dotenv import load_dotenv

# Environment variables
env = load_dotenv(dotenv_path='environment.env')
secret_env = load_dotenv(dotenv_path='secrets.env')

if not secret_env:
    print("File 'secrets.env' missing. Exiting...")
    exit(1)

# Database
DB_URI = os.getenv('DB_URI')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Transformer
MAX_SEQ_LEN = int(os.getenv('MAX_SEQ_LEN', 256))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Logging
LOG_LEVEL_STREAM = os.getenv('LOG_LEVEL_STREAM', 'DEBUG')
LOG_LEVEL_FILE = os.getenv('LOG_LEVEL_FILE', 'DEBUG')

# Generate or load Run ID
RUN_ID = os.getenv('RUN_ID')
if RUN_ID is None:
    os.environ['RUN_ID'] = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    RUN_ID = os.getenv('RUN_ID')

# General directories
DATASET_DIR = os.getenv('DATASET_DIR', './data/datasets')
PIPELINE_DIR = os.getenv('PIPELINE_DIR', './data/pipeline')
MODEL_DIR = os.getenv('MODEL_DIR', './data/models')

# Run-specific directories
RUN_DIR = os.path.join(PIPELINE_DIR, RUN_ID)
LOG_DIR = os.path.join(RUN_DIR, os.getenv('LOG_DIR', 'logs'))
PROCESSED_DATA_DIR = os.path.join(RUN_DIR, os.getenv('PROCESSED_DATA_DIR', 'processed_data'))


# Logging
def get_logger(name: str, file_name: str = None):
    """
    Creates and returns a logger with both StreamHandler and FileHandler.

    :param name: Name of the logger.
    :param file_name: Name of the log file.
    :return: Configured logger.
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL_STREAM)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL_STREAM)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(
            LOG_DIR,
            f'{file_name}.log' if file_name is not None else f'{datetime.now().strftime("%y%m%d-%H%M%S")}.log')
    )
    file_handler.setLevel(LOG_LEVEL_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_dirs():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    if not os.path.exists(PIPELINE_DIR):
        os.makedirs(PIPELINE_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Create for each run
    if not os.path.exists(RUN_DIR):
        os.makedirs(RUN_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)


def print_config():
    print(f"\n=== Configuration =================================\n")

    print(f"    > Run ID: {RUN_ID} >>>>>>>>>>>>>>>")
    print(f"    > Loading environment variables: {'Success' if (env & secret_env) else 'Failed'}")

    print(f"\n___ LOGGING _______________________________________\n")
    print(f"    - Stream:                   {LOG_LEVEL_STREAM}")
    print(f"    - File:                     {LOG_LEVEL_FILE}")

    print(f"\n___ DIRS __________________________________________\n")
    print(f"    - DATASET_DIR:              {DATASET_DIR}")
    print(f"    - PIPELINE_DIR:             {PIPELINE_DIR}")
    print(f"    - MODEL_DIR:                {MODEL_DIR}")

    print(f"    - RUN_DIR:                  {RUN_DIR}")
    print(f"    - LOG_DIR:                  {LOG_DIR}")
    print(f"    - PROCESSED_DATA_DIR:       {PROCESSED_DATA_DIR}")

    print(f"\n___ DATABASE ______________________________________\n")
    print(f"    - DB_URI:                   {DB_URI}")
    print(f"    - DB_USER:                  {DB_USER}")

    print(f"\n___ DEVICE ________________________________________\n")
    print(f"    - Device:                   {DEVICE.type}")

    print(f"\n=== End Configuration =============================\n")

