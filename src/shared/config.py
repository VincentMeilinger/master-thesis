import torch
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

dataset_dir = os.getenv('DATASET_DIR', './data/datasets')
dataset_processed_dir = os.getenv('DATASET_PROCESSED_DIR', './data/datasets_processed')
model_dir = os.getenv('MODEL_DIR', './data/models')
log_dir = os.getenv('LOG_DIR', './data/logs')
pipeline_state_file = os.getenv('PIPELINE_STATE_FILE', './data/pipeline_state.json')

db_uri = os.getenv('DB_URI')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

model_config = os.getenv('PIPELINE_CONFIG', 'pipeline_config.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

log_level_stream = os.getenv('LOG_LEVEL_STREAM', 'DEBUG')
log_level_file = os.getenv('LOG_LEVEL_FILE', 'DEBUG')


def get_params():
    with open(model_config, 'r') as file:
        params = json.load(file)
        return params


def check_params_changed():
    current_params = get_params()
    saved_params = get_pipeline_state()['params']
    for key, value in current_params.items():
        if value != saved_params[key]:
            return True
    return False


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


def save_pipeline_state(state: dict):
    with open(pipeline_state_file, 'w') as file:
        json.dump(state, file)


def get_pipeline_state():
    with open(pipeline_state_file, 'r') as file:
        state = json.load(file)

    return state


def init_pipeline_state():
    # Initialize pipeline state
    state = {
        'embed_datasets': {'state': 'not_started'},
        'populate_db': {'state': 'not_started'},
        'train_model': {'state': 'not_started'},
        'evaluate_model': {'state': 'not_started'},
        'params': get_params()
    }

    with open(pipeline_state_file, 'w') as file:
        json.dump(state, file)


def create_dirs():
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(dataset_processed_dir):
        os.makedirs(dataset_processed_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(pipeline_state_file):
        init_pipeline_state()


def clear_pipeline_state():
    """ Clear the pipeline state and all pipeline data. """
    for file in os.listdir(dataset_processed_dir):
        os.remove(os.path.join(dataset_processed_dir, file))
    init_pipeline_state()


def print_config():
    pipeline_state = get_pipeline_state()
    print(f"\n=== Configuration =================================\n")

    print(f"Loading environment variables: {'Success' if (env & secret_env) else 'Failed'}")

    print(f"\n___ LOGGING _______________________________________\n")
    print(f"    - Stream:                   {log_level_stream}")
    print(f"    - File:                     {log_level_file}")

    print(f"\n___ DIRS __________________________________________\n")
    print(f"    - DATASET_DIR:              {dataset_dir}")
    print(f"    - DATASET_PROCESSED_DIR:    {dataset_processed_dir}")
    print(f"    - MODEL_DIR:                {model_dir}")
    print(f"    - LOG_DIR:                  {log_dir}")

    print(f"\n___ DATABASE ______________________________________\n")
    print(f"    - DB_URI:                   {db_uri}")
    print(f"    - DB_USER:                  {db_user}")

    print(f"\n___ DEVICE ________________________________________\n")
    print(f"    - Device:                   {device.type}")

    print(f"\n___ PIPELINE STATE ________________________________\n")
    for key, value in pipeline_state.items():
        if key != 'params':
            print(f"    - {key}:            {value['state']}")

    print(f"\n=== End Configuration =============================\n")


# Initialize (create dirs, print config)
create_dirs()
print_config()
