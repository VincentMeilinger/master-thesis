import configparser
import json
import os

run_config = {}
_initialized = False

default_run_config = {
    'general': {},
    'transformer_dim_reduction': {
        'base_model': 'jordyvl/scibert_scivocab_uncased_sentence_transformer',
        'reduced_dim': '32',
        'num_pca_samples': '10000'
    },
    'populate_db': {
        'db_name': 'knowledge_graph',
        'max_nodes': '1000',
        'max_seq_len': '256'
    },
    'embed_datasets': {
        'transformer_model': 'jordyvl/scibert_scivocab_uncased_sentence_transformer',
        'batch_size': '10000'
    },
    'create_edges': {
        'batch_size': '10000',
        'similarity_threshold': '0.98',
        'k_nearest_limit': '10'
    },
    'train_graph_model': {},
    'evaluate_graph_model': {},
}


def load(run_id: str, run_path: str):
    global _initialized, run_config

    if _initialized:
        return

    config = configparser.ConfigParser()

    try:
        config.read(os.path.join(run_path, 'run_config.ini'))
        run_config = {section: dict(config.items(section)) for section in config.sections()}
        print("RUN_CONFIG", json.dumps(run_config, indent=4))
        if not run_config:
            raise Exception("run_config.ini is empty. Initializing with default values and stopping execution. "
                            "Modify the values in the run_config.ini file and run again.")
    except Exception as e:
        run_config = default_run_config.copy()
        run_config['general'] = {
            'run_id': run_id,
            'run_path': run_path
        }
        save()
        raise e

    print("RUN_CONFIG", json.dumps(run_config, indent=4))
    _initialized = True


def save():
    config = configparser.ConfigParser()
    for section, section_data in run_config.items():
        config[section] = section_data
    path = os.path.join(run_config['general']['run_path'], 'run_config.ini')
    with open(path, 'w') as configfile:
        config.write(configfile)


def set_config(section: str, key: str, value: str):
    if section not in run_config:
        run_config[section] = {}
    run_config[section][key] = value
    save()


def get_config(section: str, key: str):
    if section not in run_config:
        return None
    return run_config[section].get(key, None)
