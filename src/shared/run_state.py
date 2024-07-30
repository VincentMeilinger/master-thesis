import configparser
import json
import os

run_state = {}
_initialized = False


def load(run_id: str, run_path: str):
    global _initialized, run_state
    print("RUN_STATE", json.dumps(run_state, indent=4))
    if _initialized:
        return

    config = configparser.ConfigParser()

    try:
        config.read(os.path.join(run_path, 'run_state.ini'))
        run_state = {section: dict(config.items(section)) for section in config.sections()}

        if not run_state:
            raise Exception("run_state.ini is empty.")
    except Exception as e:
        run_state = {
            'general': {
                'run_id': run_id,
                'run_path': run_path
            }
        }

    print("RUN_STATE", json.dumps(run_state, indent=4))
    _initialized = True


def save():
    config = configparser.ConfigParser()
    for section, section_data in run_state.items():
        config[section] = section_data
    path = os.path.join(run_state['general']['run_path'], 'run_state.ini')
    with open(path, 'w') as configfile:
        config.write(configfile)


def set_state(section: str, key: str, value: str):
    if section not in run_state:
        run_state[section] = {}
    run_state[section][key] = value
    save()


def get_state(section: str, key: str):
    if section not in run_state:
        return None
    return run_state[section].get(key, None)


def completed(section: str, key: str):
    return get_state(section, key) == 'completed'


def reset():
    global run_state, _initialized
    try:
        path = run_state['general']['run_path']
        if path is not None:
            os.remove(os.path.join(str(path), 'run_state.ini'))
            run_state.clear()
            _initialized = False
    except Exception as e:
        print(f"Unable to reset run state: {e}")
