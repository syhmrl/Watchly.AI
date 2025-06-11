import os
import yaml

from config import _CONFIG_PATH

# Path to your YAML tracker file
_TRACKER_PATH = os.path.join(os.path.dirname(__file__), "custom_tracker.yaml")

def _load_yaml():
    """Load custom_tracker.yaml and return as dict."""
    if not os.path.exists(_TRACKER_PATH):
        raise FileNotFoundError(f"{_TRACKER_PATH} does not exist")
    with open(_TRACKER_PATH, 'r') as f:
        data = yaml.safe_load(f)
    return data

def _write_yaml(data):
    """Overwrite custom_tracker.yaml with `data` dict."""
    with open(_TRACKER_PATH, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def get_tracker_settings():
    """
    Return the full tracker settings dict from custom_tracker.yaml.
    """
    return _load_yaml()

def set_tracker_settings(new_settings: dict):
    """
    Overwrite the YAML with new_settings (must contain all keys).
    """
    _write_yaml(new_settings)

def reset_tracker_to_defaults():
    """
    Reset custom_tracker.yaml to the defaults stored in config.json under 'default_setting'.
    """
    import json
    # Load defaults from config.json
    with open(_CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    defaults = cfg.get("default_setting", {})
    if not defaults:
        raise KeyError("No 'default_setting' found in config.json")

    # Convert string booleans to actual booleans
    for k, v in defaults.items():
        if isinstance(v, str) and v.lower() in ("true","false"):
            defaults[k] = v.lower() == "true"

    # Write them to YAML
    _write_yaml(defaults)
    return defaults

def confirm_tracker_settings():
    """
    Reloads and returns the settings from YAML for confirmation.
    """
    return _load_yaml()


# Example usage:
if __name__ == "__main__":
    print("Current tracker settings:")
    print(get_tracker_settings())

    print("\nResetting to defaults...")
    defaults = reset_tracker_to_defaults()
    print("Defaults applied:", defaults)

    print("\nConfirm YAML now contains:")
    print(confirm_tracker_settings())
