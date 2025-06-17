# config.py
import json
import os

# Path to the JSON file
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def _load():
    """Load the JSON file from disk and return a dict."""
    with open(_CONFIG_PATH, "r") as f:
        data = json.load(f)
    return data

def _save(data: dict):
    """Overwrite the JSON file with updated data."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
    # To ensure any running code that uses config can re‐load, they must re‐call getter functions.

# Expose getter functions for each setting:

def get_camera_credentials():
    data = _load()
    return data["CAM_USERNAME"], data["CAM_PASSWORD"]

def get_camera_ips():
    data = _load()
    return data["CAM_IP"]

def get_model_name():
    data = _load()
    return data["model_default_setting"]["model"]

def get_frame_size():
    data = _load()
    return (data["video_parameters"]["width"], data["video_parameters"]["height"])

def get_camera_sources():
    data = _load()
    sources = data["CAMERA_SOURCES"]
    # If any entry is `null` (None), build from CAM_IP/credentials:
    username, password = get_camera_credentials()
    ips = get_camera_ips()
    new_sources = []
    for idx, val in enumerate(sources):
        if val is None and idx < len(ips):
            new_sources.append(f"rtsp://{username}:{password}@{ips[idx]}:554/Streaming/Channels/101")
        else:
            new_sources.append(val)
    return new_sources

def get_database_path():
    data = _load()
    return data["DATABASE_PATH"]

def get_model_conf():
    data = _load()
    return data["model_default_setting"]["conf"]

def get_model_iou():
    data = _load()
    return data["model_default_setting"]["iou"]

def get_roi_values():
    data = _load()
    return data.get('roi_values', {})

# Write‐back functions:

def set_camera_ips(new_ip_list):
    data = _load()
    data["CAM_IP"] = new_ip_list
    _save(data)

def set_model_name(new_model):
    data = _load()
    data["model_default_setting"]["model"] = new_model
    _save(data)

def set_frame_size(width, height):
    data = _load()
    data["video_parameters"]["width"]  = width
    data["video_parameters"]["height"] = height
    _save(data)
    
def set_model_conf(new_conf):
    data = _load()
    data["model_default_setting"]["conf"] = new_conf
    _save(data)
    
def set_model_iou(new_iou):
    data = _load()
    data["model_default_setting"]["iou"] = new_iou
    _save(data)
    
def set_roi_values(roi_name, roi_data):
    data = _load()
    data["roi_values"][roi_name] = roi_data
    _save(data)
    