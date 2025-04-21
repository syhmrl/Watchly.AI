import requests
from requests.auth import HTTPDigestAuth
import torch

# Replace these values with your camera's settings
camera_ip = "192.168.1.64"
username = "admin"
password = "Abcdefghi1"

# Example URL to get device info (this URL may vary based on the camera model and firmware)
url = f"http://{camera_ip}/ISAPI/System/deviceInfo"

response = requests.get(url, auth=HTTPDigestAuth(username, password))

if response.status_code == 200:
    #print("Device Info:")
    #print(response.text)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
else:
    print("Failed to get device info. Status code:", response.status_code)

from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.export(format="engine", device="cuda")