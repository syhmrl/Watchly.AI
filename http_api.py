import requests
from requests.auth import HTTPDigestAuth
import torch

# Replace these values with your camera's settings
# camera_ip = "192.168.1.64"
# username = "admin"
# password = "Abcdefghi1"

# # Example URL to get device info (this URL may vary based on the camera model and firmware)
# url = f"http://{camera_ip}/ISAPI/System/deviceInfo"

# response = requests.get(url, auth=HTTPDigestAuth(username, password))

# if response.status_code == 200:
#     #print("Device Info:")
#     #print(response.text)
#     print(torch.version.cuda)
#     print(torch.cuda.is_available())
# else:
#     print("Failed to get device info. Status code:", response.status_code)

# print(torch.version.cuda)
# print(torch.cuda.is_available())

import os

def check_file_in_same_directory(filename):
    """
    Checks if a file exists in the same directory as the executing script.
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, filename)
    return os.path.isfile(file_path)

# Example usage:
if check_file_in_same_directory("VideoAnalysisFrameReid.py"):
    print("another_file.py is in the same directory.")
else:
    print("another_file.py is NOT in the same directory.")