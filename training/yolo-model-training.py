from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="lattPIhNAlOdOqrvYSrY")
project = rf.workspace("yolo-train-mcoio").project("head-detection-v2")
version = project.version(3)
dataset = version.download("yolov11")

# Load the partially trained model from the last checkpoint
model = YOLO("yolo11s.pt")

# Continue training to 150 epochs, reusing the existing project/run1 settings
results = model.train(
    data=f"{dataset.location}/data.yaml",    # Path to your dataset config file
    epochs=150,                  # Total number of epochs (including already completed)
    project="yolo11_head_detection",  # Project (folder) to save results
    name="run1",                     # Run name (subfolder) to save outputs
    batch=8,                     # Batch size (images per training step)
    workers=2,                   # Number of data loader workers (CPU threads)
    cache=False,                 # Whether to cache images for speed (False to disable)
    plots=True                   # Whether to save training/validation plots
)