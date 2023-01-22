import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
results = model("video.mp4", stream=True)  # List of Results objects

for result in results:
    print(result.cls.numpy())
    # boxes = result.boxes  # Boxes object for bbox outputs
    # print(boxes.numpy())

