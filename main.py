import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
results = model("video.mp4", show=True)  # List of Results objects
print(len(results))
# for _ in results:
#     print(results.boxes)
#     cv2.imshow("show", frame)
#     cv2.waitKey(1)  # 1 millisecond
# 1111