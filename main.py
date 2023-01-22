import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
results = model("video.mp4", stream=True)  # List of Results objects

for result in results:
    # print(result.numpy())
    for box in result:
        print(box.numpy())
#     cv2.imshow("show", frame)
#     cv2.waitKey(1)  # 1 millisecond 111222
