import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model("video.mp4", return_outputs=True, show=True)  # predict on an image

for result, frame in results:
    print(frame)
    if result:
        for item in result["det"]:
            print("location", item[0:4])
            print("accurate", item[4])
            print("class", item[5])

    cv2.imshow("show", frame)
    cv2.waitKey(1)  # 1 millisecond
