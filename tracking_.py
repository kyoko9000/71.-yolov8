import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or a segmentation model .i.e yolov8n-seg.pt
results = model.track(source="video.mp4",
                      stream=True,
                      show=True,
                      )  # tracker="bytetrack.yaml"


def show_frame():
    cv2.imshow("show me", frame)
    cv2.waitKey(1)


for result, frame in results:
    show_frame()
    boxes = result[0].boxes.numpy()  # Boxes object for bbox outputs
    for box in boxes:  # there could be more than one detection
        print("id", box.id)
        print("class", box.cls)
        print("xyxy", box.xyxy)
        print("confidence", box.conf)
        print("\n")
