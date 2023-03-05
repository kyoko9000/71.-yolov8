from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or a segmentation model .i.e yolov8n-seg.pt
results = model.track(source="video.mp4",
                      # stream=True,
                      show=True,
                      tracker="bytetrack.yaml")

for result in results:
    print(result)
    # boxes = result[0].boxes.numpy()  # Boxes object for bbox outputs
    # for box in boxes:  # there could be more than one detection
    #     print("class", box.cls)
    #     print("xyxy", box.xyxy)
    #     print("conf", box.conf)