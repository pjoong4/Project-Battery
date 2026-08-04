from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="battery1/data.yaml",
    epochs=60
    patience=10
    lr0=0.01,  
    lrf=0.01,
    close_mosaic=10
    imgsz=640,
    batch=32,
    device="CUDA"
    workers=4,
    cache=True,
)