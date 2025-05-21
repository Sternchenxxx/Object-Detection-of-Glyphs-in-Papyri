from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data=r"D:\FAU\project\papyri.yaml",
    epochs=80,
    batch=16,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.003,
    lrf=0.05,
    weight_decay=0.01,
    cos_lr=True,
    cache=False,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    augment=True,
    project=r"D:\FAU\project\output4",
    name="papyri_train"
)
