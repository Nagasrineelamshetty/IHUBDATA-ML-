from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/african-wildlife.yaml", epochs=20, imgsz=640,project="animalclassification")
metrics=model.val()
print(metrics)