from ultralytics import YOLO

# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run object detection on the image from a URL or local file
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)

# Optionally, print detected class names and bounding boxes
for result in results:
    print("Detected classes:", result.names)
    print("Bounding boxes:", result.boxes.xyxy)