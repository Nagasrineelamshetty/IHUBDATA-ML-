from ultralytics import YOLO
import cv2
import os

# Load segmentation model (e.g., YOLOv8n-seg)
model = YOLO("yolov8n-seg.pt")

# Folder with input images
input_folder = "images"
output_folder = "segmented_images"
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, img_name)
        results = model(img_path, save=True, project=output_folder, name='runs')
