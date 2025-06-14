from ultralytics import YOLO
import os

model = YOLO("yolov8n-seg.pt")

input_folder = "frames"
output_folder = os.path.join(os.getcwd(), "segmented_frames")  # 👈 Full path

os.makedirs(output_folder, exist_ok=True)

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith((".jpg", ".png")):
        input_path = os.path.join(input_folder, filename)
        model(input_path, save=True, save_dir=output_folder)
        print(f"Segmented {filename}")
