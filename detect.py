import os

from ultralytics import YOLO

print("================================")
print("       ROADMIND AI TEST")
print("================================")

base_dir = os.path.dirname(os.path.abspath(__file__))
model_candidates = [
    r"C:\Users\acer\.cache\huggingface\hub\models--vinothvikas1987--pothole-detection-yolov8\snapshots\b001687443175e43442f63bef4691c3546629def\best.pt",
    os.path.join(base_dir, "data", "best.pt"),
    os.path.join(base_dir, "models", "best.pt"),
    os.path.join(base_dir, "yolo11n.pt"),
]
model_path = next((path for path in model_candidates if os.path.exists(path)), None)

if model_path is None:
    raise FileNotFoundError("No YOLO model found in the configured locations")

model = YOLO(model_path)

# Road image
image_path = os.path.join(base_dir, "data", "road_images", "road.png")

# Run AI detection
results = model(image_path)

# Save result
for result in results:
    result.save(filename=os.path.join(base_dir, "data", "road_images", "yolo_result.png"))

print()
print("AI pothole analysis completed!")
print("Result saved to:")
print("data/road_images/yolo_result.png")
