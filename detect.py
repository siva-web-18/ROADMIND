from ultralytics import YOLO

print("================================")
print("       ROADMIND AI TEST")
print("================================")

# Load the pothole detection model
model = YOLO(
    r"C:\Users\acer\.cache\huggingface\hub\models--vinothvikas1987--pothole-detection-yolov8\snapshots\b001687443175e43442f63bef4691c3546629def\best.pt"
)

# Road image
image_path = "data/road_images/road.png"

# Run AI detection
results = model(image_path)

# Save result
for result in results:
    result.save(filename="data/road_images/yolo_result.png")

print()
print("AI pothole analysis completed!")
print("Result saved to:")
print("data/road_images/yolo_result.png")
