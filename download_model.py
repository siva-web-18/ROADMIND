from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="vinothvikas1987/pothole-detection-yolov8",
    filename="best.pt"
)

print("Model downloaded successfully!")
print("Model path:", model_path)
