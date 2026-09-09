from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import glob

app = Flask(__name__)
CORS(app)

print("========================================")
print("       ROADMIND AI SERVER")
print("========================================")

# ---------------------------------------------------------
# FIND YOLO MODEL AUTOMATICALLY
# ---------------------------------------------------------

model_files = glob.glob(
    r"C:\Users\acer\.cache\huggingface\hub\**\best.pt",
    recursive=True
)

if not model_files:
    print("ERROR: best.pt model not found!")
    model = None
else:
    model_path = model_files[0]

    print("YOLO model found:")
    print(model_path)

    model = YOLO(model_path)

    print("YOLO model loaded successfully!")

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/")
def home():
    return "ROADMIND AI Server is Running!"

# ---------------------------------------------------------
# STATUS
# ---------------------------------------------------------

@app.route("/api/status")
def status():

    return jsonify({
        "status": "online",
        "system": "ROADMIND",
        "ai_model": "YOLO Pothole Detection",
        "model_loaded": model is not None
    })

# ---------------------------------------------------------
# DETECT POTHOLE
# ---------------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():

    if model is None:

        return jsonify({
            "error": "YOLO model not loaded"
        }), 500

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    image = request.files["image"]

    # -----------------------------------------------------
    # GET GPS
    # -----------------------------------------------------

    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    try:
        latitude = float(latitude) if latitude else None
        longitude = float(longitude) if longitude else None
    except ValueError:
        latitude = None
        longitude = None

    # -----------------------------------------------------
    # SAVE UPLOADED IMAGE
    # -----------------------------------------------------

    os.makedirs(
        "data/road_images",
        exist_ok=True
    )

    image_path = os.path.join(
        "data/road_images",
        "uploaded_road.png"
    )

    image.save(image_path)

    print()
    print("========================================")
    print("NEW ROAD ANALYSIS")
    print("========================================")

    print("Image:", image_path)
    print("Latitude:", latitude)
    print("Longitude:", longitude)

    # -----------------------------------------------------
    # RUN YOLO
    # -----------------------------------------------------

    results = model(image_path)

    detected = False
    max_confidence = 0.0
    detections = []

    for result in results:

        # Save AI annotated image
        result.save(
            filename="data/road_images/yolo_result.png"
        )

        if result.boxes is not None:

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": confidence
                })

                if confidence > max_confidence:
                    max_confidence = confidence

                if confidence >= 0.50:
                    detected = True

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    print()
    print("AI ANALYSIS RESULT")

    if detected:

        print("POTHOLE DETECTED")
        print(
            "Confidence:",
            round(max_confidence * 100, 2),
            "%"
        )

    else:

        print("NO POTHOLE DETECTED")

    print("========================================")

    return jsonify({

        "detected": detected,

        "confidence": max_confidence,

        "latitude": latitude,

        "longitude": longitude,

        "detections": detections,

        "result_image":
            "/data/road_images/yolo_result.png"

    })


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
