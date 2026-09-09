
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os
import glob
import uuid

# ============================================================
# ROADMIND - AI SMART ROAD MONITORING SYSTEM
# ============================================================

app = Flask(__name__)
CORS(app)

print("=" * 40)
print("          ROADMIND AI SERVER")
print("=" * 40)


# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "road_images"
)

DETECTION_CONFIDENCE = 0.05

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# FIND YOLO MODEL
# ============================================================

MODEL_PATH = None

# Prefer a model bundled with the repository, then check the local cache.
model_candidates = [
    os.path.join(BASE_DIR, "data", "best.pt"),
    os.path.join(BASE_DIR, "models", "best.pt"),
    os.path.join(BASE_DIR, "yolo11n.pt"),
    (
        r"C:\Users\acer\.cache\huggingface\hub"
        r"\models--vinothvikas1987--pothole-detection-yolov8"
        r"\snapshots\b001687443175e43442f63bef4691c3546629def"
        r"\best.pt"
    ),
]

MODEL_PATH = next(
    (path for path in model_candidates if os.path.exists(path)),
    None,
)


# Search automatically if known path is not available
if MODEL_PATH is None:

    search_locations = [

        r"C:\Users\acer\.cache\huggingface\hub",

        os.path.join(BASE_DIR, "models"),

        os.path.join(BASE_DIR, "data")

    ]

    for location in search_locations:

        matches = glob.glob(
            os.path.join(
                location,
                "**",
                "best.pt"
            ),
            recursive=True
        )

        if matches:

            MODEL_PATH = matches[0]

            break


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = None

if MODEL_PATH:

    print("✅ Model found:")
    print(MODEL_PATH)

    try:

        model = YOLO(MODEL_PATH)

        print("✅ YOLO model loaded")

        print(
            "Classes:",
            model.names
        )

    except Exception as e:

        print("❌ YOLO model loading failed")

        print("Error:", e)

else:

    print("❌ YOLO model not found")

    print("Please check best.pt")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    map_file = os.path.join(WEB_DIR, "map.html")

    if not os.path.exists(map_file):

        return jsonify({

            "success": False,

            "error": "map.html not found",

            "expected_location": map_file

        }), 404

    return send_from_directory(WEB_DIR, "map.html")


# ============================================================
# API STATUS
# ============================================================

@app.route("/api/status", methods=["GET"])
def api_status():

    return jsonify({

        "system": "ROADMIND",

        "status": "online",

        "model_loaded": model is not None,

        "model_classes":
            model.names if model else {}

    })


# ============================================================
# AI ROAD DAMAGE DETECTION
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():

    # --------------------------------------------------------
    # CHECK YOLO MODEL
    # --------------------------------------------------------

    if model is None:

        return jsonify({

            "success": False,

            "error": "YOLO model is not loaded"

        }), 500


    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error": "No image uploaded"

        }), 400


    image = request.files["image"]


    if image.filename == "":

        return jsonify({

            "success": False,

            "error": "No image selected"

        }), 400


    # --------------------------------------------------------
    # GET GPS LOCATION
    # --------------------------------------------------------

    try:

        latitude = float(
            request.form.get(
                "latitude",
                0
            )
        )

        longitude = float(
            request.form.get(
                "longitude",
                0
            )
        )

    except (ValueError, TypeError):

        latitude = 0

        longitude = 0


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex[:8]
    image_extension = os.path.splitext(image.filename)[1].lower()

    if image_extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        image_extension = ".jpg"

    image_filename = f"road_{unique_id}{image_extension}"

    image_path = os.path.join(
        UPLOAD_FOLDER,
        image_filename
    )

    try:

        image.save(image_path)

    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                f"Could not save image: {str(e)}"

        }), 500


    print()
    print("=" * 40)
    print("📸 NEW ROAD IMAGE")
    print("File:", image_path)
    print(
        "GPS:",
        latitude,
        longitude
    )
    print("=" * 40)


    # --------------------------------------------------------
    # RUN YOLO DETECTION
    # --------------------------------------------------------

    try:

        input_image = Image.open(image_path).convert("RGB")

        results = model.predict(

            source=input_image,

            conf=DETECTION_CONFIDENCE,

            iou=0.45,

            verbose=False

        )

    except Exception as e:

        print(
            "❌ YOLO detection error:",
            e
        )

        return jsonify({

            "success": False,

            "error":
                f"YOLO detection failed: {str(e)}"

        }), 500


    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    detections = []

    highest_confidence = 0.0

    detected_class = "None"

    best_damage_confidence = 0.0


    for result in results:

        if result.boxes is None:

            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            # Get class name safely
            if isinstance(
                model.names,
                dict
            ):

                class_name = model.names.get(
                    class_id,
                    "Unknown"
                )

            else:

                class_name = model.names[
                    class_id
                ]


            detections.append({

                "class_id":
                    class_id,

                "class":
                    class_name,

                "confidence":
                    confidence

            })


            if confidence > highest_confidence:

                highest_confidence = confidence

            if class_name != "Other" and confidence > best_damage_confidence:

                detected_class = class_name
                best_damage_confidence = confidence

            elif detected_class == "None":

                detected_class = class_name


    # --------------------------------------------------------
    # DAMAGE STATUS
    # --------------------------------------------------------

    detected = (
        len(detections) > 0
    )


    # --------------------------------------------------------
    # SAVE YOLO RESULT IMAGE
    # --------------------------------------------------------

    result_filename = (
        f"yolo_result_{unique_id}.jpg"
    )

    result_path = os.path.join(
        UPLOAD_FOLDER,
        result_filename
    )


    try:

        if results:

            results[0].save(
                filename=result_path
            )

    except Exception as e:

        print(
            "⚠️ Result image could not be saved:",
            e
        )


    # --------------------------------------------------------
    # PRINT AI RESULT
    # --------------------------------------------------------

    print()

    print("🤖 AI ANALYSIS COMPLETE")

    if detected:

        print(
            "🚨 ROAD DAMAGE DETECTED:",
            detected_class
        )

        print(
            "Confidence:",
            round(
                highest_confidence * 100,
                2
            ),
            "%"
        )

    else:

        print(
            "🟢 NO ROAD DAMAGE DETECTED"
        )

    print()


    # --------------------------------------------------------
    # RETURN RESULT TO MAP.HTML
    # --------------------------------------------------------

    return jsonify({

        "success": True,

        "detected": detected,

        "confidence":
            highest_confidence,

        "class":
            detected_class,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "detections":
            detections,

        "result_image":
            f"/data/road_images/{result_filename}"

    })


# ============================================================
# SERVE ROAD ANALYSIS IMAGES
# ============================================================

@app.route(
    "/data/road_images/<path:filename>",
    methods=["GET"]
)
def road_image(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "healthy",

        "system": "ROADMIND",

        "ai": (
            "online"
            if model is not None
            else "offline"
        )

    })


# ============================================================
# 404 HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error":
            "ROADMIND endpoint not found"

    }), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 40)
    print("🌐 ROADMIND WEB APPLICATION")
    print("=" * 40)

    print("Open:")
    print("http://127.0.0.1:5000")

    print()

    print("API status:")
    print("http://127.0.0.1:5000/api/status")

    print()

    print("Health:")
    print("http://127.0.0.1:5000/health")

    print("=" * 40)


    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )

