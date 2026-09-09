import cv2
import os

print("================================")
print("        ROADMIND AI")
print("   Intelligent Road Monitor")
print("================================")

# --------------------------------
# 1. Load road image
# --------------------------------

image_path = "data/road_images/road.png"

if not os.path.exists(image_path):
    print("❌ Road image not found!")
    print("Make sure road.png is inside data/road_images/")
    exit()

image = cv2.imread(image_path)

if image is None:
    print("❌ Unable to read image!")
    exit()

print("✅ Road image loaded successfully!")

height, width, _ = image.shape

print(f"📐 Image size: {width} x {height}")


# --------------------------------
# 2. Convert image to grayscale
# --------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------
# 3. Focus on road region
# --------------------------------

road_start = int(height * 0.35)

road_area = gray[road_start:height, :]


# --------------------------------
# 4. Detect dark damaged regions
# --------------------------------

_, threshold = cv2.threshold(
    road_area,
    70,
    255,
    cv2.THRESH_BINARY_INV
)


# --------------------------------
# 5. Remove small noise
# --------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)

threshold = cv2.morphologyEx(
    threshold,
    cv2.MORPH_CLOSE,
    kernel
)


# --------------------------------
# 6. Find possible hazards
# --------------------------------

contours, _ = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


hazard_detected = False
largest_area = 0
best_box = None


for contour in contours:

    area = cv2.contourArea(contour)

    if area > largest_area:

        largest_area = area

        x, y, w, h = cv2.boundingRect(contour)

        y = y + road_start

        best_box = (x, y, w, h)


# --------------------------------
# 7. Hazard detection
# --------------------------------

if largest_area > 5000:

    hazard_detected = True

    x, y, w, h = best_box

    # Draw bounding box
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        4
    )

    # Label
    cv2.putText(
        image,
        "POTENTIAL ROAD HAZARD",
        (x, max(y - 15, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


# --------------------------------
# 8. Calculate risk
# --------------------------------

if hazard_detected:

    risk_score = min(int(largest_area / 2000), 100)

    if risk_score >= 70:
        risk_level = "HIGH"

    elif risk_score >= 30:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

else:

    risk_score = 0
    risk_level = "LOW"


# --------------------------------
# 9. AI-style analysis
# --------------------------------

if hazard_detected:

    hazard_type = "POTENTIAL POTHOLE"

    # Prototype confidence estimate
    confidence = min(
        70 + int(risk_score * 0.25),
        95
    )

    if risk_level == "HIGH":

        action = "Immediate road maintenance inspection"

    elif risk_level == "MEDIUM":

        action = "Schedule maintenance inspection"

    else:

        action = "Monitor road condition"

else:

    hazard_type = "NO SIGNIFICANT HAZARD"

    confidence = 95

    action = "No immediate maintenance required"


# --------------------------------
# 10. Display AI report
# --------------------------------

print()

print("================================")
print("       🧠 ROADMIND AI REPORT")
print("================================")

print(f"🔎 Hazard Type : {hazard_type}")
print(f"🎯 Confidence : {confidence}%")
print(f"📊 Risk Score : {risk_score}/100")
print(f"🚦 Risk Level : {risk_level}")

print("--------------------------------")

print("🚧 Recommended Action:")
print(action)

print("--------------------------------")

if hazard_detected:

    print("⚠️ Road condition requires attention.")

else:

    print("✅ Road condition appears normal.")

print("================================")


# --------------------------------
# 11. Save result
# --------------------------------

output_path = "data/road_images/road_analysis.png"

cv2.imwrite(
    output_path,
    image
)

print()
print(f"📸 Analysis saved to: {output_path}")


# --------------------------------
# 12. Show result
# --------------------------------

cv2.imshow(
    "ROADMIND - AI Road Monitor",
    image
)

print()
print("Press any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()