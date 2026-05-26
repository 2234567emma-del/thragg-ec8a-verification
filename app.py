import os
import cv2
import numpy as np

from flask import Flask
from flask import render_template
from flask import request

from tensorflow.keras.models import load_model
from ocr_utils import extract_numbers_from_boxes
from logic_checker import check_ec8a_logic

# =========================
# CREATE FLASK APP
# =========================

app = Flask(__name__)

# =========================
# LOAD TRAINED MODEL
# =========================

model = load_model("models/ec8a_mobilenet_model.h5")

IMG_SIZE = 224

# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICTION ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    # =========================
    # SAVE IMAGE
    # =========================

    upload_path = os.path.join("static", file.filename)
    file.save(upload_path)

    # =========================
    # OCR EXTRACTION
    # =========================

    ocr_result = extract_numbers_from_boxes(upload_path)
    if isinstance(ocr_result, dict):
        numeric_values = ocr_result.get("values", [])
        none_count = ocr_result.get(
            "none_count",
            sum(value is None for value in numeric_values)
        )
        ocr_status = ocr_result.get("ocr_status", "failed")
        ocr_failed = ocr_result.get("ocr_failed", none_count > 2)
        logic_available = ocr_result.get("logic_available", none_count == 0)
        localization_confidence = ocr_result.get("localization_confidence", 0.0)
        row_split_confidence = ocr_result.get("row_split_confidence", 0.0)
    else:
        numeric_values = ocr_result
        none_count = sum(value is None for value in numeric_values)
        if none_count == 0 and len(numeric_values) == 8:
            ocr_status = "complete"
            ocr_failed = False
            logic_available = True
        elif none_count in (1, 2) and len(numeric_values) == 8:
            ocr_status = "partial"
            ocr_failed = False
            logic_available = False
        else:
            ocr_status = "failed"
            ocr_failed = True
            logic_available = False
        localization_confidence = 0.0
        row_split_confidence = 0.0

    print("OCR NUMERIC VALUES:", numeric_values)
    print("ROW OCR VALUES:", numeric_values)
    print("NONE COUNT:", none_count)
    print("OCR STATUS:", ocr_status)
    print("OCR FAILED:", ocr_failed)
    print("LOCALIZATION CONFIDENCE:", round(localization_confidence, 3))
    print("ROW SPLIT CONFIDENCE:", round(row_split_confidence, 3))

    if (
        logic_available
        and len(numeric_values) == 8
        and all(value is not None for value in numeric_values)
    ):
        logic_result = check_ec8a_logic(numeric_values)
        logic_failed = logic_result["status"] == "TAMPERED"
    else:
        logic_result = {
            "status": "REVIEW",
            "issues": ["Logic check skipped because OCR is incomplete."]
        }
        logic_failed = False

    image = cv2.imread(upload_path)

    if image is None:
        return "Could not read uploaded image"

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(image)[0]

    authentic_score = float(prediction[0]) * 100
    tampered_score = float(prediction[1]) * 100
    tampered_confidence = tampered_score
    heavy_cancellation = none_count >= 4
    cancellation_score = none_count / 8.0

    # =========================
    # DECISION LOGIC
    # =========================

    if tampered_score >= 80:
        decision = "TAMPERED"
        decision_reason = "high_tampered_confidence"

    elif logic_available and logic_failed:
        decision = "TAMPERED"
        decision_reason = "logic_failed"

    elif (
        authentic_score >= 75
        and tampered_score <= 25
        and logic_available
        and not logic_failed
        and none_count == 0
    ):
        decision = "AUTHENTIC"
        decision_reason = "high_authentic_confidence_and_logic_passed"

    elif ocr_failed:
        decision = "NEEDS REVIEW"
        decision_reason = "ocr_failed"

    elif none_count in (1, 2):
        decision = "NEEDS REVIEW"
        decision_reason = "partial_ocr"

    else:
        decision = "NEEDS REVIEW"
        decision_reason = "insufficient_confidence"

    print("TAMPERED CONFIDENCE:", round(tampered_confidence, 2))
    print("HEAVY CANCELLATION DEBUG ONLY:", heavy_cancellation)
    print("CANCELLATION SCORE DEBUG ONLY:", round(cancellation_score, 3))
    print("LOGIC AVAILABLE:", logic_available)
    print("LOGIC FAILED:", logic_failed)
    print("FINAL STATUS:", decision)
    print("FINAL DECISION REASON:", decision_reason)

    # =========================
    # RETURN RESULT
    # =========================

    return render_template(
        "result.html",
        decision=decision,
        authentic_score=round(authentic_score, 2),
        tampered_score=round(tampered_score, 2),
        image_path=upload_path,
        logic_result=logic_result,
        ocr_values=numeric_values
    )


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
