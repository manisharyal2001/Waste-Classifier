from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import json
import os

# ---------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------
MODEL_PATH = 'waste_classifier_model.h5'
CLASSES_PATH = 'class_indices.json'
TARGET_SIZE = (224, 224)

app = Flask(__name__, static_folder="static")
CORS(app)

# ---------------------------------------------------
# ✅ Load model & labels
# ---------------------------------------------------
model = None
CLASS_LABELS = []

try:
    model = load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)

    CLASS_LABELS = sorted(class_indices, key=class_indices.get)
    print("✅ Model loaded successfully:", CLASS_LABELS)

except Exception as e:
    print("❌ Error loading model:", e)


# ---------------------------------------------------
# ✂️ Crop object from the image
# ---------------------------------------------------
def crop_object(pil_image):
    try:
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            padding = 10

            cropped = img_cv[max(0, y - padding):y + h + padding,
                             max(0, x - padding):x + w + padding]

            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        return pil_image

    except Exception as e:
        print("⚠️ Cropping failed:", e)
        return pil_image


# ---------------------------------------------------
# 🔄 Preprocess image
# ---------------------------------------------------
def preprocess_image(img_file):
    try:
        img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
        img = crop_object(img)
        img = img.resize(TARGET_SIZE)

        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    except Exception as e:
        print("⚠️ Preprocessing failed:", e)
        return None


# ---------------------------------------------------
# 🌐 ROUTES
# ---------------------------------------------------

# ✅ When user opens link → load index.html
@app.route('/')
def serve_home():
    return send_from_directory(app.static_folder, "index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    img = preprocess_image(request.files["file"])

    if img is None:
        return jsonify({"error": "Error processing image."}), 400

    prediction = model.predict(img)
    probabilities = prediction[0]
    index = int(np.argmax(probabilities))

    predicted_label = CLASS_LABELS[index].upper()
    confidence = float(probabilities[index]) * 100

    return jsonify({
        "predicted_class": predicted_label,
        "confidence": f"{confidence:.2f}%",
        "all_confidences": {
            label.upper(): f"{float(prob) * 100:.2f}%"
            for label, prob in zip(CLASS_LABELS, probabilities)
        }
    })


# ---------------------------------------------------
# 🚀 Run Flask Server
# ---------------------------------------------------
if __name__ == '__main__':
    print("\n✅ Smart Waste Classifier Backend Running")
    print("➡ Open on browser: http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)
