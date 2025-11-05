# app.py - Flask Backend Server (with Object Cropping)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import json
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2

# ---------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------
MODEL_PATH = 'waste_classifier_model.h5'
CLASSES_PATH = 'class_indices.json'
TARGET_SIZE = (224, 224)

# ---------------------------------------------------
# ✅ Load Model and Classes
# ---------------------------------------------------
model = None
CLASS_LABELS = []

try:
    model = load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)

    CLASS_LABELS = sorted(class_indices, key=class_indices.get)

    print("✅ Model loaded successfully, classes:", CLASS_LABELS)

except Exception as e:
    print(f"❌ Error loading model or class labels: {e}")


# ---------------------------------------------------
# 🌐 Flask Setup (IMPORTANT: static_folder="static")
# ---------------------------------------------------
app = Flask(__name__, static_folder="static")
CORS(app)


# ---------------------------------------------------
# ✂️ Function: Crop main object from image
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

            x = max(0, x - padding)
            y = max(0, y - padding)
            x_end = min(img_cv.shape[1], x + w + 2 * padding)
            y_end = min(img_cv.shape[0], y + h + 2 * padding)

            cropped = img_cv[y:y_end, x:x_end]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped)

        return pil_image

    except Exception as e:
        print(f"⚠️ Cropping failed: {e}")
        return pil_image


# ---------------------------------------------------
# 🔄 Preprocess image for prediction
# ---------------------------------------------------
def preprocess_image(img_file):
    try:
        img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
        img = crop_object(img)
        img = img.resize(TARGET_SIZE)

        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    except Exception as e:
        print(f"⚠️ Preprocessing failed: {e}")
        return None


# ---------------------------------------------------
# 🌍 ROUTES
# ---------------------------------------------------

# ✅ ROOT → Serve index.html directly
@app.route('/')
def home():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    img_data = preprocess_image(file)

    if img_data is None:
        return jsonify({'error': 'Failed to process image.'}), 400

    prediction = model.predict(img_data)
    probabilities = prediction[0]

    index = int(np.argmax(probabilities))
    predicted_label = CLASS_LABELS[index]
    confidence = float(probabilities[index]) * 100

    return jsonify({
        'predicted_class': predicted_label.upper(),
        'confidence': f"{confidence:.2f}%",
        'all_confidences': {
            label.upper(): f"{float(prob) * 100:.2f}%"
            for label, prob in zip(CLASS_LABELS, probabilities)
        }
    })


# ---------------------------------------------------
# 🚀 Run Flask Server (shows link in terminal)
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
