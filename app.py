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

# =============================
# 1️⃣ Configuration
# =============================
MODEL_PATH = r'C:\Users\manis\Desktop\Study\Waste\waste_classifier_model.h5'
CLASSES_PATH = r'C:\Users\manis\Desktop\Study\Waste\class_indices.json'
TARGET_SIZE = (224, 224)

# =============================
# 2️⃣ Load Model and Classes
# =============================
model = None
CLASS_LABELS = []
try:
    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    CLASS_LABELS = sorted(class_indices, key=class_indices.get)
    print("✅ Model and Class Labels loaded successfully.")
    print("Class Labels:", CLASS_LABELS)
except Exception as e:
    print(f"❌ Error during setup: {e}")

# =============================
# 3️⃣ Flask App Setup
# =============================
app = Flask(__name__)
CORS(app)  # Allow frontend connection

# =============================
# 4️⃣ Helper Functions
# =============================

def crop_object(pil_image):
    """Crop the main object in the image using OpenCV contours."""
    try:
        # Convert PIL → NumPy (RGB → BGR for OpenCV)
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale and blur
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold for binary mask
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Largest contour = main object
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped = img_cv[y:y + h, x:x + w]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped)
        else:
            return pil_image  # fallback
    except Exception as e:
        print(f"⚠️ Error during cropping: {e}")
        return pil_image


def preprocess_image(img_file):
    """Read and preprocess the image for model prediction."""
    try:
        img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
        img = crop_object(img)  # focus only on main object
        img = img.resize(TARGET_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"⚠️ Error during image preprocessing: {e}")
        return None


# =============================
# 5️⃣ Prediction Route
# =============================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['file']
    img_data = preprocess_image(file)

    if img_data is None:
        return jsonify({'error': 'Invalid image format or processing error.'}), 400

    # Model prediction
    predictions = model.predict(img_data)
    pred_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_LABELS[pred_class_index]
    confidence = float(predictions[0][pred_class_index]) * 100

    # Construct JSON response
    response = {
        'predicted_class': predicted_class.upper(),
        'confidence': f'{confidence:.2f}%',
        'all_confidences': {
            label.upper(): f'{float(prob) * 100:.2f}%'
            for label, prob in zip(CLASS_LABELS, predictions[0])
        }
    }

    return jsonify(response)


# =============================
# 6️⃣ Run Server
# =============================
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
