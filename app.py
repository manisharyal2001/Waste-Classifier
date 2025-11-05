# app.py - Flask Backend Server (with Object Cropping)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import json
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2

## ⚙️ Configuration
MODEL_PATH = 'waste_classifier_model.h5'
CLASSES_PATH = 'class_indices.json'
TARGET_SIZE = (224, 224)

# Load Model and Classes
model = None
CLASS_LABELS = []
try:
    # Disable eager execution for better performance in a deployment context
    # tf.compat.v1.disable_eager_execution() 
    
    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    
    # Create the sorted list of class labels from the dictionary keys
    CLASS_LABELS = sorted(class_indices, key=class_indices.get)
    print("Model and Class Labels loaded successfully.")
    print("Class Labels:", CLASS_LABELS)
except Exception as e:
    print(f"❌ Critical Error during setup: Failed to load model or classes. Ensure '{MODEL_PATH}' and '{CLASSES_PATH}' exist. Error: {e}")


## 🌐 Flask App Setup
app = Flask(__name__)
CORS(app)  


## 🛠️ Helper Functions

def crop_object(pil_image):
    """Crop the main object in the image using OpenCV contours."""
    try:
        # Convert PIL (RGB) → NumPy (BGR for OpenCV)
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale and blur
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold for binary mask 
        # Use simple Otsu's thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Largest contour = main object
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Add padding for better object capture (e.g., 10 pixels)
            padding = 10 
            x = max(0, x - padding)
            y = max(0, y - padding)
            
            # Ensure crop doesn't go beyond image bounds
            x_end = min(img_cv.shape[1], x + w + 2 * padding)
            y_end = min(img_cv.shape[0], y + h + 2 * padding)
            
            cropped = img_cv[y:y_end, x:x_end]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped)
        else:
            return pil_image  # fallback to original if no contours found
    except Exception as e:
        # Fallback in case OpenCV fails to process the image correctly
        print(f"⚠️ Error during cropping, falling back to original image: {e}")
        return pil_image


def preprocess_image(img_file):
    """Read and preprocess the image for model prediction."""
    try:
        # Read the image from the file stream
        img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
        
        # Apply the object cropping
        img = crop_object(img)  
        
        # Resize to model's expected input size
        img = img.resize(TARGET_SIZE)
        
        # Convert to array, normalize, and add batch dimension
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"⚠️ Error during image preprocessing: {e}")
        return None


## 🌐 API Routes

@app.route('/')
def home():
    """Simple root route to confirm the server is running."""
    html_content = """
    <div style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
        <h1 style="color: #1e40af;">✅ Flask Server is Running!</h1>
        <p style="font-size: 1.2em;">This server provides the prediction API for the Smart Waste Classifier.</p>
        <p style="font-size: 1.5em; font-weight: bold; color: #10b981;">To use the application, please open the <code style="background-color: #f0f0f0; padding: 4px 6px; border-radius: 4px;">index.html</code> file directly in your browser.</p>
        <p style="margin-top: 30px; color: #888;">The API route is <code style="background-color: #f0f0f0; padding: 3px 5px; border-radius: 3px;">/predict</code> (POST).</p>
    </div>
    """
    return render_template_string(html_content)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['file']
    img_data = preprocess_image(file)

    if img_data is None:
        return jsonify({'error': 'Invalid image format or processing error.'}), 400

    # Model prediction
    predictions = model.predict(img_data)
    
    # Keras/TensorFlow prediction output is a numpy array
    probabilities = predictions[0]

    pred_class_index = np.argmax(probabilities)
    predicted_class = CLASS_LABELS[pred_class_index]
    confidence = float(probabilities[pred_class_index]) * 100

    # Construct JSON response
    response = {
        'predicted_class': predicted_class.upper(),
        # Confidence as a clean string percentage
        'confidence': f'{confidence:.2f}%', 
        # Detailed confidences for all classes
        'all_confidences': {
            label.upper(): f'{float(prob) * 100:.2f}%'
            for label, prob in zip(CLASS_LABELS, probabilities)
        }
    }

    return jsonify(response)


## 🖥️ Run Server (LOCAL CONFIG)

if __name__ == '__main__':
   # Running with debug=True reloads the server automatically on code changes
    app.run(host='127.0.0.1', port=5000, debug=True)