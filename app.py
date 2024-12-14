from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model("mobilenetv2_model.keras", compile=False)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Define input image size (224, 224, 3 for MobileNetV2)
IMAGE_SIZE = (224, 224)

# List of class labels
CLASS_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    try:
        # Load and preprocess the image
        image = Image.open(file).convert('RGB')
        image = image.resize(IMAGE_SIZE)
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(image)
        predicted_class_idx = np.argmax(predictions, axis=-1)[0]
        confidence = np.max(predictions)

        # Map index to class label
        predicted_class_label = CLASS_LABELS[predicted_class_idx]

        # Return result
        return jsonify({
            'predicted_class': predicted_class_label,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
