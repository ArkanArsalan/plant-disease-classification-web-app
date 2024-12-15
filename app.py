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
    "Apple Apple scab", "Apple Black rot", "Apple Cedar apple rust", "Apple healthy",
    "Blueberry healthy", "Cherry (including sour) Powdery mildew", "Cherry (including sour) healthy",
    "Corn (maize) Cercospora-leaf-spot Gray-leaf-spot," "Corn (maize) Common rust",
    "Corn (maize) Northern Leaf Blight", "Corn (maize) healthy", "Grape Black rot",
    "Grape Esca (Black Measles)", "Grape Leaf blight (Isariopsis Leaf Spot)", "Grape healthy",
    "Orange Haunglongbing (Citrus greening)", "Peach Bacterial spot", "Peach healthy",
    "Pepper, bell Bacterial spot", "Pepper, bell healthy", "Potato Early blight", "Potato Late blight",
    "Potato healthy", "Raspberry healthy", "Soybean healthy", "Squash Powdery mildew",
    "Strawberry Leaf scorch", "Strawberry healthy", "Tomato Bacterial spot", "Tomato Early blight",
    "Tomato Late blight", "Tomato Leaf Mold", "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite", "Tomato Target Spot",
    "Tomato Tomato Yellow Leaf Curl Virus", "Tomato Tomato mosaic virus", "Tomato healthy"
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
        image = Image.open(file).convert('RGB')
        image = image.resize(IMAGE_SIZE)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class_idx = np.argmax(predictions, axis=-1)[0]
        confidence = np.max(predictions)

        predicted_class_label = CLASS_LABELS[predicted_class_idx]

        return jsonify({
            'predicted_class': predicted_class_label,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)