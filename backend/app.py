import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model("skin_tone_model.keras")
print("✅ Model loaded successfully")

# Define class names
class_names = ['Black', 'Brown', 'White', 'Fair', 'Dark Brown', 'Light Brown', 'Medium Brown', 'Deep Dark']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        image_path = os.path.join("temp_image.jpg")
        image_file.save(image_path)

        # Preprocess the image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        # Remove temp file
        os.remove(image_path)

        return jsonify({'predicted_skin_tone': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)
