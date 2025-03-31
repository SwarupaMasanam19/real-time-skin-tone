import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("skin_tone_model.keras")

# Define class labels (change these based on your dataset)
class_labels = ["Black", "Brown", "White", "Fair", "Dark Brown", "Light Brown", "Medium Brown", "Deep Dark"]  

# Load and preprocess the test image
img_path = "black.jpg"  # Change this to your test image path
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Predict class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class

# Print skin tone
print(f"Predicted Skin Tone: {class_labels[predicted_class]}")
