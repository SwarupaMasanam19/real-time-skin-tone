import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
try:
    model = load_model("skin_tone_model.keras")  # Ensure the model file exists
except Exception as e:
    print("Error: Could not load the model. Check the file path and format.")
    print("Details:", e)
    exit()

# Updated skin tone labels
classes = ["Fair", "Light Brown", "Medium Brown", "Ebony", "Tan", "Deep Dark", "Dark Brown", "Rich Dark"]

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Ensures better performance on Windows
cap.set(3, 1920)  # Width
cap.set(4, 1080)  # Height

print("Press 'Space' to capture the image.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press Space to capture and close camera
    if key == 32:
        # Resize the image to match model input shape (128x128)
        img = cv2.resize(frame, (128, 128))
        img = img / 255.0  # Normalize pixel values (scale between 0 and 1)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict skin tone
        prediction = model.predict(img)
        print("Raw Model Output:", prediction)

        predicted_class = np.argmax(prediction)  # Get index of highest probability

        if predicted_class >= len(classes):  # Fix out-of-range error
            print("Error: Predicted index out of range!")
        else:
            predicted_tone = classes[predicted_class]
            print("Predicted Skin Tone:", predicted_tone)

        # Delay for a moment so the user can see the output
        cv2.waitKey(2000)

        # Release camera & close window after capturing
        cap.release()
        cv2.destroyAllWindows()
        break
