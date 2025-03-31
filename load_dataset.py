import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define dataset paths
dataset_paths = {
    "brown": "D:/skin_tone_dataset/brown_shade_resized",
    "black": "D:/skin_tone_dataset/black_shade_resized",
    "white": "D:/skin_tone_dataset/white_shade_resized"
}

label_map = {"brown": 0, "black": 1, "white": 2}

X = []
y = []

# Check if dataset is loading
for label, path in dataset_paths.items():
    print(f"Checking folder: {path}")
    if not os.path.exists(path):
        print(f"❌ Error: Path does not exist - {path}")
    else:
        files = os.listdir(path)
        print(f"✅ Found {len(files)} images in {label}")

    for file in files:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Warning: Could not load {img_path}")
        else:
            img = cv2.resize(img, (256, 256))  # Ensure all images are same size
            X.append(img)
            y.append(label_map[label])

# Convert to NumPy arrays
X = np.array(X, dtype="float32") / 255.0  # Normalize pixel values
y = np.array(y, dtype="int")
y = to_categorical(y, num_classes=3)

# Check dataset size
print(f"\n✅ Total images loaded: {len(X)}")

# Train-test split
if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Training images: {len(X_train)}, Testing images: {len(X_test)}")
else:
    print("❌ Error: No images loaded!")

