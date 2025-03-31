import os
import numpy as np
import joblib
from PIL import Image

# Define dataset paths
dataset_paths = {
    "black": r"D:\skin_tone_dataset\black_shade_resized",
    "brown": r"D:\skin_tone_dataset\brown_shade_resized",
    "white": r"D:\skin_tone_dataset\white_shade_resized"
}

data = []  # Store feature vectors
labels = []  # Store corresponding labels

def extract_avg_rgb(image_path):
    """Extract average RGB values from an image."""
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image_array = np.array(image)
    avg_rgb = np.mean(image_array, axis=(0, 1))  # Average across width & height
    return avg_rgb

# Process each category
for label, folder_path in dataset_paths.items():
    if not os.path.exists(folder_path):
        print(f"❌ Warning: Folder not found -> {folder_path}")
        continue
    
    print(f"🔍 Processing folder: {folder_path}")
    image_count = 0

    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        try:
            avg_rgb = extract_avg_rgb(image_path)
            data.append(avg_rgb)  # Append feature (RGB values)
            labels.append(label)  # Append class label
            image_count += 1
        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")

    print(f"✅ {image_count} images processed for {label}")

# Convert lists to numpy arrays
if len(data) == 0:
    print("❌ No images processed. Check dataset paths.")
else:
    data = np.array(data)
    labels = np.array(labels)

    # Save dataset
    joblib.dump({"X": data, "y": labels}, "dataset.pkl")
    print("✅ Dataset saved successfully as dataset.pkl!")
