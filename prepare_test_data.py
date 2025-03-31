import joblib
import numpy as np

# 🔹 Extract real image features (RGB values)
X_test = np.array([[92.6744, 73.2376, 57.2276]])  # Example extracted values

# 🔹 Normalize RGB values (0-255 → 0-1)
X_test = X_test / 255.0

# 🔹 Save test data
joblib.dump({"X_test": X_test}, "test_data.pkl")
print("✅ Test data saved with normalized features!")
