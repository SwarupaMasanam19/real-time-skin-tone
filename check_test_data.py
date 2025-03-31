import joblib

# Load the saved test data
try:
    test_data = joblib.load("test_data.pkl")
    print("✅ Loaded test data successfully!")
    print("📊 Test Data Features:", test_data["X_test"])
except FileNotFoundError:
    print("❌ Error: test_data.pkl not found!")
