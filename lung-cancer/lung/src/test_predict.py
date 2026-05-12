import joblib
import numpy as np
import os

# Print current working directory to see where we are
print(f"Current working directory: {os.getcwd()}")

# Check if file exists
model_path = "src/model.pkl"
if os.path.exists(model_path):
    print(f"✓ Found model at: {model_path}")
    print(f"File size: {os.path.getsize(model_path)} bytes")
else:
    print(f"✗ Model not found at: {model_path}")
    # List files in src directory
    if os.path.exists("src"):
        print("Files in src directory:")
        for file in os.listdir("src"):
            print(f"  - {file}")
    else:
        print("src directory doesn't exist!")

# Try loading with full error details
try:
    model = joblib.load(model_path)
    print("✓ Model loaded successfully!")
    print(f"✓ Model type: {type(model)}")
    
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    traceback.print_exc()