import os
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")

def load_my_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model
