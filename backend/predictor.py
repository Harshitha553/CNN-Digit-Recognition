import os
from tensorflow.keras.models import load_model
from backend.preprocess import preprocess_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_development",
    "digit_model.h5"
)

print("Loading model from:", MODEL_PATH)

model = load_model(MODEL_PATH)

def predict_digit(image_path):
    img = preprocess_image(image_path)

    prediction = model.predict(img)

    digit = prediction.argmax()
    confidence = prediction.max()

    return digit, confidence