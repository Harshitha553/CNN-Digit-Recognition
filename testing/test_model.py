from backend.predictor import predict_digit

image_path="testing/sample_images/digit1.png"

digit,confidence=predict_digit(image_path)

print("Prediction:",digit)

print("Confidence:",confidence)