import sys
import os
import streamlit as st
import tempfile

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from backend.predictor import predict_digit

st.title("CNN Digit Recognition")

uploaded_file=st.file_uploader(
    "Upload Digit Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file:

    st.image(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    digit, confidence = predict_digit(temp_path)
    f"Predicted Digit: {digit}"

    st.write(
        f"Confidence: {confidence*100:.2f}%"
    )