import streamlit as st
from model_loader import load_my_model
from preprocess import preprocess_image
from PIL import Image
import numpy as np

# Load the model outside to avoid reloading it every time
model = load_my_model()

st.title("Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    arr = preprocess_image(img)
    input_batch = np.expand_dims(arr, axis=0)

    predictions = model.predict(input_batch)
    st.write("✅ Prediction:", predictions)
