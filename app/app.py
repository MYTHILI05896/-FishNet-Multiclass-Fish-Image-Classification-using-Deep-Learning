import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Title and Description
st.title("🐟 FishNet: Multiclass Fish Image Classifier")
st.markdown("Upload a fish image to classify it into one of 11 fish species.")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join("notebooks", "best_model_cnn.h5")

    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Define class labels
class_labels = [
    'animal fish', 
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.markdown("### 🧠 Prediction")
    st.success(f"🎯 The model predicts this is a **{predicted_class}**.")
