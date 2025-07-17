import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your CNN model
model = tf.keras .models.load_model('your_model.h5')

# Function to make predictions
def make_prediction(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize the image to the input size of your model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    return predictions

# Create the Streamlit app
st.title("Fresh Fish Classifier")
st.write("Upload an image of a fish to classify it as fresh or non-fresh")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Make prediction
    predictions = make_prediction(uploaded_file)
    # Get the class with the highest probability
    class_index = np.argmax(predictions)
    if class_index == 0:
        st.write("The fish is fresh!")
    else:
        st.write("The fish is non-fresh!")
