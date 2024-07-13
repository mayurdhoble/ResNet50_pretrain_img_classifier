import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.utils import load_img, img_to_array  # Assuming you have Keras installed
import numpy as np

# Load the pre-trained ResNet50 model (one-time load)
model = ResNet50(weights='imagenet')

def classify_image(img_path):
  """Classifies an image using the ResNet50 model."""
  img = load_img(img_path, target_size=(224, 224))
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)
  predictions = model.predict(img_array)
  decoded_predictions = decode_predictions(predictions, top=3)[0]
  return decoded_predictions

st.title("Image Classification with Streamlit")
uploaded_file = st.file_uploader("Choose an image to classify", type="jpg")

if uploaded_file is not None:
  # Read the uploaded image
  img_bytes = uploaded_file.read()
  st.image(img_bytes)

  # Classify the image
  predictions = classify_image(uploaded_file.name)

  # Display predictions
  st.subheader("Top 3 Predictions:")
  for i, (imagenet_id, label, score) in enumerate(predictions):
    st.write(f"{i + 1}: {label} ({score:.2f})")

