import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


MODEL_PATH = 'crop_classifier_finetuned.tflite'
LABELS = {0: 'Maize', 1: 'Mustard', 2: 'Wheat'}

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    img_array = np.asarray(image)

    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.expand_dims(normalized_image_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0] # Return the first (and only) result

st.title("🌾 Crop Type Detector")
st.write("Upload a photo of a crop (Maize, Mustard, or Wheat) to detect its type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Crop Image', use_column_width=True)
    
    st.write("Analyzing...")

    interpreter = load_model()
    predictions = predict_image(interpreter, image)

    predicted_index = np.argmax(predictions)
    confidence_score = np.max(predictions) * 100
    predicted_label = LABELS[predicted_index]

    # if confidence_score < 50:
    #     st.error(f"⚠️ Low Confidence ({confidence_score:.1f}%). Are you sure this is a crop?")
    # else:
    st.success(f"✅ Prediction: **{predicted_label}**")
    st.info(f"Confidence: {confidence_score:.2f}%")

    st.write("---")
    st.write("Detailed Analysis:")

    st.bar_chart({label: float(predictions[idx]) for idx, label in LABELS.items()})
