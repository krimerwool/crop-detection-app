import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


MODEL_PATH = 'crop_classifier.tflite'
LABELS = {0: 'Maize', 1: 'Mustard', 2: 'Wheat'}

# 2. LOAD MODEL FUNCTION (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# 3. PREDICTION FUNCTION
def predict_image(interpreter, image):
    # Get input and output details from the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # -- Preprocessing --
    # Resize to the same size we used in training (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to array
    img_array = np.asarray(image)
    
    # Normalize (divide by 255.0) - CRITICAL STEP
    # Converting to float32 is usually required by TFLite
    normalized_image_array = (img_array.astype(np.float32) / 255.0)

    # Add Batch Dimension: Shape becomes (1, 224, 224, 3)
    data = np.expand_dims(normalized_image_array, axis=0)

    # -- Inference --
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    
    # -- Get Result --
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0] # Return the first (and only) result

# 4. STREAMLIT UI LAYOUT
st.title("🌾 Crop Type Detector")
st.write("Upload a photo of a crop (Maize, Mustard, or Wheat) to detect its type.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Crop Image', use_column_width=True)
    
    st.write("Analyzing...")
    
    # Load model and predict
    interpreter = load_model()
    predictions = predict_image(interpreter, image)
    
    # Get the highest confidence class
    predicted_index = np.argmax(predictions)
    confidence_score = np.max(predictions) * 100
    predicted_label = LABELS[predicted_index]

    # Display Result
    # if confidence_score < 50:
    #     st.error(f"⚠️ Low Confidence ({confidence_score:.1f}%). Are you sure this is a crop?")
    # else:
    st.success(f"✅ Prediction: **{predicted_label}**")
    st.info(f"Confidence: {confidence_score:.2f}%")
    
    # Show detailed probability bars
    st.write("---")
    st.write("Detailed Analysis:")
    # Simple bar chart of probabilities
    st.bar_chart({label: float(predictions[idx]) for idx, label in LABELS.items()})