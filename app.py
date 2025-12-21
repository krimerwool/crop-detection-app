import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH_MAIN = 'crop_classifier_hybrid_finetuned.tflite'
LABELS_MAIN = {0: 'Maize', 1: 'Mustard', 2: 'Wheat'}

MODEL_PATH_MAIZE = 'maize_stage_classifier.tflite'
LABELS_MAIZE_STAGE = {0: 'Maturity', 1: 'Reproductive', 2: 'Seedling', 3: 'Vegetative'}

MODEL_PATH_WHEAT = 'wheat_stage_classifier (1).tflite'
LABELS_WHEAT_STAGE = {0: 'Flowering', 1: 'Jointing', 2: 'Ripening', 3: 'Seedling'}

AGE_ESTIMATES = {
    'Maize': {
        'Seedling': '0 - 2 Weeks',
        'Vegetative': '3 - 7 Weeks',
        'Reproductive': '8 - 10 Weeks',
        'Maturity': '11 - 15+ Weeks'
    },
    'Wheat': {
        'Seedling': '1 - 3 Weeks',
        'Jointing': '4 - 6 Weeks',
        'Flowering': '7 - 9 Weeks',
        'Ripening': '10 - 14+ Weeks'
    }
}

@st.cache_resource
def load_all_models():
    # Load Main Model
    interpreter_main = tf.lite.Interpreter(model_path=MODEL_PATH_MAIN)
    interpreter_main.allocate_tensors()
    
    # Load Maize Stage Model
    interpreter_maize = tf.lite.Interpreter(model_path=MODEL_PATH_MAIZE)
    interpreter_maize.allocate_tensors()
    
    # Load Wheat Stage Model
    interpreter_wheat = tf.lite.Interpreter(model_path=MODEL_PATH_WHEAT)
    interpreter_wheat.allocate_tensors()
    
    return interpreter_main, interpreter_maize, interpreter_wheat

def predict_with_interpreter(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocessing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.expand_dims(normalized_image_array, axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    
    # Result
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0]

st.set_page_config(page_title="Crop & Age Detector", page_icon="🌾")
st.title("CROP & AGE DETECTOR ")
st.write("Upload a crop photo to identify its Type, Growth Phase, and Estimated Age.")

# Load models once
interpreter_main, interpreter_maize, interpreter_wheat = load_all_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Thinking...")
    
    # --- STEP 1: DETECT CROP TYPE ---
    preds_main = predict_with_interpreter(interpreter_main, image)
    crop_idx = np.argmax(preds_main)
    crop_conf = np.max(preds_main) * 100
    detected_crop = LABELS_MAIN[crop_idx]
    
    # Display Crop Result
    st.success(f"**Crop Identified:** {detected_crop} ({crop_conf:.1f}%)")
    
    # --- STEP 2: CASCADE LOGIC (Detect Phase) ---
    if detected_crop == "Maize":
        st.info("⚡ Analyzing Maize Growth Stage...")
        
        preds_stage = predict_with_interpreter(interpreter_maize, image)
        stage_idx = np.argmax(preds_stage)
        detected_stage = LABELS_MAIZE_STAGE[stage_idx]
        estimated_age = AGE_ESTIMATES['Maize'].get(detected_stage, "Unknown")
        
        col1, col2 = st.columns(2)
        col1.metric("Growth Phase", detected_stage)
        col2.metric("Estimated Age", estimated_age)
        
    elif detected_crop == "Wheat":
        st.info("⚡ Analyzing Wheat Growth Stage...")
        
        preds_stage = predict_with_interpreter(interpreter_wheat, image)
        stage_idx = np.argmax(preds_stage)
        detected_stage = LABELS_WHEAT_STAGE[stage_idx]
        estimated_age = AGE_ESTIMATES['Wheat'].get(detected_stage, "Unknown")
        
        col1, col2 = st.columns(2)
        col1.metric("Growth Phase", detected_stage)
        col2.metric("Estimated Age", estimated_age)
        
    elif detected_crop == "Mustard":
        st.warning("⚠️ Growth stage detection for Mustard is currently under development.")

    st.write("---")
    with st.expander("See Detailed Probabilities"):
        st.write("Crop Type Probabilities:")
        st.bar_chart({label: float(preds_main[idx]) for idx, label in LABELS_MAIN.items()})
