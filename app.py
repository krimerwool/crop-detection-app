import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
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
    """Loads all TFLite models into memory once."""
    try:
        interpreter_main = tf.lite.Interpreter(model_path=MODEL_PATH_MAIN)
        interpreter_main.allocate_tensors()
    except Exception as e:
        st.error(f"Error loading Main Model ({MODEL_PATH_MAIN}): {e}")
        return None, None, None

    try:
        interpreter_maize = tf.lite.Interpreter(model_path=MODEL_PATH_MAIZE)
        interpreter_maize.allocate_tensors()
    except:
        interpreter_maize = None 

    try:
        interpreter_wheat = tf.lite.Interpreter(model_path=MODEL_PATH_WHEAT)
        interpreter_wheat.allocate_tensors()
    except:
        interpreter_wheat = None 

    return interpreter_main, interpreter_maize, interpreter_wheat

def predict_with_interpreter(interpreter, image):
    """Runs inference on a single image using the given interpreter."""
    if interpreter is None:
        return np.zeros(10) 
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocessing (Standard 0-1 Normalization for your models)
    size = (224, 224) 
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalize (0 to 1)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.expand_dims(normalized_image_array, axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    
    # Result
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0]

st.set_page_config(page_title="Crop & Age Detector", page_icon="🌾", layout="wide")

# Sidebar for Mode Selection
mode = st.sidebar.radio("Select Mode:", ["Single Image Analysis", "Batch Processing (Bulk)"])

interpreter_main, interpreter_maize, interpreter_wheat = load_all_models()

if interpreter_main is None:
    st.stop()


if mode == "Single Image Analysis":
    st.title("🌾 Crop Doctor: Single Analysis")
    st.write("Upload a crop photo to identify its Type and Growth Phase.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            st.write("### Analysis Results")
            
            # --- DETECT CROP ---
            preds_main = predict_with_interpreter(interpreter_main, image)
            crop_idx = np.argmax(preds_main)
            crop_conf = np.max(preds_main) * 100
            
            detected_crop = LABELS_MAIN.get(crop_idx, "Unknown")
            
            if crop_conf < 40: 
                st.warning(f"**Uncertain:** Looks like {detected_crop} ({crop_conf:.1f}%), but confidence is low.")
            else:
                st.success(f"**Crop Identified:** {detected_crop} ({crop_conf:.1f}%)")
            
            # --- CASCADE LOGIC ---
            if detected_crop == "Maize":
                if interpreter_maize:
                    preds_stage = predict_with_interpreter(interpreter_maize, image)
                    stage_idx = np.argmax(preds_stage)
                    detected_stage = LABELS_MAIZE_STAGE.get(stage_idx, "Unknown")
                    estimated_age = AGE_ESTIMATES['Maize'].get(detected_stage, "Unknown")
                    
                    st.info(f"**Growth Stage:** {detected_stage}")
                    st.caption(f"Estimated Age: {estimated_age}")
                    
            elif detected_crop == "Wheat":
                if interpreter_wheat:
                    preds_stage = predict_with_interpreter(interpreter_wheat, image)
                    stage_idx = np.argmax(preds_stage)
                    detected_stage = LABELS_WHEAT_STAGE.get(stage_idx, "Unknown")
                    estimated_age = AGE_ESTIMATES['Wheat'].get(detected_stage, "Unknown")
                    
                    st.info(f"**Growth Stage:** {detected_stage}")
                    st.caption(f"Estimated Age: {estimated_age}")
            
            elif detected_crop == "Mustard":
                 st.warning("⚠️ Growth stage detection for Mustard is currently under development.")

            st.write("---")
            st.write("Confidence Distribution:")
            chart_data = {label: float(preds_main[idx]) for idx, label in LABELS_MAIN.items() if idx < len(preds_main)}
            st.bar_chart(chart_data)


elif mode == "Batch Processing (Bulk)":
    st.title("📂 Batch Processing")
    st.write("Upload multiple images at once to classify them in bulk.")

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"**Total Images Selected:** {len(uploaded_files)}")
        
        if st.button("Start Processing"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Statistics Counters
            count_maize = 0
            count_wheat = 0
            count_mustard = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update Progress
                status_text.text(f"Processing image {i+1} of {len(uploaded_files)}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                try:
                    # Load Image
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # 1. Detect Crop
                    preds = predict_with_interpreter(interpreter_main, image)
                    crop_idx = np.argmax(preds)
                    confidence = np.max(preds) * 100
                    predicted_crop = LABELS_MAIN.get(crop_idx, "Unknown")
                    
                    # 2. Detect Stage (Cascade Logic)
                    detected_stage = "N/A"
                    estimated_age = "N/A"
                    
                    if predicted_crop == "Maize" and interpreter_maize:
                        count_maize += 1
                        preds_stage = predict_with_interpreter(interpreter_maize, image)
                        stage_idx = np.argmax(preds_stage)
                        detected_stage = LABELS_MAIZE_STAGE.get(stage_idx, "Unknown")
                        estimated_age = AGE_ESTIMATES['Maize'].get(detected_stage, "Unknown")
                        
                    elif predicted_crop == "Wheat" and interpreter_wheat:
                        count_wheat += 1
                        preds_stage = predict_with_interpreter(interpreter_wheat, image)
                        stage_idx = np.argmax(preds_stage)
                        detected_stage = LABELS_WHEAT_STAGE.get(stage_idx, "Unknown")
                        estimated_age = AGE_ESTIMATES['Wheat'].get(detected_stage, "Unknown")
                        
                    elif predicted_crop == "Mustard":
                        count_mustard += 1
                        detected_stage = "Development"

                    # Append to results
                    results.append({
                        "Filename": uploaded_file.name,
                        "Predicted Crop": predicted_crop,
                        "Confidence (%)": round(confidence, 2),
                        "Growth Stage": detected_stage,
                        "Est. Age": estimated_age
                    })
                    
                except Exception as e:
                    results.append({
                        "Filename": uploaded_file.name,
                        "Predicted Crop": "Error",
                        "Confidence (%)": 0.0,
                        "Growth Stage": str(e),
                        "Est. Age": ""
                    })

            progress_bar.empty()
            status_text.success("✅ Batch Processing Complete!")
            
            # --- SUMMARY METRICS ---
            st.markdown("### 📊 Summary Report")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Images", len(uploaded_files))
            col2.metric("Maize", count_maize)
            col3.metric("Wheat", count_wheat)
            col4.metric("Mustard", count_mustard)
            
            # --- DETAILED TABLE ---
            st.write("### 📝 Detailed Logs")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report as CSV",
                csv,
                "crop_analysis_report.csv",
                "text/csv",
                key='download-csv'
            )
