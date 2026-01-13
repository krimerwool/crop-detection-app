import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


MODEL_PATH_BARREN = 'barren_vs_crop_model.tflite'
MODEL_PATH_MAIN = 'crop_classifier_5crops_v1.tflite'
MODEL_PATH_MAIZE = 'maize_stage_classifier.tflite'
MODEL_PATH_WHEAT = 'wheat_stage_classifier (1).tflite' 

LABELS_5_CROPS = {0: 'Maize', 1: 'Mustard', 2: 'Rice', 3: 'Sugarcane', 4: 'Wheat'}
LABELS_MAIZE_STAGE = {0: 'Maturity', 1: 'Reproductive', 2: 'Seedling', 3: 'Vegetative'}
LABELS_WHEAT_STAGE = {0: 'Flowering', 1: 'Jointing', 2: 'Ripening', 3: 'Seedling'}

AGE_ESTIMATES = {
    'Maize': {
        'Seedling': '0 - 2 Weeks', 'Vegetative': '3 - 7 Weeks',
        'Reproductive': '8 - 10 Weeks', 'Maturity': '11 - 15+ Weeks'
    },
    'Wheat': {
        'Seedling': '1 - 3 Weeks', 'Jointing': '4 - 6 Weeks',
        'Flowering': '7 - 9 Weeks', 'Ripening': '10 - 14+ Weeks'
    }
}

@st.cache_resource
def load_all_models():
    """Loads all TFLite models into memory once."""
    interpreters = {}
    model_map = {
        'barren': MODEL_PATH_BARREN,
        'main': MODEL_PATH_MAIN,
        'maize': MODEL_PATH_MAIZE,
        'wheat': MODEL_PATH_WHEAT
    }

    for key, path in model_map.items():
        try:
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            interpreters[key] = interpreter
        except Exception as e:
            st.error(f"Error loading {key} model ({path}): {e}")
            interpreters[key] = None
    return interpreters

def predict_with_interpreter(interpreter, image, is_binary=False, normalize=True):
    """
    Runs inference.
    normalize=True  -> (img / 255.0) -> Range 0.0 to 1.0 (For Crop/Stage models)
    normalize=False -> (img)         -> Range 0.0 to 255.0 (For Barren model)
    """
    if interpreter is None:
        return None

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)

    img_array = img_array.astype(np.float32)
    
    if normalize:
        input_data = img_array / 255.0
    else:
        input_data = img_array
        
    data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    
    # Result
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    if is_binary:
        return prediction[0][0] 
    else:
        return prediction[0]   

# UI SETUP

st.set_page_config(page_title="Crop-Detector-Demo", page_icon="🌾", layout="wide")

mode = st.sidebar.radio("Select Mode:", ["Single Image Analysis", "Batch Processing (Bulk)"])
models = load_all_models()


if mode == "Single Image Analysis":
    st.title("Crop-Detector-Demo")
    st.markdown("**Workflow:** Barren Check → Crop ID → Growth Stage")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            st.write("### Analysis Results")

            probability = predict_with_interpreter(models['barren'], image, is_binary=True, normalize=False)

            if probability > 0.5:
                is_barren = False
                st.success(f"✅ **Land Status:** Non-Barren (Crop Detected)")
                st.caption(f"Barren Confidence: {probability*100:.1f}%") 
            else:
                # IT IS BARREN
                is_barren = True
                conf = 1.0 - probability
                st.error(f"🟫 **Result: Barren Land Detected**")
                st.write(f"Confidence: {conf*100:.1f}%")
                st.warning("Analysis stopped.")

            if not is_barren:

                preds_main = predict_with_interpreter(models['main'], image, normalize=True)
                crop_idx = np.argmax(preds_main)
                crop_conf = np.max(preds_main) * 100
                detected_crop = LABELS_5_CROPS.get(crop_idx, "Unknown")
                
                st.info(f"**Identified Crop:** {detected_crop} ({crop_conf:.1f}%)")
                
                with st.expander("See Probabilities"):
                    st.bar_chart({label: float(preds_main[idx]) for idx, label in LABELS_5_CROPS.items()})

                # STAGE DETECTION
                if detected_crop == "Maize" and models['maize']:
                    st.markdown("---")
                    preds_stage = predict_with_interpreter(models['maize'], image, normalize=True)
                    stage_idx = np.argmax(preds_stage)
                    detected_stage = LABELS_MAIZE_STAGE.get(stage_idx, "Unknown")
                    estimated_age = AGE_ESTIMATES['Maize'].get(detected_stage, "Unknown")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Growth Phase", detected_stage)
                    c2.metric("Est. Age", estimated_age)
                
                elif detected_crop == "Wheat" and models['wheat']:
                    st.markdown("---")
                    preds_stage = predict_with_interpreter(models['wheat'], image, normalize=True)
                    stage_idx = np.argmax(preds_stage)
                    detected_stage = LABELS_WHEAT_STAGE.get(stage_idx, "Unknown")
                    estimated_age = AGE_ESTIMATES['Wheat'].get(detected_stage, "Unknown")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Growth Phase", detected_stage)
                    c2.metric("Est. Age", estimated_age)
                
                elif detected_crop in ["Mustard", "Rice", "Sugarcane"]:
                    st.markdown("---")
                    st.caption(f"ℹ️ Stage detection for **{detected_crop}** coming soon.")

# BATCH PROCESSING LOGIC 

elif mode == "Batch Processing (Bulk)":
    st.title("📂 Batch Processing")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files and st.button("Start Processing"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        counts = {'Barren': 0, 'Maize': 0, 'Wheat': 0, 'Mustard': 0, 'Rice': 0, 'Sugarcane': 0}
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing image {i+1} of {len(uploaded_files)}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            try:
                image = Image.open(uploaded_file).convert('RGB')

                prob = predict_with_interpreter(models['barren'], image, is_binary=True, normalize=False)
                
                if prob <= 0.5: 
                    counts['Barren'] += 1
                    results.append({
                        "Filename": uploaded_file.name, "Status": "Barren Land",
                        "Crop": "N/A", "Confidence": f"{(1-prob)*100:.1f}%"
                    })
                    continue 

                preds = predict_with_interpreter(models['main'], image, normalize=True)
                crop_idx = np.argmax(preds)
                crop_conf = np.max(preds) * 100
                detected_crop = LABELS_5_CROPS.get(crop_idx, "Unknown")
                
                if detected_crop in counts: counts[detected_crop] += 1

                detected_stage = "N/A"
                est_age = "N/A"
                
                if detected_crop == "Maize" and models['maize']:
                    s_preds = predict_with_interpreter(models['maize'], image, normalize=True)
                    detected_stage = LABELS_MAIZE_STAGE.get(np.argmax(s_preds), "Unknown")
                    est_age = AGE_ESTIMATES['Maize'].get(detected_stage, "Unknown")
                elif detected_crop == "Wheat" and models['wheat']:
                    s_preds = predict_with_interpreter(models['wheat'], image, normalize=True)
                    detected_stage = LABELS_WHEAT_STAGE.get(np.argmax(s_preds), "Unknown")
                    est_age = AGE_ESTIMATES['Wheat'].get(detected_stage, "Unknown")
                
                results.append({
                    "Filename": uploaded_file.name, "Status": "Cultivated",
                    "Crop": detected_crop, "Confidence": f"{crop_conf:.1f}%",
                    "Stage": detected_stage, "Age": est_age
                })
                
            except Exception as e:
                 results.append({"Filename": uploaded_file.name, "Status": "Error", "Crop": str(e)})

        progress_bar.empty()
        status_text.success("✅ Complete!")
        
        st.write("### 📊 Summary")
        cols = st.columns(len(counts))
        for idx, (key, val) in enumerate(counts.items()):
            cols[idx].metric(key, val)
        
        st.write("### 📝 Logs")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
