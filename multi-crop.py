import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Paths to your models
MODEL_PATH_CROP = 'final_model.tflite'
MODEL_PATH_BARREN = 'barren_vs_crop_model.tflite'

# Input Sizes
SIZE_CROP_MODEL = 260    # EfficientNetV2
SIZE_BARREN_MODEL = 224  # MobileNetV2

# Labels
LABELS_CROP = {0: 'Maize', 1: 'Rice', 2: 'Soybean', 3: 'Sugarcane'}

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    """Loads both TFLite models into memory"""
    interpreters = {}
    try:
        # Load Crop Model
        crop_interp = tf.lite.Interpreter(model_path=MODEL_PATH_CROP)
        crop_interp.allocate_tensors()
        interpreters['crop'] = crop_interp
        
        # Load Barren Model
        barren_interp = tf.lite.Interpreter(model_path=MODEL_PATH_BARREN)
        barren_interp.allocate_tensors()
        interpreters['barren'] = barren_interp
        
        return interpreters
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ==========================================
# 3. PREPROCESSING FUNCTIONS (CRITICAL)
# ==========================================

def preprocess_for_barren(image_np):
    """
    MobileNetV2 Preprocessing:
    - Resize to 224x224
    - Normalize to [-1, 1] using (img / 127.5) - 1.0
    """
    img = cv2.resize(image_np, (SIZE_BARREN_MODEL, SIZE_BARREN_MODEL))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0  # Specific normalization for MobileNet
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_for_crop(image_np):
    """
    EfficientNetV2 Preprocessing:
    - Resize to 260x260
    - Keep range [0, 255] (Model handles its own scaling)
    """
    img = cv2.resize(image_np, (SIZE_CROP_MODEL, SIZE_CROP_MODEL))
    img = img.astype(np.float32)
    # NO DIVISION by 255.0 here because include_preprocessing=True was used
    img = np.expand_dims(img, axis=0)
    return img

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================

def is_sky_or_background(region_image, sky_threshold=0.4):
    """Detect if region is mostly sky to skip processing"""
    hsv = cv2.cvtColor(region_image, cv2.COLOR_RGB2HSV)
    # Check top 40% of the region only
    top_portion = hsv[:int(region_image.shape[0] * 0.4), :]
    
    # Blue Sky
    blue_mask = cv2.inRange(top_portion, np.array([90, 50, 100]), np.array([130, 255, 255]))
    # Gray/White Sky
    gray_mask = cv2.inRange(top_portion, np.array([0, 0, 180]), np.array([180, 40, 255]))
    
    combined_mask = cv2.bitwise_or(blue_mask, gray_mask)
    if combined_mask.size == 0: return False
    
    return (np.count_nonzero(combined_mask) / combined_mask.size) > sky_threshold

def predict_region_cascaded(interpreters, region_image):
    """
    Logic:
    1. Check Barren Model.
    2. If Barren -> Return "Barren Land".
    3. If Crop -> Run Crop Model -> Return "Maize/Rice/etc".
    """
    
    # --- STEP 1: BARREN CHECK ---
    input_barren = preprocess_for_barren(region_image)
    
    barren_interp = interpreters['barren']
    b_in = barren_interp.get_input_details()
    b_out = barren_interp.get_output_details()
    
    barren_interp.set_tensor(b_in[0]['index'], input_barren)
    barren_interp.invoke()
    
    # Get probability that it is a CROP (Non-Barren)
    prob_is_crop = barren_interp.get_tensor(b_out[0]['index'])[0][0]
    
    # Threshold Logic from your snippet:
    # If prob > 0.5, it is Non-Barren (Crop).
    # If prob <= 0.5, it is Barren.
    
    if prob_is_crop <= 0.5:
        confidence = 1.0 - prob_is_crop
        return -1, "Barren Land", confidence

    # --- STEP 2: CROP CLASSIFICATION (Only if Step 1 passed) ---
    input_crop = preprocess_for_crop(region_image)
    
    crop_interp = interpreters['crop']
    c_in = crop_interp.get_input_details()
    c_out = crop_interp.get_output_details()
    
    crop_interp.set_tensor(c_in[0]['index'], input_crop)
    crop_interp.invoke()
    
    predictions = crop_interp.get_tensor(c_out[0]['index'])[0]
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    
    return predicted_idx, LABELS_CROP[predicted_idx], confidence

# ==========================================
# 5. GRID PROCESSING
# ==========================================

def analyze_image_grid(image_pil, interpreters, grid_size=3):
    """Splits image into grid and analyzes each cell"""
    image_np = np.array(image_pil)
    img_height, img_width = image_np.shape[:2]
    
    cell_h = img_height // grid_size
    cell_w = img_width // grid_size
    
    detections = []
    
    # We use two passes: Aligned (0,0) and Offset (0.5, 0.5) to catch boundary objects
    offsets = [(0,0), (0.5, 0.5)]
    
    for off_x, off_y in offsets:
        start_x_px = int(cell_w * off_x)
        start_y_px = int(cell_h * off_y)
        
        for row in range(grid_size):
            for col in range(grid_size):
                y1 = row * cell_h + start_y_px
                y2 = y1 + cell_h
                x1 = col * cell_w + start_x_px
                x2 = x1 + cell_w
                
                # Bounds check
                if y2 > img_height or x2 > img_width: continue
                
                # Extract Region
                region = image_np[y1:y2, x1:x2]
                
                # Filter small/empty regions
                if region.shape[0] < 50 or region.shape[1] < 50: continue
                if is_sky_or_background(region): continue
                
                # Predict
                c_id, c_name, conf = predict_region_cascaded(interpreters, region)
                
                detections.append({
                    'class_id': c_id,
                    'class_name': c_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

    # Simple Non-Max Suppression (Remove duplicates/overlaps)
    # For visualization clarity, we filter by high confidence only
    final_detections = [d for d in detections if d['confidence'] > 0.60]
    
    return final_detections

# ==========================================
# 6. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Crop & Barren Analysis", page_icon="🌾", layout="wide")

st.title("🌾 Field Analysis: Crop & Barren Detection")
st.markdown("""
This tool uses a multi-model approach:
1. **Barren Detector (MobileNetV2):** Scans for empty land first.
2. **Crop Classifier (EfficientNetV2):** Identifies crops only in non-barren areas.
""")

interpreters = load_models()

uploaded_file = st.file_uploader("Upload Field Image", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreters:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
        
    with st.spinner("Analyzing field regions..."):
        results = analyze_image_grid(image, interpreters)
        
    # --- VISUALIZATION ---
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    for det in results:
        x1, y1, x2, y2 = det['bbox']
        label = det['class_name']
        conf = det['confidence']
        
        # Color Coding
        if label == "Barren Land":
            color = "red"  # Distinct color for barren
            display_text = f"BARREN ({conf:.0%})"
        else:
            # Crop Colors
            if "Maize" in label: color = "gold"
            elif "Rice" in label: color = "cyan"
            elif "Soybean" in label: color = "lightgreen"
            elif "Sugarcane" in label: color = "orange"
            else: color = "blue"
            display_text = f"{label} ({conf:.0%})"
            
        # Draw Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw Label
        text_w, text_h = draw.textbbox((0, 0), display_text, font=font)[2:]
        draw.rectangle([x1, y1, x1 + text_w + 10, y1 + text_h + 10], fill=color)
        draw.text((x1+5, y1+5), display_text, fill="black", font=font)
        
    with col2:
        st.image(draw_img, caption="Detected Regions", use_container_width=True)
        
        # Stats
        st.write("### 📊 Detection Summary")
        if not results:
            st.warning("No distinct regions detected (possibly all sky or ambiguous).")
        else:
            df = pd.DataFrame(results)
            counts = df['class_name'].value_counts()
            
            # Custom Metric Display
            c1, c2, c3 = st.columns(3)
            barren_count = counts.get("Barren Land", 0)
            crop_count = len(results) - barren_count
            
            c1.metric("Total Regions", len(results))
            c2.metric("Healthy Crop Regions", crop_count)
            c3.metric("Barren Regions", barren_count, delta_color="inverse")
            
            st.bar_chart(counts)
            
            with st.expander("View Raw Data"):
                st.dataframe(df[['class_name', 'confidence', 'bbox']])
