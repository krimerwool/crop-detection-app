import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH_MAIN = 'final_model.tflite'
MODEL_PATH_BARREN = 'barren_vs_crop_model.tflite' # Make sure you have this file

# Main Crop Labels
LABELS_CROP = {
    0: 'Maize',
    1: 'Rice',
    2: 'Soybean',
    3: 'Sugarcane'
}

# Image Settings
IMAGE_SIZE = 260   # EfficientNet Input

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    """Loads both Crop and Barren models"""
    interpreters = {}
    try:
        # Load Main Crop Model
        main_interp = tf.lite.Interpreter(model_path=MODEL_PATH_MAIN)
        main_interp.allocate_tensors()
        interpreters['main'] = main_interp
        
        # Load Barren Model
        barren_interp = tf.lite.Interpreter(model_path=MODEL_PATH_BARREN)
        barren_interp.allocate_tensors()
        interpreters['barren'] = barren_interp
        
        return interpreters
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ==========================================
# ENSEMBLE LOGIC (Updated with Barren Check)
# ==========================================

def preprocess_image_for_model(image, target_size=224, normalize=True):
    """Preprocess image for TFLite model"""
    # Resize
    img = cv2.resize(image, (target_size, target_size))
    
    # Convert to float32
    img = img.astype(np.float32)
    
    # Normalize if required (Barren model usually needs 0-1, EfficientNetV2 needs 0-255)
    if normalize:
        img = img / 255.0
        
    img = np.expand_dims(img, axis=0)
    return img

def is_sky_or_background(region_image, sky_threshold=0.4):
    """Detect if region is mostly sky"""
    # Convert RGB (from PIL/Streamlit) to HSV
    hsv = cv2.cvtColor(region_image, cv2.COLOR_RGB2HSV)
    top_portion = hsv[:int(region_image.shape[0] * 0.4), :]

    # Define Blue Sky range
    blue_sky_mask = cv2.inRange(top_portion,
                                 np.array([90, 50, 100]),
                                 np.array([130, 255, 255]))
                                 
    # Define Gray/Overcast Sky range
    gray_sky_mask = cv2.inRange(top_portion,
                                 np.array([0, 0, 180]),
                                 np.array([180, 40, 255]))

    sky_mask = cv2.bitwise_or(blue_sky_mask, gray_sky_mask)
    
    if sky_mask.size == 0: return False
    
    sky_ratio = np.count_nonzero(sky_mask) / sky_mask.size
    return sky_ratio > sky_threshold

def predict_region_smart(interpreters, region_image):
    """
    Smart Inference: 
    1. Checks for Barren Land First.
    2. If NOT Barren, checks for Crop Type.
    """
    
    # --- STEP 1: BARREN CHECK ---
    # Barren model typically expects 224x224, normalized 0-1
    input_barren = preprocess_image_for_model(region_image, target_size=224, normalize=False) 
    # Note: Adjust 'normalize' based on how you trained the barren model. 
    # Usually simple CNNs need /255.0. Assuming your Barren model expects raw 0-255 here based on typical TFLite usage, 
    # but if it was trained with 1/255 rescale, change normalize=True.
    # Let's assume normalize=False (0-255) for consistency with previous cells unless specified.
    
    # Actually, standard Keras usually wants 0-1. Let's try 0-1 for Barren just to be safe if standard ResNet/MobileNet.
    input_barren_norm = input_barren / 255.0 

    barren_interp = interpreters['barren']
    b_in = barren_interp.get_input_details()
    b_out = barren_interp.get_output_details()
    
    barren_interp.set_tensor(b_in[0]['index'], input_barren_norm)
    barren_interp.invoke()
    barren_prob = barren_interp.get_tensor(b_out[0]['index'])[0][0] # Assuming binary output [prob]

    # Threshold for Barren
    if barren_prob < 0.5: # Assuming < 0.5 means "Barren" (0=Barren, 1=Crop) - ADJUST BASED ON YOUR MODEL
        # Wait, usually output is probability of CLASS 1.
        # If Class 0 = Barren, Class 1 = Crop:
        # Prob(Crop) < 0.5 implies Barren.
        confidence = 1.0 - barren_prob
        return -1, "Barren Land", confidence, None

    # --- STEP 2: CROP CLASSIFICATION ---
    # If we are here, it's a Crop. Run the Main Model.
    
    # EfficientNetV2 expects 0-255 (include_preprocessing=True)
    input_crop = preprocess_image_for_model(region_image, target_size=IMAGE_SIZE, normalize=False)
    
    main_interp = interpreters['main']
    m_in = main_interp.get_input_details()
    m_out = main_interp.get_output_details()
    
    main_interp.set_tensor(m_in[0]['index'], input_crop)
    main_interp.invoke()
    predictions = main_interp.get_tensor(m_out[0]['index'])[0]
    
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    
    return predicted_idx, LABELS_CROP[predicted_idx], confidence, predictions

def classify_grid(image_np, interpreters, grid_size=3, offset_x=0, offset_y=0, pass_name='grid'):
    """Analyzes image in a grid pattern"""
    img_height, img_width = image_np.shape[:2]
    cell_height = img_height // grid_size
    cell_width = img_width // grid_size

    offset_x_px = int(cell_width * offset_x)
    offset_y_px = int(cell_height * offset_y)

    detections = []

    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate coordinates
            y1 = row * cell_height + offset_y_px
            y2 = y1 + cell_height
            x1 = col * cell_width + offset_x_px
            x2 = x1 + cell_width

            # Boundary checks
            y1 = max(0, min(y1, img_height - 50))
            y2 = max(50, min(y2, img_height))
            x1 = max(0, min(x1, img_width - 50))
            x2 = max(50, min(x2, img_width))

            if y2 <= y1 or x2 <= x1: continue

            region = image_np[y1:y2, x1:x2]
            
            # Skip small fragments
            if region.shape[0] < 50 or region.shape[1] < 50: continue

            # Sky Filter
            if is_sky_or_background(region): continue

            # Predict
            c_id, c_name, conf, probs = predict_region_smart(interpreters, region)

            detections.append({
                'class_id': c_id,
                'class_name': c_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'pass': pass_name
            })

    return detections

def run_ensemble_pipeline(image_pil, interpreters):
    """Driver function for the 3-pass system"""
    # Convert PIL to Numpy (RGB)
    image_np = np.array(image_pil)
    
    # Pass 1: Full Image
    full_id, full_name, full_conf, _ = predict_region_smart(interpreters, image_np)
    full_res = {
        'class_id': full_id, 'class_name': full_name, 
        'confidence': full_conf, 'bbox': [0, 0, image_np.shape[1], image_np.shape[0]],
        'pass': 'full'
    }

    # Pass 2: Aligned Grid
    grid_aligned = classify_grid(image_np, interpreters, grid_size=3, offset_x=0, offset_y=0, pass_name='aligned')

    # Pass 3: Offset Grid
    grid_offset = classify_grid(image_np, interpreters, grid_size=3, offset_x=0.5, offset_y=0.5, pass_name='offset')

    # --- VOTING LOGIC ---
    # We combine all results. 
    # If Barren is dominant in a grid cell, it stays Barren.
    # If Crop is dominant, it stays Crop.
    
    all_detections = grid_aligned + grid_offset
    
    # Simple logic: If we have grid detections, trust them for localization. 
    # If grids are empty (e.g. all sky), fallback to full image.
    
    final_results = []
    
    if not all_detections:
        final_results.append(full_res)
    else:
        # Group by overlapping areas (Simplified Non-Max Suppression)
        # For this demo, we will just return high-confidence grid detections 
        # to show multiple crops if present.
        
        for det in all_detections:
            if det['confidence'] > 0.65: # Confidence Threshold
                final_results.append(det)

    return final_results, full_res

# ==========================================
# UI & VISUALIZATION
# ==========================================

st.set_page_config(page_title="Crop & Barren Detector", page_icon="🌾", layout="wide")
st.title("🌾 Advanced Crop Field Analysis")
st.markdown("Detected: **Maize, Rice, Soybean, Sugarcane** & **Barren Land** (Multi-Region)")

interpreters = load_models()

uploaded_file = st.file_uploader("Upload Field Image", type=["jpg", "png", "jpeg"])

if uploaded_file and interpreters:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Running 3-Pass Ensemble Analysis..."):
        detections, full_context = run_ensemble_pipeline(image, interpreters)

    # Draw Detections
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # Load font (optional, fallback to default)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    unique_crops = set()
    
    for det in detections:
        box = det['bbox']
        label = f"{det['class_name']} ({det['confidence']:.0%})"
        unique_crops.add(det['class_name'])
        
        # Color coding
        if det['class_name'] == 'Barren Land':
            color = "brown"
        elif det['class_name'] == 'Maize':
            color = "gold"
        elif det['class_name'] == 'Sugarcane':
            color = "lightgreen"
        else:
            color = "cyan"
            
        # Draw Box (x1, y1, x2, y2)
        draw.rectangle(box, outline=color, width=4)
        
        # Draw Label Background
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([box[0], box[1], box[0] + text_w + 10, box[1] + text_h + 10], fill=color)
        draw.text((box[0]+5, box[1]+5), label, fill="black", font=font)

    with col2:
        st.image(draw_img, caption="Analyzed Regions", use_container_width=True)
        
        st.write("### 🔍 Analysis Report")
        
        # Overall Scene Context
        st.info(f"**Dominant Scene:** {full_context['class_name']} ({full_context['confidence']:.1f}%)")
        
        st.write("**Detailed Grid Breakdown:**")
        if "Barren Land" in unique_crops:
            st.warning("⚠️ **Barren Patches Detected:** Some parts of the field appear uncultivated.")
        
        crop_counts = pd.DataFrame([d['class_name'] for d in detections], columns=['Type']).value_counts()
        st.bar_chart(crop_counts)
        
        with st.expander("See Raw Detection Data"):
            st.dataframe(pd.DataFrame(detections))
