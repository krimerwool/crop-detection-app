import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================
MODEL_CROP = "final_model.tflite"
MODEL_BARREN = "barren_vs_crop_model_v2.tflite"

CROP_SIZE = 260
BARREN_SIZE = 224

GRID_SIZE = 3

CONF_THRESH = 0.65
VOTE_THRESH = 3

LABELS = {
    0: "Maize",
    1: "Rice",
    2: "Soybean",
    3: "Sugarcane"
}

COLORS = {
    "Rice": "cyan",
    "Sugarcane": "orange",
    "Maize": "yellow",
    "Soybean": "lightgreen",
    "Barren": "red"
}

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    crop = tf.lite.Interpreter(model_path=MODEL_CROP)
    crop.allocate_tensors()

    barren = tf.lite.Interpreter(model_path=MODEL_BARREN)
    barren.allocate_tensors()

    return crop, barren

crop_interp, barren_interp = load_models()

crop_in = crop_interp.get_input_details()
crop_out = crop_interp.get_output_details()
barren_in = barren_interp.get_input_details()
barren_out = barren_interp.get_output_details()

# =====================================================
# PREPROCESSING
# =====================================================
def preprocess_barren(img):
    img = cv2.resize(img, (BARREN_SIZE, BARREN_SIZE))
    img = img.astype(np.float32)
    return np.expand_dims(img, 0)

def preprocess_crop(img):
    img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
    img = img.astype(np.float32)
    return np.expand_dims(img, 0)

# =====================================================
# SKY DETECTION
# =====================================================
def is_sky_or_background(region, threshold=0.4):
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    top = hsv[: int(region.shape[0] * 0.4), :]

    blue = cv2.inRange(top, (90, 50, 100), (130, 255, 255))
    gray = cv2.inRange(top, (0, 0, 180), (180, 50, 255))

    mask = cv2.bitwise_or(blue, gray)
    if mask.size == 0:
        return False

    return (np.count_nonzero(mask) / mask.size) > threshold

# =====================================================
# FULL IMAGE PASS
# =====================================================
def classify_full_image(image):
    """
    Pass 1:
    - Run barren vs crop model on entire image
    - If non-barren, run crop classifier
    """

    # ---- BARREN MODEL ----
    barren_interp.set_tensor(
        barren_in[0]["index"],
        preprocess_barren(image)
    )
    barren_interp.invoke()

    prob_crop = barren_interp.get_tensor(
        barren_out[0]["index"]
    )[0][0]

    barren_result = {
        "is_crop": prob_crop > 0.5,
        "confidence": float(prob_crop if prob_crop > 0.5 else 1 - prob_crop)
    }

    # # If barren → stop here
    # if prob_crop <= 0.5:
    #     return barren_result, None

    # ---- CROP MODEL ----
    crop_interp.set_tensor(
        crop_in[0]["index"],
        preprocess_crop(image)
    )
    crop_interp.invoke()

    probs = crop_interp.get_tensor(
        crop_out[0]["index"]
    )[0]

    idx = int(np.argmax(probs))

    full_crop = {
        "crop": LABELS[idx],
        "confidence": float(probs[idx]),
        "all_probs": probs.tolist()
    }

    return barren_result, full_crop

# =====================================================
# REGION CLASSIFICATION
# =====================================================
def classify_region(region):
    barren_interp.set_tensor(barren_in[0]["index"], preprocess_barren(region))
    barren_interp.invoke()

    prob_crop = barren_interp.get_tensor(barren_out[0]["index"])[0][0]

    if prob_crop <= 0.5:
        return None

    crop_interp.set_tensor(crop_in[0]["index"], preprocess_crop(region))
    crop_interp.invoke()
    probs = crop_interp.get_tensor(crop_out[0]["index"])[0]

    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

# =====================================================
# GRID PASS
# =====================================================
def run_grid(image, ox=0.0, oy=0.0):
    h, w = image.shape[:2]
    ch, cw = h // GRID_SIZE, w // GRID_SIZE

    detections = []

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1 = int(r * ch + oy * ch)
            y2 = y1 + ch
            x1 = int(c * cw + ox * cw)
            x2 = x1 + cw

            if y2 > h or x2 > w:
                continue

            region = image[y1:y2, x1:x2]

            if region.shape[0] < 60 or region.shape[1] < 60:
                continue

            if is_sky_or_background(region):
                continue

            pred = classify_region(region)
            if pred is None:
                continue

            crop, conf = pred

            if conf >= CONF_THRESH:
                detections.append({
                    "crop": crop,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })

    return detections

# =====================================================
# DRAW
# =====================================================
def draw_boxes(image, detections):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORS.get(d["crop"], "white")
        label = f"{d['crop']} {int(d['conf']*100)}%"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 5, y1 + 5), label, fill=color)

    return img

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(layout="wide")
st.title(" Multi-Crop Detection — 3-Pass Ensemble System")

uploaded = st.file_uploader("Upload field image", type=["jpg", "jpeg", "png"])

if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)

    st.image(image_pil, caption="Original Image", width="stretch")

    with st.spinner("Running full ensemble inference..."):
        barren_result, full_crop = classify_full_image(image_np)

        grid_aligned = run_grid(image_np, 0.0, 0.0)
        grid_offset = run_grid(image_np, 0.5, 0.5)

        all_dets = grid_aligned + grid_offset

        votes = defaultdict(list)
        for d in all_dets:
            votes[d["crop"]].append(d)

        print("\n================ ENSEMBLE DEBUG =================")
        final = []

        for crop, items in votes.items():
            avg_conf = np.mean([i["conf"] for i in items])
            vote_count = len(items)

            print(f"{crop}:")
            for i in items:
                print(f"   conf={i['conf']:.3f} bbox={i['bbox']}")
            print(f"   TOTAL VOTES = {vote_count}")
            print(f"   AVG CONF    = {avg_conf:.3f}\n")

            is_full_prior = full_crop and crop == full_crop["crop"]

            vote_req = 2 if is_full_prior else VOTE_THRESH
            conf_req = 0.60 if is_full_prior else CONF_THRESH

            if vote_count >= vote_req and avg_conf >= conf_req or (vote_count >= 1 and avg_conf >= 0.99):
                xs = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in items]
                ys = [(b["bbox"][1] + b["bbox"][3]) / 2 for b in items]

                cx, cy = int(np.mean(xs)), int(np.mean(ys))

                location = (
                    "top" if cy < image_np.shape[0]/3 else
                    "center" if cy < image_np.shape[0]*2/3 else
                    "bottom"
                )
                

                final.append({
                    "Crop": crop,
                    "Votes": vote_count,
                    "Avg Confidence (%)": round(avg_conf * 100, 2),
                    "Location": location,
                    "Source": "full-image-prior" if is_full_prior else "grid-consensus",
                    "Full Crop Result": full_crop["crop"]
                })

# =====================================================
        if full_crop is not None:
            full_crop_name = full_crop["crop"]
            final_crops = [item["Crop"] for item in final]

            if full_crop_name not in final_crops:
                print("\n⚠️ FULL IMAGE CROP NOT PRESENT IN ENSEMBLE")
                print("➡️ Replacing ensemble output with full-image result")

                final = [{
                    "Crop": full_crop_name,
                    "Votes": "FULL",
                    "Avg Confidence (%)": round(full_crop["confidence"] * 100, 2),
                    "Location": "entire_field",
                    "Source": "full-image-override"
                }]               
# =====================================================
# =====================================================
# FALLBACK: FULL IMAGE CROP IF ENSEMBLE FAILS
# =====================================================
        if not final and full_crop is not None:
            print("\n ENSEMBLE EMPTY — FALLING BACK TO FULL IMAGE PREDICTION")

            final.append({
                "Crop": full_crop["crop"],
                "Votes": "FULL",
                "Avg Confidence (%)": round(full_crop["confidence"] * 100, 2),
                "Location": "entire_field",
                "Source": "full-image-fallback"
            })
    # ================= VISUAL =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pass 2 — Aligned Grid")
        st.image(draw_boxes(image_pil, grid_aligned), width="stretch")

    with col2:
        st.subheader("Pass 3 — Offset Grid")
        st.image(draw_boxes(image_pil, grid_offset), width="stretch")

    st.subheader(" Full Image Barren Detection")
    if not barren_result["is_crop"]:
        st.error(f"Barren Land ({barren_result['confidence']*100:.1f}%)")
    else:
        st.success(f"Non-Barren Field ({barren_result['confidence']*100:.1f}%)")

    st.subheader(" Final Ensemble Output")
    if final:
        st.dataframe(pd.DataFrame(final), use_container_width=True)
    else:
        st.warning("No crop satisfied ensemble conditions.")
