
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)
# 🎨 Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)
# Load model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

# Rebuild model (same as training)
base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(15, activation='softmax')
])

# Load weights
model.load_weights("model.weights.h5")

# Class names (IMPORTANT: same order as training)
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]
disease_details = {
    "Pepper__bell___Bacterial_spot": {
        "cause": "🦠 Caused by bacteria in warm, wet conditions.",
        "treatment": "💊 Use copper-based sprays and remove infected leaves.",
        "precaution": "🛡️ Avoid overhead watering and use disease-free seeds.",
        "severity": "High"
    },
    "Pepper__bell___healthy": {
        "cause": "🌿 No disease present.",
        "treatment": "✅ No treatment needed.",
        "precaution": "💧 Maintain proper watering and sunlight.",
        "severity": "Low"
    },
    "Potato___Early_blight": {
        "cause": "🦠 Fungal infection due to humidity.",
        "treatment": "💊 Apply fungicide and remove affected leaves.",
        "precaution": "🛡️ Practice crop rotation.",
        "severity": "Medium"
    },
    "Potato___Late_blight": {
        "cause": "🦠 Severe fungal disease in wet conditions.",
        "treatment": "💊 Immediate fungicide application required.",
        "precaution": "🛡️ Remove infected plants quickly.",
        "severity": "High"
    },
    "Potato___healthy": {
        "cause": "🌿 Healthy plant.",
        "treatment": "✅ No treatment needed.",
        "precaution": "💧 Regular monitoring.",
        "severity": "Low"
    },
    "Tomato_Bacterial_spot": {
        "cause": "🦠 Bacterial infection.",
        "treatment": "💊 Copper sprays recommended.",
        "precaution": "🛡️ Avoid leaf wetness.",
        "severity": "Medium"
    },
    "Tomato_Early_blight": {
        "cause": "🦠 Fungal infection.",
        "treatment": "💊 Apply fungicide.",
        "precaution": "🛡️ Remove infected leaves.",
        "severity": "Medium"
    },
    "Tomato_Late_blight": {
        "cause": "🦠 Severe fungal infection.",
        "treatment": "💊 Remove plant + fungicide.",
        "precaution": "🛡️ Avoid moisture buildup.",
        "severity": "High"
    },
    "Tomato_Leaf_Mold": {
        "cause": "🍃 High humidity fungus.",
        "treatment": "💊 Improve ventilation.",
        "precaution": "🛡️ Reduce humidity.",
        "severity": "Medium"
    },
    "Tomato_Septoria_leaf_spot": {
        "cause": "🦠 Fungal spots on leaves.",
        "treatment": "💊 Remove infected leaves.",
        "precaution": "🛡️ Avoid overhead watering.",
        "severity": "Medium"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "cause": "🐛 Pest infestation.",
        "treatment": "💊 Use neem oil or insecticide.",
        "precaution": "🛡️ Regular inspection.",
        "severity": "Medium"
    },
    "Tomato_Target_Spot": {
        "cause": "🦠 Fungal disease.",
        "treatment": "💊 Fungicide application.",
        "precaution": "🛡️ Avoid wet leaves.",
        "severity": "Medium"
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "cause": "🦠 Virus spread by whiteflies.",
        "treatment": "❌ No cure, remove plant.",
        "precaution": "🛡️ Control insects.",
        "severity": "High"
    },
    "Tomato_Tomato_mosaic_virus": {
        "cause": "🦠 Viral infection.",
        "treatment": "❌ Remove infected plants.",
        "precaution": "🛡️ Clean tools.",
        "severity": "High"
    },
    "Tomato_healthy": {
        "cause": "🌿 Healthy plant.",
        "treatment": "✅ No treatment needed.",
        "precaution": "💧 Maintain care.",
        "severity": "Low"
    }
}
# Title
st.title("🌿 AI Plant Disease Detection App")

st.markdown("### 🍃 Upload a leaf image and let AI detect the disease")

st.divider()
# 🌱 Sidebar
st.sidebar.title("🌱 About")
st.sidebar.info(
    "This app uses Deep Learning (MobileNetV2) to detect plant diseases.\n\n"
    "Upload a leaf image to get prediction instantly."
)


# 🎛️ Mode selection
mode = st.radio("Choose Input Method:", ["Upload Image 📤", "Take Photo 📸"],horizontal=True)

uploaded_file = None

if mode == "Upload Image 📤":
    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

elif mode == "Take Photo 📸":
    uploaded_file = st.camera_input("📸 Capture a leaf image")
# Use whichever is available


if uploaded_file is not None:
    col1, col2 = st.columns([1,1])

    # 🖼️ IMAGE
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # 🤖 PREDICTION
    with col2:
        image_resized = image.resize((128, 128))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🤖 AI is analyzing the leaf... Please wait"):
            prediction = model.predict(img_array, verbose=0)

        # Top predictions
        top3 = np.argsort(prediction[0])[-3:][::-1]

        st.markdown("### 🌿 Prediction Results")

        for i in top3:
            st.markdown(f"✔ **{class_names[i]}** — `{prediction[0][i]*100:.2f}%`")

        # Best result
        result_index = top3[0]
        disease = class_names[result_index]
        confidence = prediction[0][result_index]
        details = disease_details[disease]
        st.markdown("---")
        st.markdown(f"## 🌱 Final Diagnosis")
        st.success(f"{disease}")
        st.progress(float(confidence))
        

        # 🎨 Severity color
        if details["severity"] == "High":
            st.error(f"🔴 High Severity — Immediate action required!")
        elif details["severity"] == "Medium":
            st.warning(f"🟠 Medium Severity — Monitor and treat soon")
        else:
            st.success(f"🟢 Low Severity — Plant is healthy or minor issue")

       
        # 📋 Info sections
        st.markdown("### Cause")
        st.write(details["cause"])

        st.markdown("### Treatment")
        st.write(details["treatment"])

        st.markdown("### Precautions")
        st.write(details["precaution"])
st.markdown("---")
st.caption("🌱 Built with ❤️ using TensorFlow & Streamlit") 
