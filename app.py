import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import requests

# Config
st.set_page_config(page_title="🧬 Skin Cancer Classifier", layout="centered")

# Labels
CLASS_NAMES = {
    0: "Basal Cell Carcinoma (BCC)",
    1: "Benign Keratosis-like Lesions (BKL)",
    2: "Dermatofibroma (DF)",
    3: "Melanoma (MEL)",
    4: "Melanocytic Nevi (NV)",
    5: "Others"
}

MODEL_PATH = "model/DenseNet121_Merged_Pytorch.pth"
MODEL_URL = "https://github.com/mdShihabShorkarSuvo/skin-cancer-classifier/raw/main/model/DenseNet121_Merged_Pytorch.pth"

# Download model if not found
def download_model(path):
    try:
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
            st.success("✅ Model downloaded!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")

# Model definition (must match your training script)
class SkinCancerClassifier(nn.Module):
    def __init__(self):
        super(SkinCancerClassifier, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 6)

    def forward(self, x):
        return self.model(x)

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model(MODEL_PATH)
    model = SkinCancerClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Sidebar
with st.sidebar:
    st.markdown("## 📝 How to Use")
    st.markdown("""
    1. Upload a **skin lesion image** (JPG/PNG).
    2. The model will classify the type.
    3. View prediction and confidence chart.
    """)

# Header
st.markdown("""
    <h1 style='text-align:center;'>🧬 Skin Cancer Classifier</h1>
    <p style='text-align:center;'>Upload a skin lesion image to classify the type of cancer using AI.</p>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

        model = load_model()
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]

        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        predicted_label = CLASS_NAMES[predicted_class]

        st.markdown(f"""
            <div style='background:#f0f8ff;padding:20px;border-radius:10px;'>
            <h3>🔍 Prediction Result</h3>
            <p><b>Class:</b> {predicted_class} - {predicted_label}</p>
            <p><b>Confidence:</b> {confidence*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # Probability Chart
        df = pd.DataFrame(probs, index=CLASS_NAMES.values(), columns=["Confidence"])
        st.bar_chart(df)

    except Exception as e:
        st.error(f"🚫 Error: {e}")
else:
    st.info("📷 Please upload an image.")

# Footer
st.markdown("""
    <hr>
    <div style='text-align:center'>
        Developed by <b>Md. Shihab Shorkar</b> | Model: <b>DenseNet121</b>
    </div>
""", unsafe_allow_html=True)
