import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import os
import numpy as np
import pandas as pd
from torchvision.models import convnext_tiny

# Constants
MODEL_URL = "https://github.com/mdShihabShorkarSuvo/skin-cancer-classifier-New/raw/main/model/ConvNeXtV2_Tiny_Merged_Pytorch.pth"
MODEL_PATH = "model/ConvNeXtV2_Tiny_Merged_Pytorch.pth"
CLASS_NAMES = {
    0: "Basal Cell Carcinoma (BCC)",
    1: "Benign Keratosis-like Lesions (BKL)",
    2: "Dermatofibroma (DF)",
    3: "Melanoma (MEL)",
    4: "Melanocytic Nevi (NV)",
    5: "Others"
}

# Download model if not found
def download_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
        st.success("✅ Model downloaded!")

# Load model
@st.cache_resource
def load_model():
    download_model()
    model = convnext_tiny(pretrained=False, num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# UI Setup
st.set_page_config(page_title="🧬 Skin Cancer Classifier", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧬 Skin Cancer Classifier</h1>
    <p style='text-align: center;'>Upload a skin lesion image and classify using ConvNeXtV2 Tiny (PyTorch)</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("1. Upload a **clear skin image** (JPG/PNG).\n2. Model will classify the lesion.\n3. Check predicted class and confidence.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    model = load_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    st.success(f"🎯 Predicted: **{CLASS_NAMES[predicted_class]}** ({confidence * 100:.2f}%)")

    # Chart
    st.markdown("### 📊 Probability Chart")
    prob_df = pd.DataFrame(probabilities.numpy(), index=[CLASS_NAMES[i] for i in range(len(CLASS_NAMES))], columns=["Confidence"])
    st.bar_chart(prob_df)

