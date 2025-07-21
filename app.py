import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms
from collections import OrderedDict

# Page config
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.title("🔬 Skin Cancer Classifier - ConvNeXtV2 Tiny")
st.markdown("Upload a skin lesion image to classify using ConvNeXtV2 Tiny.")

# Constants
MODEL_PATH = "model/ConvNeXtV2_Tiny_Merged_Pytorch.pth"
CLASS_NAMES = [
    'Basal Cell Carcinoma (BCC)', 
    'Benign Keratosis-like Lesions (BKL)', 
    'Dermatofibroma (DF)', 
    'Melanoma (MEL)', 
    'Melanocytic Nevi (NV)', 
    'Others'
]

# Define model using timm
class SkinCancerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # load ConvNeXtV2 Tiny backbone, no pretrained weights here
        self.model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=len(CLASS_NAMES))

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = SkinCancerClassifier()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    # Fix possible prefix 'model.' in keys:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k[6:]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

model = load_model()

# Image transforms (ConvNeXt expects 224x224 input, normalized as below)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    ),
])

# Upload image
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)  # batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_label = CLASS_NAMES[predicted.item()]

    st.success(f"🧠 Prediction: {predicted_label}")
    st.info(f"📊 Confidence: {confidence.item():.2%}")

else:
    st.info("Please upload a skin lesion image to get started.")

# Footer
st.markdown("""
---
<div style="text-align:center">
    Developed by <b>Md. Shihab Shorkar</b> | Model: <b>ConvNeXtV2 Tiny (PyTorch + timm)</b>
</div>
""", unsafe_allow_html=True)
