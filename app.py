import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Title and description
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.title("🔬 Skin Cancer Classifier")
st.markdown("Upload a skin lesion image to predict its class.")

# Constants
MODEL_PATH = "model/DenseNet121_Merged_Pytorch.pth"
CLASS_NAMES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
               'Dermatofibroma', 'Melanocytic nevi', 'Melanoma']

# Define model class using torchvision DenseNet121
class SkinCancerClassifier(nn.Module):
    def __init__(self):
        super(SkinCancerClassifier, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, len(CLASS_NAMES))

    def forward(self, x):
        return self.model(x)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    model = SkinCancerClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = CLASS_NAMES[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

    # Display results
    st.success(f"🧠 **Prediction:** {prediction}")
    st.info(f"📊 **Confidence:** {confidence:.2%}")
