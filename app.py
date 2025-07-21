import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict

# Page config
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.title("🔬 Skin Cancer Classifier")
st.markdown("Upload a skin lesion image to predict its class.")

# Constants
MODEL_PATH = "model/DenseNet121_Merged_Pytorch.pth"
CLASS_NAMES = ['Basal Cell Carcinoma (BCC)', 
               'Benign Keratosis-like Lesions (BKL)', 
               'Dermatofibroma (DF)', 
               'Melanoma (MEL)', 
               'Melanocytic Nevi (NV)', 
               'Others']

# Define model class using torchvision DenseNet121
class SkinCancerClassifier(nn.Module):
    def __init__(self):
        super(SkinCancerClassifier, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, len(CLASS_NAMES))

    def forward(self, x):
        return self.model(x)

# Load model with prefix fix
@st.cache_resource
def load_model():
    model = SkinCancerClassifier()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    
    # Remove 'model.' prefix from keys if exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[6:]  # remove 'model.' prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
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

    image_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_label = CLASS_NAMES[predicted.item()]

    st.success(f"🧠 Prediction: {predicted_label}")
    st.info(f"📊 Confidence: {confidence.item():.2%}")

else:
    st.info("📷 Please upload a skin lesion image to get started.")
    
# Footer
st.markdown("""
    <hr>
    <div style='text-align:center'>
        Developed by <b>Md. Shihab Shorkar</b> | Model: <b>DenseNet121 (PyTorch)</b>
    </div>
""", unsafe_allow_html=True)
