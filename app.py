import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ---------------------------------------------------
# 1. Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("save_model/resnet50_cifar100_checkpoint.pth", map_location="cpu")

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes = checkpoint["classes"]
    return model, classes

model, classes = load_model()

# ---------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return classes[predicted.item()]

# ---------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------
st.title("🐯 CIFAR-100 Image Classifier (ResNet-50)")
st.write("Upload an image OR choose one of the 10 CIFAR-10 sample images.")

# ---------------------------------------------------
# Upload Section
# ---------------------------------------------------
st.subheader("📤 Upload Your Own Image")

upload_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload_file:
    img = Image.open(upload_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Uploaded Image"):
        pred = predict(img)
        st.success(f"Predicted Class: **{pred}**")

st.markdown("---")

# ---------------------------------------------------
# Section: Select from CIFAR-10 Samples (10 samples only)
# ---------------------------------------------------
st.subheader("🖼️ Choose From 10 CIFAR-100 Sample Images")

sample_dir = "cifar10_samples"   # <--- your 10 images folder

if not os.path.exists(sample_dir):
    st.error("Sample image folder not found! (expected: cifar100_10_samples/)")
else:
    all_samples = sorted(os.listdir(sample_dir))

    selected = st.selectbox("Select a CIFAR-10 Sample Image:", all_samples)

    if selected:
        img_path = os.path.join(sample_dir, selected)
        img = Image.open(img_path).convert("RGB")
        st.image(img, caption=selected, width=300)

        if st.button("🔍 Predict Sample Image"):
            pred = predict(img)
            st.info(f"Predicted Class: **{pred}**")
