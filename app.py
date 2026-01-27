import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import BetterCNN

import json

with open("class_order.json") as f:
    class_order = json.load(f)


# ----------------------------
# Load labels
# ----------------------------
labels_df = pd.read_csv("labels.csv")
class_names = dict(zip(labels_df.ClassId, labels_df.Name))
num_classes = len(class_names)

# ----------------------------
# Load model
# ----------------------------
model = BetterCNN(num_classes)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location="cpu"))
model.eval()

# ----------------------------
# Image transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚦 Traffic Sign Recognition")
st.write("Upload a traffic sign image to get the prediction.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    # ✅ CORRECT LABEL MAPPING
    predicted_index = predicted_class
    true_class_id = int(class_order[predicted_index])
    sign_name = class_names[true_class_id]

    confidence = probs[0][predicted_index].item() * 100

    st.subheader("Prediction")
    st.write(f"**Traffic Sign:** {sign_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
