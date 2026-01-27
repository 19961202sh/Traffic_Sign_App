import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
from model import BetterCNN

with open("class_id_to_name.json") as f:
    class_id_to_name = json.load(f)

num_classes = len(class_id_to_name)


model = BetterCNN(num_classes)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location="cpu"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


st.title("🚦 Traffic Sign Recognition")
st.write("Upload a traffic sign image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    sign_name = class_id_to_name[str(predicted_class)]
    confidence = probs[0][predicted_class].item() * 100

    st.subheader("Prediction")
    st.write(f"**Traffic Sign:** {sign_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
