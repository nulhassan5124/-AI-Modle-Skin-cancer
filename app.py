import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ============ PAGE SETUP ============
st.set_page_config(page_title="DermaScan AI Pro", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #F2F6FC; }
    .title { font-size: 38px; font-weight: 800; color: #0A3D62; margin-bottom: 0px; }
    .subtitle { font-size: 15px; color: #555; margin-top: 0px; }
    .card { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
    .disclaimer { background: #FFF8E1; border-left: 5px solid #FFB300; padding: 12px; border-radius: 8px; font-size: 13px; }
    .footer { text-align:center; color:grey; font-size:13px; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

# ============ MODEL ============
class LightweightCNN(nn.Module):
    def __init__(self):
        super(LightweightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 7)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    model = LightweightCNN()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model Error: {e}. Check model.pth is in same folder.")

# ============ CLASSES INFO ============
class_info = {
    'akiec': ("Actinic Keratoses", "Pre-Cancerous", "Medium"),
    'bcc': ("Basal Cell Carcinoma", "Skin Cancer", "High"),
    'bkl': ("Benign Keratosis", "Benign - Non Cancerous", "Low"),
    'df': ("Dermatofibroma", "Benign", "Low"),
    'mel': ("Melanoma", "Serious Skin Cancer", "Very High"),
    'nv': ("Melanocytic Nevi (Mole)", "Benign Mole", "Low"),
    'vasc': ("Vascular Lesion", "Benign", "Low")
}
classes = list(class_info.keys())

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## 🔬 **DermaScan AI**")
    st.markdown("**Version:** Pro 2.0")
    st.markdown("**Model:** Lightweight CNN")
    st.markdown("**Dataset:** HAM10000 (10,015 Images)")
    st.markdown("**Framework:** PyTorch")
    st.divider()
    st.markdown("### 📋 Detects 7 Types")
    for k, v in class_info.items():
        st.write(f"• {v[0]}")
    st.divider()
    st.markdown("### 👨‍💻 Developer")
    st.write("**NulHassan**")
    st.write("AI Healthcare Developer")
    st.write("Lahore, Pakistan")
    if st.button("🔄 Reset App"):
        st.rerun()

# ============ HEADER ============
st.markdown('<p class="title">🔬 DermaScan AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Early Screening System for Skin Lesions | For Research & Screening Purpose Only</p>', unsafe_allow_html=True)
st.divider()

# ============ MAIN ============
col1, col2 = st.columns([1, 1.25])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 1. Upload Lesion Image")
    st.write("Clear, close-up photo upload karen. Blurry photo ka result accurate nahi hoga.")
    uploaded_file = st.file_uploader("Choose JPG / PNG / JPEG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image - Preview", use_container_width=True)
        st.caption(f"File: {uploaded_file.name} | Size: {image.size}")
    else:
        st.info("👆 Image upload karen, result right side me aayega.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None and model_loaded:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 2. AI Analysis Report")
        with st.spinner("AI Model Analyzing... Please Wait"):
            transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                top_probs, top_idx = torch.topk(probs, 3)

        pred = classes[top_idx[0]]
        full_name, category, risk = class_info[pred]
        conf = top_probs[0].item()*100

        if risk in ["High", "Very High"]:
            st.error(f"**Prediction: {full_name}**")
        elif risk == "Medium":
            st.warning(f"**Prediction: {full_name}**")
        else:
            st.success(f"**Prediction: {full_name}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{conf:.2f}%")
        c2.metric("Category", category)
        c3.metric("Risk Level", risk)

        st.write("**Confidence Meter**")
        st.progress(int(conf))

        st.write("**Top 3 Predictions**")
        for i in range(3):
            cc = classes[top_idx[i]]
            pp = top_probs[i].item()*100
            st.write(f"{class_info[cc][0]} — {pp:.1f}%")
            st.progress(int(pp))

        report = f"""DermaScan AI Pro - Screening Report

Prediction: {full_name}
Category: {category}
Risk Level: {risk}
Confidence: {conf:.2f}%

Top 3:
1. {class_info[classes[top_idx[0]]][0]} - {top_probs[0].item()*100:.2f}%
2. {class_info[classes[top_idx[1]]][0]} - {top_probs[1].item()*100:.2f}%
3. {class_info[classes[top_idx[2]]][0]} - {top_probs[2].item()*100:.2f}%

Disclaimer: This is AI screening only, not a medical diagnosis. Consult dermatologist.
"""
        st.download_button("📄 Download Full Report", data=report, file_name="DermaScan_Report.txt", use_container_width=True)
        st.markdown(f"<div class='disclaimer'>⚠️ <b>Important Disclaimer:</b> Ye result AI prediction hai, final diagnosis nahi hai. Confirm karne ke liye qualified dermatologist se zaroor rabta karen.</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ℹ️ How It Works")
        st.write("""
        **Step 1:** Skin lesion ki clear photo upload karen  
        **Step 2:** Lightweight CNN (PyTorch) image ko 128x128 me analyze karta hai  
        **Step 3:** 3 second me Top 3 predictions + Risk Level + Report mil jata hai
        """)
        st.divider()
        st.write("**Why DermaScan?**")
        st.write("✅ Fast - No waiting for lab")
        st.write("✅ Low Cost - Early screening at home")
        st.write("✅ 7 Classes - Complete HAM10000 coverage")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="footer">© 2026 DermaScan AI Pro | Developed by NulHassan | Lahore, Pakistan | v2.0</p>', unsafe_allow_html=True)
