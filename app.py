import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ============ ULTRA PRO PAGE SETUP ============
st.set_page_config(page_title="DermaScan AI Pro", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# CUSTOM CSS - Is se hi sara look pro lagega
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Header Gradient */
    .header-box {
        background: linear-gradient(135deg, #0A3D62 0%, #0F6CA3 100%);
        padding: 30px 35px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
    }
    .header-box h1 { color: white; margin:0; font-size:42px; font-weight:800; }
    .header-box p { color: #D6EAF8; margin:5px 0 0 0; font-size:16px; }
    
    /* Cards */
    .pro-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.07);
        border: 1px solid #EEF2F7;
        height: 100%;
    }
    
    /* Upload Button */
    .stFileUploader > div { border-radius: 15px; border: 2px dashed #0F6CA3; background: #F8FBFF; }
    
    .risk-high { background: #FDEDEC; color: #CB4335; padding: 8px 15px; border-radius: 20px; font-weight: 600; }
    .risk-low { background: #EAFAF1; color: #27AE60; padding: 8px 15px; border-radius: 20px; font-weight: 600; }
    .risk-med { background: #FEF9E7; color: #B7950B; padding: 8px 15px; border-radius: 20px; font-weight: 600; }
    
    .disclaimer { background: #FFF8E1; border-left: 6px solid #FFC107; padding: 15px; border-radius: 10px; font-size: 13px; margin-top:15px;}
    .footer { text-align:center; color:#95A5A6; font-size:13px; margin-top:40px; padding-top:20px; border-top:1px solid #eee; }
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
    st.error(f"Model Error: {e}")

class_info = {
    'akiec': ("Actinic Keratoses", "Pre-Cancerous", "Medium"),
    'bcc': ("Basal Cell Carcinoma", "Skin Cancer", "High"),
    'bkl': ("Benign Keratosis", "Benign", "Low"),
    'df': ("Dermatofibroma", "Benign", "Low"),
    'mel': ("Melanoma", "Serious", "Very High"),
    'nv': ("Melanocytic Nevi (Mole)", "Benign Mole", "Low"),
    'vasc': ("Vascular Lesion", "Benign", "Low")
}
classes = list(class_info.keys())

# ============ SIDEBAR PRO ============
with st.sidebar:
    st.markdown("## 🔬 **DermaScan AI**")
    st.markdown("<span style='background:#0F6CA3;color:white;padding:4px 10px;border-radius:12px;font-size:12px;'>PRO v3.0</span>", unsafe_allow_html=True)
    st.write("")
    st.markdown("**Model:** Lightweight CNN (PyTorch)")
    st.markdown("**Dataset:** HAM10000 - 10,015 Images")
    st.markdown("**Input:** 128x128 RGB")
    st.divider()
    st.markdown("### 🎯 Accuracy Metrics")
    st.metric("Model Type", "CNN", "Custom Trained")
    st.metric("Classes", "7 Types")
    st.metric("Status", "Active", "Ready")
    st.divider()
    st.caption("Developed by NulHassan\nLahore, Pakistan")

# ============ HEADER ============
st.markdown("""
<div class="header-box">
    <h1>🔬 DermaScan AI Pro</h1>
    <p>AI-Powered Early Screening System for Skin Lesions | Research & Educational Purpose Only | Built with PyTorch</p>
</div>
""", unsafe_allow_html=True)

# ============ MAIN 2 COLUMNS ============
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Lesion Image")
    st.write("High quality, close-up image upload karen. Blurry image se accuracy kam hogi.")
    uploaded_file = st.file_uploader("Upload JPG, PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        st.success(f"✅ Image Loaded: {uploaded_file.name}")
    else:
        st.markdown("""
        <div style='border:2px dashed #AED6F1; border-radius:15px; padding:40px; text-align:center; background:#F8FBFF;'>
            <h3>🖼️ No Image Selected</h3>
            <p style='color:grey;'>Yahan aap ki image ka preview aayega</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None and model_loaded:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### 📊 AI Analysis Report")
        with st.spinner("🔬 AI Model Analyzing - Please wait 3 sec..."):
            transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                top_probs, top_idx = torch.topk(probs, 3)

        pred = classes[top_idx[0]]
        full_name, category, risk = class_info[pred]
        conf = top_probs[0].item()*100

        # Risk Badge
        if risk in ["High", "Very High"]:
            badge = f"<span class='risk-high'>🔴 Risk: {risk}</span>"
        elif risk == "Medium":
            badge = f"<span class='risk-med'>🟡 Risk: {risk}</span>"
        else:
            badge = f"<span class='risk-low'>🟢 Risk: {risk}</span>"

        st.markdown(f"#### Prediction: {full_name} {badge}", unsafe_allow_html=True)
        st.write("")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{conf:.2f}%")
        m2.metric("Category", category)
        m3.metric("Class Code", pred.upper())

        st.write("**Confidence Level**")
        st.progress(int(conf))

        st.divider()
        st.markdown("**Top 3 Predictions (AI Ranking)**")
        for i in range(3):
            cc = classes[top_idx[i]]
            pp = top_probs[i].item()*100
            name = class_info[cc][0]
            st.write(f"**{i+1}. {name}** — {pp:.1f}%")
            st.progress(int(pp))

        # Download
        report = f"DermaScan AI Pro Report\nPrediction: {full_name}\nRisk: {risk}\nConfidence: {conf:.2f}%\nDisclaimer: Screening only, not diagnosis."
        st.write("")
        st.download_button("📄 Download Professional Report (TXT)", data=report, file_name="DermaScan_Pro_Report.txt", use_container_width=True, type="primary")
        
        st.markdown("<div class='disclaimer'>⚠️ <b>Disclaimer:</b> This tool is for educational & early screening only. It is NOT a medical diagnosis. Final confirmation ke liye Dermatologist se zaroor consult karen.</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        **Step 1:** Lesion ki clear photo upload karen  
        **Step 2:** Lightweight CNN model (PyTorch) 128x128 pe analyze karta hai  
        **Step 3:** 3 second me Report + Risk + Confidence mil jata hai
        
        ---
        #### Why DermaScan Pro?
        ✅ **Fast:** Lab ka wait khatam, second me result  
        ✅ **Affordable:** Ghar baithe early screening  
        ✅ **Complete:** 7 Types (HAM10000 Dataset)  
        ✅ **Private:** Image server pe save nahi hoti
        
        ---
        **Built for Clinics & Researchers in Pakistan**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='footer'>© 2026 DermaScan AI Pro v3.0 | Developed by NulHassan | Lahore, Pakistan | For Research Purpose Only</div>", unsafe_allow_html=True)
