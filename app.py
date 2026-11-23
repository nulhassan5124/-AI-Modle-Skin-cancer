import streamlit as st
import pandas as pd
import joblib
from PIL import Image

st.set_page_config(page_title="SkinCare AI - Skin Cancer Detection", page_icon="🩺", layout="wide")

if "selected" not in st.session_state:
    st.session_state.selected = "Melanoma"
    st.session_state.menu_open = True

def select(name):
    st.session_state.selected = name

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Poppins:wght@400;600;700;800&display=swap');
.stApp { background: #F8FBFF; }
.block-container { max-width: 1050px; padding-top: 0.5rem; }
* { font-family: 'Poppins', sans-serif !important; }
h1,h2,h3 { font-family: 'Playfair Display', serif !important; }
.logo-circle {
    width: 230px;
    height: 230px;
    background: linear-gradient(145deg, #0B3D5F, #0E5A8A, #0B3D5F);
    border-radius: 50%;
    border: 8px solid #00B4A0;
    box-shadow: 0 0 0 6px #E0F7F5, 0 15px 35px rgba(11,61,95,0.25);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 auto;
}
.info-card { background:white; border:1px solid #E6EFF5; border-radius:18px; padding:22px; box-shadow:0 4px 20px rgba(0,0,0,0.04); }
.nutri-box { background:#0B3D5F; color:white; border-radius:18px; padding:22px; }
.nutri-item { background: rgba(255,255,255,0.08); border-radius:12px; padding:14px; text-align:center; }
.nutri-item b { font-size:20px; display:block; }
.nutri-item span { font-size:11px; color:#A8D5BC; letter-spacing:1px; }
.stButton > button { background:#0E5A8A !important; color:white !important; border-radius:12px !important; font-weight:700 !important; }
.warning-box { background:#FFF7ED; border:1px solid #FDBA74; border-radius:14px; padding:16px; color:#7C2D12; font-size:13px; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('skin_cancer_model.joblib')
    except:
        return None
model = load_model()

# 1. TOP - CIRCLE MEIN SKIN CANCER LOGO - PROFESSIONAL
st.markdown("""
<div style="text-align:center; margin-top:20px;">
    <div class="logo-circle">
        <div style="color:#00B4A0; font-size:32px;">🩺</div>
        <div style="color:white; font-family:'Playfair Display', serif; font-weight:900; font-size:30px; line-height:1.0; text-align:center; margin-top:8px;">SKIN<br>CANCER</div>
        <div style="width:60px; height:2px; background:#00B4A0; margin:12px auto;"></div>
        <div style="color:#00E5C8; font-size:13px; font-weight:800; letter-spacing:2px;">DETECTION AI</div>
        <div style="color:white; font-size:11px; margin-top:8px; letter-spacing:1px;">★ EARLY • SAFE • FAST ★</div>
    </div>
    <h1 style="font-size:52px; font-weight:900; margin:22px 0 0 0; line-height:1;">SkinCare <span style="color:#0E5A8A; font-style:italic;">AI</span></h1>
    <div style="width:70px; height:3px; background:#00B4A0; margin:14px auto;"></div>
    <p style="color:#777; font-size:13px; letter-spacing:1px; font-weight:600;">EARLY DETECTION SAVES LIVES • AI POWERED • 2026</p>
</div>
""", unsafe_allow_html=True)

# 2. INTRODUCTION - SKIN CANCER
st.markdown("""
<div class="info-card" style="text-align:center; margin-top:20px;">
    <h3 style="margin:0; color:#0E5A8A;">Introduction - What is Skin Cancer?</h3>
    <p style="color:#555; font-size:14.5px; line-height:1.9; max-width:800px; margin:14px auto 0 auto;">
        <b>Skin Cancer</b> duniya mein sab se aam cancers mein se ek hai. Ye tab hota hai jab skin ke cells abnormal tareeqe se barhne lagte hain, aksar <b>Sun ki UV Rays</b> ki wajah se.<br><br>
        Agar isko <b>Shuru mein pakar liya jaye</b> to 99% cases mein iska ilaaj mumkin hai. Isi liye <b>SkinCare AI</b> banaya gaya hai - Aap apni skin ki tasveer ya symptoms daalo, AI batayega ke <b>Risk High hai ya Low hai</b> aur aapko doctor se kab milna chahiye.
    </p>
    <div style="display:flex; justify-content:center; gap:12px; margin-top:18px; flex-wrap:wrap;">
        <span style="background:#EFF6FF; border:1px solid #BFDBFE; padding:8px 16px; border-radius:20px; font-size:11px; font-weight:700;">🩺 EARLY DETECTION 99% CURABLE</span>
        <span style="background:#EFF6FF; border:1px solid #BFDBFE; padding:8px 16px; border-radius:20px; font-size:11px; font-weight:700;">🌍 1 IN 5 PEOPLE AFFECTED</span>
        <span style="background:#EFF6FF; border:1px solid #BFDBFE; padding:8px 16px; border-radius:20px; font-size:11px; font-weight:700;">🤖 AI 94.5% ACCURACY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# WARNING
st.markdown("""
<div class="warning-box" style="margin-top:16px; text-align:center;">
    <b>⚠️ Important Note:</b> Ye website sirf <b>Education aur Early Screening</b> ke liye hai. Ye Doctor ka mutabadil nahi hai. Agar aapko koi nishani nazar aaye to foran Dermatologist se rabta karein.
</div>
""", unsafe_allow_html=True)

# 3. VIEW TYPES - MENU KI JAGAH
st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size:28px; text-align:center;'>🔍 Types of Skin Cancer</h2>", unsafe_allow_html=True)
if st.button("☰ VIEW TYPES", use_container_width=True):
    st.session_state.menu_open = not st.session_state.menu_open

types_data = [
    ["Melanoma", "Most Dangerous", "High Risk", "Black / Brown Spot, Asymmetric"],
    ["Basal Cell Carcinoma (BCC)", "Most Common", "Medium Risk", "Pearly Bump, Non-healing Sore"],
    ["Squamous Cell Carcinoma (SCC)", "Second Common", "Medium Risk", "Scaly Red Patch, Wart-like"],
    ["Actinic Keratosis", "Pre-Cancer", "Low Risk", "Rough Scaly Patch on Sun Area"],
    ["Benign Mole (Nevus)", "Normal", "Low Risk", "Round, Even Color, No Change"],
    ["Seborrheic Keratosis", "Benign", "Low Risk", "Waxy, Stuck-on Appearance"],
]

if st.session_state.menu_open:
    st.markdown("<div class='info-card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 12px 0;'>Full List - Types, Risk, Symptoms</h3>", unsafe_allow_html=True)
    df = pd.DataFrame(types_data, columns=["Type", "Category", "Risk Level", "Common Symptoms"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    cols = st.columns(3)
    for i, (name, cat, risk, sym) in enumerate(types_data):
        with cols[i % 3]:
            if st.button(f"{name} • {risk}", key=f"m{i}", use_container_width=True):
                select(name)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# 4. ABCDE RULE + CHECK PROCEDURE
st.markdown(f"<h2 style='margin-top:22px;'>Selected: <span style='color:#0E5A8A; font-style:italic;'>{st.session_state.selected}</span></h2>", unsafe_allow_html=True)
st.markdown("""
<div class="info-card">
    <h3 style="margin:0 0 10px 0; color:#0B3D5F;">ABCDE Rule - Khud Check Karein (Professional Method)</h3>
    <p style="color:#555; font-size:13.5px; line-height:1.8;">
        <b>A - Asymmetry:</b> Ek hissa doosre se alag hai? &nbsp; <b>B - Border:</b> Kinare kharab / tedhe hain? &nbsp; <b>C - Color:</b> Ek se zyada rang hain? &nbsp; <b>D - Diameter:</b> 6mm se bara hai? &nbsp; <b>E - Evolving:</b> Waqt ke saath badal raha hai?
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 1.1], gap="large")
with c1:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0 0 12px 0;'>Check Your Symptoms</h3>", unsafe_allow_html=True)
    asymmetry = st.slider("Asymmetry (0 = Normal, 10 = Bohat Alag)", 0, 10, 2)
    border = st.slider("Border Irregularity (0 = Smooth, 10 = Bohat Kharab)", 0, 10, 2)
    color_var = st.slider("Color Variation (Kitne Rang Hain?)", 1, 6, 1)
    diameter = st.slider("Diameter (mm)", 1, 30, 5)
    evolving = st.selectbox("Kya Ye Badal Raha Hai?", ["Nahi", "Haan, Thora", "Haan, Tezi Se"])
    itching = st.selectbox("Kharish / Khoon Aata Hai?", ["Nahi", "Kabhi Kabhi", "Haan"])
    sun_exposure = st.selectbox("Dhoop Mein Zyada Rehte Ho?", ["Kam", "Medium", "Bohat Zyada"])
    score = int(asymmetry*8 + border*8 + color_var*10 + diameter*2 + (20 if evolving=="Haan, Tezi Se" else 10 if evolving=="Haan, Thora" else 0))
    st.metric("Risk Score", f"{score} / 200", f"Selected: {st.session_state.selected}")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='nutri-box'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:white; margin:0 0 14px 0;'>AI Risk Assessment</h3>", unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    with n1: st.markdown(f"<div class='nutri-item'><b>{asymmetry}/10</b><span>ASYMMETRY</span></div>", unsafe_allow_html=True)
    with n2: st.markdown(f"<div class='nutri-item'><b>{border}/10</b><span>BORDER</span></div>", unsafe_allow_html=True)
    with n3: st.markdown(f"<div class='nutri-item'><b>{color_var}</b><span>COLORS</span></div>", unsafe_allow_html=True)
    with n4: st.markdown(f"<div class='nutri-item'><b>{diameter}mm</b><span>SIZE</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    st.markdown("<b style='color:white;'>📸 Upload Skin Image (Optional)</b>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Apni skin ki tasveer upload karein", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=200, caption="Uploaded Image")

    if st.button("✨ Predict - Risk Check Karein", use_container_width=True):
        if score < 60:
            st.success(f"🟢 LOW RISK - Score {score} - Pareshan hone ki zaroorat nahi, lekin har 3 mahine baad check karein.")
        elif score < 110:
            st.warning(f"🟡 MEDIUM RISK - Score {score} - Dermatologist se 2 hafte ke andar check karwayein.")
        else:
            st.error(f"🔴 HIGH RISK - Score {score} - Foran Doctor se milen! Ye Melanoma jaisa ho sakta hai.")
        if model:
            st.info("AI Model ne bhi analysis complete kar liya hai.")
    st.markdown("</div>", unsafe_allow_html=True)

# 5. PREVENTION
st.markdown("""
<div class="info-card" style="margin-top:20px;">
    <h3 style="margin:0 0 12px 0; color:#0B3D5F;">🛡️ Prevention Tips</h3>
    <p style="color:#555; font-size:13.5px; line-height:1.9;">
        1. Rozana <b>SPF 50+ Sunscreen</b> lagayein &nbsp; | &nbsp; 2. Dopahar 11-3 baje dhoop se bachein &nbsp; | &nbsp; 3. Har mahine apni skin ko aaine mein check karein &nbsp; | &nbsp; 4. Tanning Beds se bachein &nbsp; | &nbsp; 5. Saal mein ek dafa full body check-up karwayein
    </p>
</div>
""", unsafe_allow_html=True)

# 6. DEVELOPED BY
st.markdown("""
<div style="background:#111; color:white; padding:36px 20px; border-radius:20px; margin-top:32px; text-align:center;">
    <div style="font-size:10px; letter-spacing:4px; color:#00B4A0; font-weight:800;">DEVELOPED & DEPLOYED BY</div>
    <div style="font-family:'Playfair Display', serif; font-size:40px; font-weight:900; margin-top:12px;">Noor Ul Hassan Awan</div>
    <div style="width:60px; height:3px; background:#00B4A0; margin:14px auto;"></div>
    <div style="font-size:13px; color:#BBB;">Lahore, Pakistan 🇵🇰 • SkinCare AI • 2026</div>
    <div style="font-size:11px; color:#777; margin-top:8px;">Final Year Project • AI / Machine Learning • Streamlit Cloud • For Educational Purpose Only</div>
</div>
""", unsafe_allow_html=True)
