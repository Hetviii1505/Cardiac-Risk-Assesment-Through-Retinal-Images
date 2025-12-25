import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Cardiac Risk Assessment",
    layout="wide"
)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("cardiac_risk_model")

# ================= SESSION STATE =================
if "score" not in st.session_state:
    st.session_state.score = None
if "label" not in st.session_state:
    st.session_state.label = None

# ================= FUNCTIONS =================
def preprocess(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict(image):
    score = model.predict(preprocess(image), verbose=0)[0][0]
    label = "High Risk" if score >= 0.5 else "Low Risk"
    return score, label

def generate_report(score, label):
    t = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    return f"""
CARDIAC RISK ASSESSMENT REPORT
=============================

Date & Time: {t}

Risk Category : {label}
Risk Score    : {score:.2f}

Generated using AI-based retinal image analysis.
"""

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align:center; color:#b30000;'>🫀 Cardiac Risk Assessment Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-based non-invasive screening using retinal images</p>",
    unsafe_allow_html=True
)

st.divider()

# ================= LAYOUT =================
left, middle, right = st.columns([1.2, 1.2, 1])

# ================= SINGLE IMAGE =================
with left:
    st.subheader("🔍 Single Image Screening")

    file = st.file_uploader(
        "Upload retinal fundus image",
        type=["jpg", "png", "jpeg"]
    )

    if file:
        image = Image.open(file)
        st.image(image, width=350)

        score, label = predict(image)
        st.session_state.score = score
        st.session_state.label = label

        st.progress(float(score))

        if label == "High Risk":
            st.error("🔴 HIGH CARDIAC RISK")
        else:
            st.success("🟢 LOW CARDIAC RISK")

        st.metric("Risk Probability", f"{score:.2f}")

        report = generate_report(score, label)
        st.download_button(
            "📄 Download Report",
            report,
            file_name="cardiac_risk_report.txt"
        )

    # Load and display Grad-CAM image
    gradcam_img = Image.open("gradcam_result.png")

    st.image(
        gradcam_img,
        caption="Red/Yellow regions → higher influence on prediction",
        use_container_width=True
    )

# ================= BATCH =================
with middle:
    st.subheader("📂 Batch Screening")

    files = st.file_uploader(
        "Upload multiple retinal images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if files:
        for f in files:
            img = Image.open(f)
            s, l = predict(img)
            st.write(f"**{f.name}** → {l} ({s:.2f})")

# ================= RISK ANALYTICS =================
with right:
    st.subheader("📊 Risk Analytics")

    if st.session_state.score is not None:
        score = st.session_state.score
        label = st.session_state.label

        # Risk Level
        if score >= 0.75:
            st.error("🟥 VERY HIGH RISK")
        elif score >= 0.5:
            st.warning("🟧 MODERATE RISK")
        else:
            st.success("🟩 LOW RISK")

        # Probability Bar (Streamlit-native)
        st.markdown("**Risk Probability**")
        st.progress(float(score))

        # Recommendations
        st.markdown("### 🩺 Recommended Actions")
        if score >= 0.75:
            st.write("• Immediate cardiologist consultation")
            st.write("• ECG / Echocardiography")
            st.write("• Aggressive lifestyle intervention")
        elif score >= 0.5:
            st.write("• Periodic monitoring")
            st.write("• BP & lipid profile check")
            st.write("• Lifestyle modification")
        else:
            st.write("• Routine follow-up")
            st.write("• Maintain healthy lifestyle")

        # Follow-up Timeline
        st.markdown("### ⏱ Suggested Follow-up")
        if score >= 0.75:
            st.write("Within **2–4 weeks**")
        elif score >= 0.5:
            st.write("Within **3–6 months**")
        else:
            st.write("Within **12 months**")

    else:
        st.info("Run a single-image prediction to view analytics")

st.divider()
# ================= FOOTER =================
st.caption(" • Streamlit • Retinal Image Analysis •")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 Cardiac Risk Assessment. All rights reserved.</p>",
    unsafe_allow_html=True
)



