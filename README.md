# 🫀 Cardiac Risk Assessment Through Retinal Images

## 📌 Project Overview
This project presents an **AI-powered, non-invasive cardiac risk screening system** that predicts an individual’s **cardiovascular risk** using **retinal fundus images**. Since retinal blood vessels reflect systemic vascular health, deep learning can identify early indicators of cardiac risk without requiring invasive medical tests.

The system uses a **Convolutional Neural Network (CNN)** trained on retinal images and is deployed as an **interactive Streamlit web application**.

---

## ❓ Problem Statement
Conventional cardiac screening methods rely on:
- Blood tests
- ECG and imaging scans
- Hospital visits

These approaches are **invasive, costly, and not scalable** for early population-level screening. There is a need for a **fast, low-cost, and non-invasive screening solution** for early cardiac risk detection.

---

## 💡 Proposed Solution
- Use **retinal fundus images** as a proxy for cardiovascular health
- Apply **deep learning (CNN)** to extract vascular features
- Predict cardiac risk as **High Risk / Low Risk**
- Provide **confidence scores** and **visual explainability using Grad-CAM**

---

## 🧠 Key Features
- 📤 Single retinal image screening
- 📂 Batch screening of multiple images
- 🫀 Cardiac risk prediction (High / Low)
- 📊 Confidence score for prediction reliability
- 🔍 **Grad-CAM visualization** highlighting influential retinal regions
- 📊 Risk analytics with recommended actions
- ⏱ Suggested medical follow-up timelines
- 📄 Downloadable cardiac risk report
- 🌐 Web-based interface using Streamlit

---

## 📊 Dataset Used
- **Gaussian-filtered retinal fundus images**
- Source: **EyePACS / APTOS Diabetic Retinopathy Dataset**
- Gaussian filtering enhances vessel clarity and reduces noise
- Due to the unavailability of direct cardiac labels:
  - No DR / Mild DR → **Low Cardiac Risk**
  - Moderate / Severe / Proliferative DR → **High Cardiac Risk**

This proxy-based approach enables effective cardiac risk stratification.

---

## ⚙️ Project Workflow
1. Retinal image upload
2. Image preprocessing (resize, normalization)
3. CNN-based deep learning inference
4. Risk score generation
5. Risk classification (High / Low)
6. Grad-CAM explainability
7. Risk analytics and recommendations
8. Report generation

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow & Keras** – Deep learning model
- **OpenCV & PIL** – Image preprocessing
- **NumPy & Pandas** – Data handling
- **Grad-CAM** – Model explainability
- **Streamlit** – Web application deployment

---

## 🚀 Deployment
The application is deployed using **Streamlit Cloud** and supports real-time inference through a web interface.

### Folder Structure
cardiac-risk-app/
│
├── app.py
├── train_model.py
├── split_dataset.py
├── requirements.txt
├── gradcam_result.png
├── cardiac_risk_model/
│ ├── assets/
│ ├── variables/
│ └── saved_model.pb


---

## 📈 Results & Outcomes
- Accurate prediction of cardiac risk from retinal images
- Explainable AI results using Grad-CAM
- Demonstrates feasibility of non-invasive cardiac screening
- Suitable as an early screening and clinical decision-support tool

---

## 🔮 Future Scope
- Incorporate real cardiac outcome datasets
- Combine retinal images with clinical parameters (BP, cholesterol, ECG)
- Improve accuracy using advanced CNN architectures
- Deploy as a mobile or hospital-integrated system
- Perform clinical validation and trials

---

## ⚠️ Disclaimer
This system is intended for **educational and research purposes only** and **does not replace professional medical diagnosis**.

---

## 👤 Author
**Hetvi Patel**  
B.Tech – Computer Engineering  
AI / Machine Learning Project

