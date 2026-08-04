# ==========================================================
# HEART DISEASE PREDICTION SYSTEM
# Developed by: Anita Okechukwu
# ==========================================================

# ==========================
# IMPORT LIBRARIES
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

/* Main App */
.main{
    background-color:#f8f9fa;
}

/* Title */
.title{
    font-size:42px;
    font-weight:bold;
    color:#c62828;
}

/* Subtitle */
.subtitle{
    font-size:20px;
    color:#555555;
}

/* Section Headers */
.section-header{
    font-size:28px;
    color:#c62828;
    font-weight:bold;
    margin-top:20px;
}

/* Cards */
.card{

    background-color:white;

    padding:20px;

    border-radius:12px;

    box-shadow:0px 3px 10px rgba(0,0,0,0.15);

    margin-bottom:15px;
}

/* Prediction Box */

.prediction{

    padding:25px;

    border-radius:12px;

    text-align:center;

    font-size:26px;

    font-weight:bold;
}

/* Footer */

.footer{

    text-align:center;

    color:gray;

    margin-top:40px;

    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD TRAINED MODEL
# ==========================

MODEL_PATH = "models/heart_disease_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_PATH = "models/feature_names.pkl"

try:

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    feature_names = joblib.load(FEATURE_PATH)

except Exception as e:

    st.error("❌ Error loading model files.")

    st.exception(e)

    st.stop()

# ==========================
# FEATURE IMPORTANCE
# ==========================

def get_feature_importance():

    importance = pd.DataFrame({

        "Feature":feature_names,

        "Importance":model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    return importance

# ==========================
# CREATE INPUT DATAFRAME
# ==========================

def create_input_dataframe(data):

    input_df = pd.DataFrame([data])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(

        columns=feature_names,

        fill_value=0

    )

    return input_df

# ==========================
# RISK LEVEL FUNCTION
# ==========================

def risk_level(probability):

    if probability < 0.30:

        return (

            "🟢 Low Risk",

            "Maintain a healthy lifestyle and continue regular medical check-ups."

        )

    elif probability < 0.70:

        return (

            "🟡 Moderate Risk",

            "Consider discussing your cardiovascular risk factors with a healthcare professional."

        )

    else:

        return (

            "🔴 High Risk",

            "Seek medical evaluation from a qualified healthcare professional as soon as possible."

        )

# ==========================
# TITLE
# ==========================

st.markdown(
    '<p class="title">❤️ Heart Disease Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
"""
<p class="subtitle">

Predict the likelihood of heart disease using
Machine Learning and patient clinical information.

</p>
""",
unsafe_allow_html=True
)

st.divider()

# ==========================
# INTRODUCTION
# ==========================

col1, col2 = st.columns([3,1])

with col1:

    st.info(
"""
This application predicts whether a patient is at risk
of heart disease based on demographic, lifestyle,
and clinical information.

The prediction is generated using a trained
Random Forest Machine Learning model.
"""
    )

with col2:

    st.metric(
        "Model",
        "Random Forest"
    )

    st.metric(
        "Accuracy",
        "99.5%"
    )

    st.metric(
        "ROC-AUC",
        "1.000"
    )

st.divider()

# ==========================================================
# SIDEBAR - PATIENT INFORMATION
# ==========================================================

st.sidebar.header("🩺 Patient Information")

st.sidebar.markdown(
"""
Enter the patient's demographic,
clinical and lifestyle information.
"""
)

# -------------------------
# Numerical Inputs
# -------------------------

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=50
)

cholesterol = st.sidebar.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=400,
    value=200
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure (mmHg)",
    min_value=80,
    max_value=220,
    value=120
)

heart_rate = st.sidebar.number_input(
    "Heart Rate (bpm)",
    min_value=40,
    max_value=180,
    value=75
)

exercise_hours = st.sidebar.slider(
    "Exercise Hours per Week",
    0,
    20,
    5
)

stress_level = st.sidebar.slider(
    "Stress Level",
    1,
    10,
    5
)

blood_sugar = st.sidebar.number_input(
    "Blood Sugar (mg/dL)",
    min_value=50,
    max_value=300,
    value=100
)

# -------------------------
# Categorical Inputs
# -------------------------

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

smoking = st.sidebar.selectbox(
    "Smoking Status",
    ["Never", "Former", "Current"]
)

alcohol = st.sidebar.selectbox(
    "Alcohol Intake",
    ["Moderate", "Heavy"]
)

family_history = st.sidebar.selectbox(
    "Family History",
    ["No", "Yes"]
)

diabetes = st.sidebar.selectbox(
    "Diabetes",
    ["No", "Yes"]
)

obesity = st.sidebar.selectbox(
    "Obesity",
    ["No", "Yes"]
)

angina = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    [
        "Asymptomatic",
        "Atypical Angina",
        "Non-anginal Pain",
        "Typical Angina"
    ]
)

st.sidebar.divider()

predict_button = st.sidebar.button(
    "❤️ Predict Heart Disease",
    use_container_width=True
)

# ==========================================================
# DISPLAY PATIENT SUMMARY
# ==========================================================

st.markdown(
'<p class="section-header">Patient Summary</p>',
unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("### Demographics")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("### Clinical")

    st.write(f"**Blood Pressure:** {blood_pressure}")
    st.write(f"**Heart Rate:** {heart_rate}")
    st.write(f"**Blood Sugar:** {blood_sugar}")
    st.write(f"**Cholesterol:** {cholesterol}")

    st.markdown("</div>", unsafe_allow_html=True)

with col3:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("### Lifestyle")

    st.write(f"**Smoking:** {smoking}")
    st.write(f"**Alcohol:** {alcohol}")
    st.write(f"**Exercise Hours:** {exercise_hours}")
    st.write(f"**Stress Level:** {stress_level}")

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ==========================================================
# CREATE INPUT DATA
# ==========================================================

patient_data = {

    "Age": age,

    "Gender": gender,

    "Cholesterol": cholesterol,

    "Blood Pressure": blood_pressure,

    "Heart Rate": heart_rate,

    "Smoking": smoking,

    "Alcohol Intake": alcohol,

    "Exercise Hours": exercise_hours,

    "Family History": family_history,

    "Diabetes": diabetes,

    "Obesity": obesity,

    "Stress Level": stress_level,

    "Blood Sugar": blood_sugar,

    "Exercise Induced Angina": angina,

    "Chest Pain Type": chest_pain

}

# ==========================================================
# PREDICTION SECTION
# ==========================================================

if predict_button:

    # Create input dataframe
    input_df = create_input_dataframe(patient_data)

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Prediction probability
    probability = model.predict_proba(input_df)[0][1]

    # Risk Level
    risk, recommendation = risk_level(probability)

    st.markdown(
        '<p class="section-header">Prediction Results</p>',
        unsafe_allow_html=True
    )

    # ==========================
    # METRICS
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            "Heart Disease" if prediction == 1 else "No Heart Disease"
        )

    with col2:

        st.metric(
            "Probability",
            f"{probability*100:.2f}%"
        )

    with col3:

        st.metric(
            "Risk Level",
            risk
        )

    st.divider()

    # ==========================
    # PROGRESS BAR
    # ==========================

    st.subheader("Prediction Probability")

    st.progress(float(probability))

    st.write(f"Risk Probability: **{probability*100:.2f}%**")

    st.divider()

    # ==========================
    # RISK MESSAGE
    # ==========================

    if probability >= 0.70:

        st.error(f"""
### 🔴 High Risk

The model predicts a **high probability** of heart disease.

**Recommendation**

{recommendation}
""")

    elif probability >= 0.30:

        st.warning(f"""
### 🟡 Moderate Risk

The model predicts a **moderate probability** of heart disease.

**Recommendation**

{recommendation}
""")

    else:

        st.success(f"""
### 🟢 Low Risk

The model predicts a **low probability** of heart disease.

**Recommendation**

{recommendation}
""")

    # ==========================
    # DETAILED PATIENT SUMMARY
    # ==========================

    st.divider()

    st.subheader("Patient Information Summary")

    summary = pd.DataFrame({

        "Variable":[
            "Age",
            "Gender",
            "Cholesterol",
            "Blood Pressure",
            "Heart Rate",
            "Smoking",
            "Alcohol Intake",
            "Exercise Hours",
            "Family History",
            "Diabetes",
            "Obesity",
            "Stress Level",
            "Blood Sugar",
            "Exercise Induced Angina",
            "Chest Pain Type"
        ],

        "Value":[
            age,
            gender,
            cholesterol,
            blood_pressure,
            heart_rate,
            smoking,
            alcohol,
            exercise_hours,
            family_history,
            diabetes,
            obesity,
            stress_level,
            blood_sugar,
            angina,
            chest_pain
        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # ==========================
    # DOWNLOAD REPORT
    # ==========================

    report = summary.copy()

    report.loc[len(report)] = ["Prediction",
                               "Heart Disease" if prediction == 1 else "No Heart Disease"]

    report.loc[len(report)] = ["Probability",
                               f"{probability*100:.2f}%"]

    report.loc[len(report)] = ["Risk Level",
                               risk]

    csv = report.to_csv(index=False)

    st.download_button(

        label="📥 Download Prediction Report",

        data=csv,

        file_name="heart_disease_prediction_report.csv",

        mime="text/csv"
    )

    # ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.divider()

st.markdown(
    '<p class="section-header">📊 Feature Importance</p>',
    unsafe_allow_html=True
)

importance = get_feature_importance()

top_features = importance.head(10)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.set_title("Top 10 Most Important Features")

st.pyplot(fig)

with st.expander("View Feature Importance Table"):

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()

st.markdown(
    '<p class="section-header">🤖 Model Information</p>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "Random Forest"
    )

with col2:

    st.metric(
        "Accuracy",
        "99.5%"
    )

with col3:

    st.metric(
        "ROC-AUC",
        "1.000"
    )

st.info("""
The Random Forest classifier was selected because it achieved
excellent predictive performance while maintaining strong
generalization on the evaluation dataset.
""")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.divider()

st.markdown(
    '<p class="section-header">📋 About This Project</p>',
    unsafe_allow_html=True
)

st.markdown("""
This application predicts the likelihood of heart disease using
Machine Learning.

### Features Used

- Age
- Gender
- Cholesterol
- Blood Pressure
- Heart Rate
- Smoking Status
- Alcohol Intake
- Exercise Hours
- Family History
- Diabetes
- Obesity
- Stress Level
- Blood Sugar
- Exercise Induced Angina
- Chest Pain Type

### Technologies

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Joblib
""")

# ==========================================================
# MEDICAL DISCLAIMER
# ==========================================================

st.divider()

st.warning("""
### ⚠️ Medical Disclaimer

This application is intended for educational and demonstration
purposes only.

The prediction generated by this machine learning model should
NOT be considered a medical diagnosis.

Always consult a qualified healthcare professional for
medical advice, diagnosis, or treatment.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class="footer">

❤️ Heart Disease Prediction System

<br><br>

Developed by <b>Anita Okechukwu</b>

<br>

Healthcare Data Analyst | Machine Learning Enthusiast | Registered Midwife

<br><br>

Built with Python • Streamlit • Scikit-Learn

</div>
""",
unsafe_allow_html=True
)