import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb

# ===============================
# LOAD MODELS AND PREPROCESSOR
# ===============================
st.set_page_config(page_title="AI Hiring Decision Demo")

st.title("Hiring Decision Prediction Demo")
st.write("This app uses a Neural Network and XGBoost to estimate hiring decision outcomes.")

# Load saved artifacts
import joblib
import tensorflow as tf

preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
nn_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "nn_hiring_model.h5"))
xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_hiring_model.pkl"))

# ===============================
# USER INPUT FORM
# ===============================
st.subheader("Enter Candidate Information")

with st.form("candidate_form"):
    age = st.number_input("Age", min_value=18, max_value=70, value=25)
    experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=2)
    prev_companies = st.number_input("Number of Previous Companies", min_value=0, max_value=20, value=1)
    distance = st.number_input("Distance From Company (km)", min_value=0, max_value=200, value=5)
    interview = st.slider("Interview Score", 0, 100, 70)
    skills = st.slider("Skill Assessment Score", 0, 100, 75)
    personality = st.slider("Personality Score", 0, 100, 80)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
    strategy = st.selectbox("Recruitment Strategy", ["LinkedIn", "Referral", "Job Portal", "Campus"])

    submitted = st.form_submit_button("Predict Hiring Decision")

# ===============================
# PREDICTION LOGIC
# ===============================
if submitted:

    # Create a dataframe matching training schema
    sample = pd.DataFrame([{
        "Age": age,
        "ExperienceYears": experience,
        "PreviousCompanies": prev_companies,
        "DistanceFromCompany": distance,
        "InterviewScore": interview,
        "SkillScore": skills,
        "PersonalityScore": personality,
        "Gender": gender,
        "EducationLevel": education,
        "RecruitmentStrategy": strategy
    }])

    # preprocess
    X_processed = preprocessor.transform(sample)

    # NN prediction
    nn_prob = nn_model.predict(X_processed)[0][0]
    nn_pred = 1 if nn_prob > 0.5 else 0

    # XGBoost prediction
    xgb_pred = xgb_model.predict(X_processed)[0]

    st.subheader("Results")

    st.write(f"**Neural Network probability of hiring:** {nn_prob:.3f}")
    st.write(f"Neural Network decision: {'Hire' if nn_pred == 1 else 'Reject'}")

    st.write("---")

    st.write(f"XGBoost decision: {'Hire' if int(xgb_pred) == 1 else 'Reject'}")

    st.success("Prediction completed successfully.")

