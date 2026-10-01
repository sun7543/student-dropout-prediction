import joblib
import streamlit as st
import pandas as pd

pipeline = joblib.load('./dropout_pipeline.joblib')
st.set_page_config(page_title="Student Dropout Prediction", layout="centered")

def predict(age, year_of_study, department, residence_type, attendance_percentage, study_per_day, Previous_GPA, Backlogs, Screen_Time_Hours, Part_Time_Job, Family_Income_Bracket, Financial_Stress_Score, Family_Support_Score, Stress_Level, Anxiety_Score, Motivation_Score, Counseling_Access):
    features = {
        "Age": age,
        "Year_of_Study": year_of_study,
        "Department": department,
        "Residence_Type": residence_type,
        "Attendance_Percent": attendance_percentage,
        "Study_Hours_Per_Day": study_per_day,
        "Previous_GPA": Previous_GPA,
        "Backlogs": Backlogs,
        "Screen_Time_Hours": Screen_Time_Hours,
        "Part_Time_Job": Part_Time_Job,
        "Family_Income_Bracket": Family_Income_Bracket,
        "Financial_Stress_Score": Financial_Stress_Score,
        "Family_Support_Score": Family_Support_Score,
        "Stress_Level": Stress_Level,
        "Anxiety_Score": Anxiety_Score,
        "Motivation_Score": Motivation_Score,
        "Counseling_Access": Counseling_Access,
    }
    features = pd.DataFrame([features])
    results = pipeline.predict(features)[0]
    return str(results)

st.markdown("## Student Dropout Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", min_value=17, max_value=25, value=21, step=1)
with col2:
    year_of_study = st.selectbox("Year of study", [1, 2, 3, 4])
with col3:
    department = st.selectbox("Department", ["Arts", "Business", "Engineering", "Law", "Medicine", "Science"])

col1, col2, col3 = st.columns(3)
with col1:
    residence_type = st.selectbox("Residence type", ["Day Scholar", "Hostel", "PG/Rented"])
with col2:
    attendance_percentage = st.slider("Attendance (%)", min_value=30.0, max_value=100.0, value=80.0, step=0.1)
with col3:
    study_per_day = st.slider("Study hours per day", min_value=0.0, max_value=8.1, value=4.0, step=0.1)

col1, col2, col3 = st.columns(3)
with col1:
    Previous_GPA = st.slider("Previous GPA", min_value=2.42, max_value=10.0, value=7.0, step=0.01)
with col2:
    Backlogs = st.slider("Backlogs", min_value=0, max_value=6, value=0, step=1)
with col3:
    Screen_Time_Hours = st.slider("Screen time (hours)", min_value=0.5, max_value=12.6, value=5.0, step=0.1)

col1, col2, col3 = st.columns(3)
with col1:
    Part_Time_Job = st.selectbox("Part-time job", ["No", "Yes"])
with col2:
    Family_Income_Bracket = st.selectbox(
        "Family income bracket", ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"]
    )
with col3:
    Financial_Stress_Score = st.slider("Financial stress", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

col1, col2, col3 = st.columns(3)
with col1:
    Family_Support_Score = st.slider("Family support", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
with col2:
    Stress_Level = st.slider("Stress level", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
with col3:
    Anxiety_Score = st.slider("Anxiety score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

col1, col2 = st.columns(2)
with col1:
    Motivation_Score = st.slider("Motivation score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
with col2:
    Counseling_Access = st.selectbox("Counseling access", ["No", "Yes"])

if st.button("Predict", type="primary"):
    prediction = predict(
        age,
        year_of_study,
        department,
        residence_type,
        attendance_percentage,
        study_per_day,
        Previous_GPA,
        Backlogs,
        Screen_Time_Hours,
        Part_Time_Job,
        Family_Income_Bracket,
        Financial_Stress_Score,
        Family_Support_Score,
        Stress_Level,
        Anxiety_Score,
        Motivation_Score,
        Counseling_Access,
    )
    st.text_input("Predicted dropout risk", value=prediction, disabled=True)
