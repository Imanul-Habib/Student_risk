import joblib
import streamlit as st
from pathlib import Path
from risk_engine import predict

st.set_page_config(page_title="Student Risk Engine", page_icon="🎓", layout="wide")
st.title("🎓 Student Academic Risk Engine")
st.caption("University policy-based risk classification with Random Forest support")

model_path = Path("models/risk_model.joblib")
if not model_path.exists():
    st.error("Model not found. Put dcs_student_data.csv in data/ and run: python train.py")
    st.stop()

model = joblib.load(model_path)

def optional_number(label, minimum=0.0, maximum=100.0):
    value = st.number_input(label, min_value=minimum, max_value=maximum, value=None, step=0.5)
    return value

st.info("Enter at least 2 attributes. Attendance and Final Score are the strongest policy-defining inputs; all other fields are optional.")

with st.form("risk_form"):
    st.subheader("Core risk inputs")
    c1, c2 = st.columns(2)
    attendance = c1.number_input("Attendance (%)", 0.0, 100.0, value=None, step=0.5)
    final_score = c2.number_input("Final Score", 0.0, 100.0, value=None, step=0.5)

    st.subheader("Academic information")
    cols = st.columns(3)
    values = {}
    for i, (key, label) in enumerate([
        ("Midterm_Score", "Midterm Score"), ("Assignments_Avg", "Assignments Average"),
        ("Quizzes_Avg", "Quizzes Average"), ("Participation_Score", "Participation Score"),
        ("Projects_Score", "Projects Score"), ("math_score", "Math Score"),
        ("reading_score", "Reading Score"), ("writing_score", "Writing Score"),
        ("science_score", "Science Score"),
    ]):
        values[key] = cols[i % 3].number_input(label, 0.0, 100.0, value=None, step=0.5)

    st.subheader("Student information")
    c1, c2, c3 = st.columns(3)
    values["Age"] = c1.number_input("Age", 15, 80, value=None, step=1)
    values["Gender"] = c2.selectbox("Gender", ["", "Female", "Male"])
    values["Department"] = c3.selectbox("Department", ["", "Computer Science", "Engineering", "Business", "Mathematics"])
    values["test_preparation_course"] = st.selectbox("Test preparation course", ["", "none", "completed"])

    submitted = st.form_submit_button("Assess Risk", type="primary", use_container_width=True)

if submitted:
    values["Attendance (%)"] = attendance
    values["Final_Score"] = final_score
    values = {k: v for k, v in values.items() if v is not None and v != ""}

    if len(values) < 2:
        st.warning("Please enter at least 2 fields.")
        st.stop()

    result = predict(model, values)
    risk = result["risk"]

    a, b, c = st.columns(3)
    a.metric("Risk Level", risk)
    b.metric("Model Confidence", f'{result["confidence"] * 100:.1f}%')
    c.metric("Inputs Used", len(result["inputs_used"]))

    if risk == "High":
        st.error("🔴 HIGH RISK — immediate academic intervention is recommended.")
    elif risk == "Medium":
        st.warning("🟡 MEDIUM RISK — monitor performance and consider targeted support.")
    else:
        st.success("🟢 LOW RISK — no immediate intervention indicated by this policy.")

    if result["rule_risk"]:
        st.write("**Decision source:** transparent hard-risk policy")
    else:
        st.warning("**Limited-input mode:** Attendance and Final Score were not both supplied. The Random Forest is making a proxy prediction from the available fields.")

    st.subheader("Why?")
    for reason in result["explanation"]:
        st.write("• " + reason)

    st.subheader("Model probabilities")
    st.bar_chart(result["probabilities"])
