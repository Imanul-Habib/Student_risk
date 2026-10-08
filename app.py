import joblib,streamlit as st
from pathlib import Path
from risk_engine import predict

st.set_page_config(page_title="Student Risk Engine",page_icon="🎓",layout="wide")
st.title("🎓 Student Academic Risk Engine")
st.caption("Policy-based risk classification with Random Forest support")
m=Path("models/risk_model.joblib")
if not m.exists():
    st.error("Model not found. Put dcs_student_data.csv in data/ and run: python train.py")
    st.stop()
model=joblib.load(m)

def opt_num(label,min_value=0.,max_value=100.):
    s=st.text_input(label,placeholder="Optional")
    if not s.strip(): return None
    try:
        x=float(s)
        if x<min_value or x>max_value: st.error(f"{label} must be between {min_value:g} and {max_value:g}."); return None
        return x
    except ValueError:
        st.error(f"{label} must be numeric."); return None

st.info("Enter at least 2 attributes. You may provide any subset up to all available fields. Attendance and Final Score give the clearest policy-based assessment.")
with st.form("risk"):
    c1,c2,c3=st.columns(3); v={}
    with c1:
        v["Attendance (%)"]=opt_num("Attendance (%)")
        v["Final_Score"]=opt_num("Final Score")
        v["Midterm_Score"]=opt_num("Midterm Score")
        v["Assignments_Avg"]=opt_num("Assignments Avg")
        v["Quizzes_Avg"]=opt_num("Quizzes Avg")
    with c2:
        v["Participation_Score"]=opt_num("Participation Score")
        v["Projects_Score"]=opt_num("Projects Score")
        v["math_score"]=opt_num("Math Score")
        v["reading_score"]=opt_num("Reading Score")
        v["writing_score"]=opt_num("Writing Score")
    with c3:
        v["science_score"]=opt_num("Science Score")
        v["Age"]=opt_num("Age",15,100)
        v["Gender"]=st.selectbox("Gender",["","Female","Male"])
        v["Department"]=st.selectbox("Department",["","Computer Science","Engineering","Business","Mathematics"])
        v["test_preparation_course"]=st.selectbox("Test preparation",["","none","completed"])
    go=st.form_submit_button("Assess Risk")

if go:
    supplied=sum(x is not None and str(x).strip()!="" for x in v.values())
    if supplied<2:
        st.warning("Please enter at least 2 fields.")
        st.stop()
    r=predict(model,v)
    st.subheader(f"Risk: {r['risk']}")
    a,b=st.columns(2)
    with a:
        st.write("**Model probabilities**"); st.json(r["probabilities"])
    with b:
        if r["rule_risk"]: st.write(f"**Hard-rule result:** {r['rule_risk']}")
        else: st.write("**Hard-rule result:** unavailable until both Attendance and Final Score are supplied")
        st.write("**Why:**")
        for x in r["explanation"]: st.write("• "+x)
    st.caption("The Random Forest learns the project's policy-defined risk label; it is not a measured dropout/failure prediction.")
