import joblib,streamlit as st
from pathlib import Path
from risk_engine import predict
st.set_page_config(page_title='Student Risk Engine',page_icon='🎓',layout='wide')
st.title('🎓 Student Academic Risk Engine')
st.caption('Policy-based risk classification with Random Forest support')
m=Path('models/risk_model.joblib')
if not m.exists(): st.error('Run python train.py first.'); st.stop()
model=joblib.load(m)
st.info('Enter at least 2 fields. Attendance and Final Score are the strongest policy signals; additional fields are optional.')
with st.form('risk'):
 c1,c2,c3=st.columns(3); v={}
 with c1:
  v['Attendance (%)']=st.number_input('Attendance (%)',0.,100.,75.); v['Final_Score']=st.number_input('Final Score',0.,100.,70.)
  v['Midterm_Score']=st.number_input('Midterm Score',0.,100.,0.); v['Assignments_Avg']=st.number_input('Assignments Avg',0.,100.,0.); v['Quizzes_Avg']=st.number_input('Quizzes Avg',0.,100.,0.)
 with c2:
  v['Participation_Score']=st.number_input('Participation Score',0.,100.,0.); v['Projects_Score']=st.number_input('Projects Score',0.,100.,0.)
  v['math_score']=st.number_input('Math Score',0.,100.,0.); v['reading_score']=st.number_input('Reading Score',0.,100.,0.); v['writing_score']=st.number_input('Writing Score',0.,100.,0.)
 with c3:
  v['science_score']=st.number_input('Science Score',0.,100.,0.); v['Age']=st.number_input('Age',15,100,20)
  v['Gender']=st.selectbox('Gender',['Female','Male']); v['Department']=st.selectbox('Department',['Computer Science','Engineering','Business','Mathematics']); v['test_preparation_course']=st.selectbox('Test preparation',['none','completed'])
 go=st.form_submit_button('Assess Risk')
if go:
 r=predict(model,v); st.subheader(f"Risk: {r['risk']}")
 a,b=st.columns(2)
 with a: st.write('Model probabilities'); st.json(r['probabilities'])
 with b:
  if r['rule_risk']: st.write(f"Hard-rule result: **{r['rule_risk']}**")
  st.write('Why:'); [st.write('• '+x) for x in r['explanation']]
 st.caption("The model learns the project's policy-defined risk label; it is not a measured dropout/failure prediction.")