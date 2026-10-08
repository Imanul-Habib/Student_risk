"""Student Risk Engine."""
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC=["Age","Attendance (%)","Midterm_Score","Final_Score","Assignments_Avg","Quizzes_Avg","Participation_Score","Projects_Score","math_score","reading_score","writing_score","science_score"]
CATEGORICAL=["Gender","Department","test_preparation_course"]
FEATURES=NUMERIC+CATEGORICAL

def clean_data(df):
    df=df.copy().drop_duplicates()
    df.columns=[c.strip() for c in df.columns]
    for c in ["Gender","Department","test_preparation_course"]:
        df[c]=df[c].astype("string").str.strip()
    df["Gender"]=df["Gender"].str.title()
    df["Department"]=df["Department"].replace({"CS":"Computer Science","Computer Science":"Computer Science","Engineering":"Engineering","Math":"Mathematics","Mathematics":"Mathematics","Business":"Business"})
    df["math_score"]=pd.to_numeric(df["math_score"].astype(str).str.strip(),errors="coerce")
    numeric=["Age","Attendance (%)","Midterm_Score","Final_Score","Assignments_Avg","Quizzes_Avg","Participation_Score","Projects_Score","math_score","reading_score","writing_score","science_score"]
    for c in numeric: df[c]=pd.to_numeric(df[c],errors="coerce")
    df.loc[(df["Age"]<15)|(df["Age"]>100),"Age"]=np.nan
    for c in numeric[1:]: df.loc[(df[c]<0)|(df[c]>100),c]=np.nan
    return df

def add_risk_label(df):
    a,f=df["Attendance (%)"],df["Final_Score"]
    df=df.copy()
    df["Risk_Label"]=np.select([(a<75)|(f<55),((a>=75)&(a<80))|((f>=55)&(f<70))],["High","Medium"],default="Low")
    return df

def build_pipeline():
    prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),NUMERIC),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL)])
    model=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1,min_samples_leaf=2)
    return Pipeline([("preprocessor",prep),("model",model)])

def hard_rule(attendance,final_score):
    if attendance is None or final_score is None or pd.isna(attendance) or pd.isna(final_score): return None
    if attendance<75 or final_score<55: return "High"
    if attendance<80 or final_score<70: return "Medium"
    return "Low"

def prepare_input(record):
    return pd.DataFrame([{c:record.get(c,np.nan) for c in FEATURES}])

def explain(record):
    reasons=[]; a=record.get("Attendance (%)"); f=record.get("Final_Score")
    if a is not None and not pd.isna(a):
        if a<75: reasons.append(f"Attendance is {a:.1f}%, below the 75% high-risk threshold.")
        elif a<80: reasons.append(f"Attendance is {a:.1f}%, in the 75–80% medium-risk band.")
    if f is not None and not pd.isna(f):
        if f<55: reasons.append(f"Final score is {f:.1f}, below the 55 high-risk threshold.")
        elif f<70: reasons.append(f"Final score is {f:.1f}, in the 55–70 medium-risk band.")
    if not reasons: reasons.append("Attendance and Final Score were not both supplied; the model used the available fields with imputation.")
    return reasons

def predict(model,record):
    x=prepare_input(record); label=model.predict(x)[0]; p=model.predict_proba(x)[0]
    return {"risk":label,"probabilities":{c:float(v) for c,v in zip(model.classes_,p)},"rule_risk":hard_rule(record.get("Attendance (%)"),record.get("Final_Score")),"explanation":explain(record)}
