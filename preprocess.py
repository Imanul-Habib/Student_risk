from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"dcs_student_data.csv"
TARGET="Risk_Label"
NUMERIC=["Age","Attendance (%)","Midterm_Score","Final_Score","Assignments_Avg","Quizzes_Avg","Participation_Score","Projects_Score","math_score","reading_score","writing_score","science_score"]
CATEGORICAL=["Gender","Department","test_preparation_course"]
FEATURES=NUMERIC+CATEGORICAL
IDS=["Student_ID","First_Name","Last_Name","Email"]

def load_data(path=DATA):
    return pd.read_csv(path)

def clean(df):
    df=df.drop_duplicates().copy()
    for c in ["Gender","Department","test_preparation_course"]:
        if c in df: df[c]=df[c].astype("string").str.strip()
    if "Department" in df:
        df["Department"]=df["Department"].str.title().replace({"Cs":"Computer Science","Math":"Mathematics"})
    for c in NUMERIC:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    for c in ["Attendance (%)"]+[x for x in NUMERIC if x!="Age"]:
        df.loc[(df[c]<0)|(df[c]>100),c]=np.nan
    df.loc[(df["Age"]<15)|(df["Age"]>80),"Age"]=np.nan
    a,f=df["Attendance (%)"],df["Final_Score"]
    high=(a<75)|(f<55)
    med=((a>=75)&(a<80))|((f>=55)&(f<70))
    df[TARGET]=np.select([high.fillna(False),med.fillna(False)],["High","Medium"],default="Low")
    df.loc[a.isna()|f.isna(),TARGET]=pd.NA
    return df

def get_xy(path=DATA):
    d=clean(load_data(path)).dropna(subset=[TARGET])
    return d[FEATURES],d[TARGET]

def policy(attendance,final_score):
    if attendance is None or final_score is None:return None
    if attendance<75 or final_score<55:return "High"
    if attendance<80 or final_score<70:return "Medium"
    return "Low"
