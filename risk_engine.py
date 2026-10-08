"""Core preprocessing, policy rules, Random Forest pipeline and inference helpers."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
NUMERIC = ["Age","Attendance (%)","Midterm_Score","Final_Score","Assignments_Avg","Quizzes_Avg","Participation_Score","Projects_Score","math_score","reading_score","writing_score","science_score"]
CATEGORICAL = ["Gender","Department","test_preparation_course"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Risk_Label"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype("string").str.strip()
    if "Gender" in df:
        df["Gender"] = df["Gender"].str.title()
    if "Department" in df:
        df["Department"] = df["Department"].str.title().replace({
            "Cs": "Computer Science", "Math": "Mathematics"
        })
    if "test_preparation_course" in df:
        df["test_preparation_course"] = df["test_preparation_course"].str.lower()

    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Age" in df:
        df.loc[(df["Age"] < 15) | (df["Age"] > 80), "Age"] = np.nan

    for c in [x for x in NUMERIC if x != "Age"]:
        df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan

    return df

def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    a, f = df["Attendance (%)"], df["Final_Score"]
    high = (a < 75) | (f < 55)
    medium = ((a >= 75) & (a < 80)) | ((f >= 55) & (f < 70))
    df[TARGET] = np.select([high.fillna(False), medium.fillna(False)], ["High", "Medium"], default="Low")
    df.loc[a.isna() | f.isna(), TARGET] = pd.NA
    return df

def build_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    model = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    return Pipeline([("preprocessor", prep), ("model", model)])

def hard_rule(attendance, final_score):
    if attendance is None or final_score is None or pd.isna(attendance) or pd.isna(final_score):
        return None
    if attendance < 75 or final_score < 55:
        return "High"
    if attendance < 80 or final_score < 70:
        return "Medium"
    return "Low"

def prepare_input(record):
    return pd.DataFrame([{c: record.get(c, np.nan) for c in FEATURES}], columns=FEATURES)

def explain(record):
    reasons = []
    a, f = record.get("Attendance (%)"), record.get("Final_Score")
    if a is not None and not pd.isna(a):
        reasons.append(
            f"Attendance is {a:.1f}%: high-risk below 75%, medium-risk from 75% to below 80%."
        )
    if f is not None and not pd.isna(f):
        reasons.append(
            f"Final score is {f:.1f}: high-risk below 55, medium-risk from 55 to below 70."
        )
    if not reasons:
        reasons.append("Attendance and Final Score were not supplied; the Random Forest used the available fields with imputation.")
    return reasons

def predict(model, record):
    x = prepare_input(record)
    label = str(model.predict(x)[0])
    probs = model.predict_proba(x)[0]
    rule = hard_rule(record.get("Attendance (%)"), record.get("Final_Score"))
    return {
        "risk": rule or label,
        "model_prediction": label,
        "probabilities": {str(c): float(v) for c, v in zip(model.classes_, probs)},
        "confidence": float(max(probs)),
        "rule_risk": rule,
        "decision_source": "hard_policy" if rule else "random_forest_limited_input",
        "explanation": explain(record),
    }
