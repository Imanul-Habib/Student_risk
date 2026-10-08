"""Reproducible dataset exploration for the Student Risk Engine."""
from pathlib import Path
import pandas as pd
from risk_engine import clean_data, add_risk_label

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"dcs_student_data.csv"

def main():
    raw=pd.read_csv(DATA)
    clean=clean_data(raw)
    labeled=add_risk_label(clean)
    print("RAW SHAPE:",raw.shape)
    print("DUPLICATES:",int(raw.duplicated().sum()))
    print("CLEAN SHAPE:",clean.shape)
    print("\nCOLUMNS:")
    print(raw.columns.tolist())
    print("\nDTYPES:")
    print(raw.dtypes)
    print("\nRISK DISTRIBUTION:")
    print(labeled["Risk_Label"].value_counts(dropna=False))
    print("\nCORRELATIONS WITH FINAL SCORE:")
    print(clean.select_dtypes("number").corr(numeric_only=True)["Final_Score"].sort_values(ascending=False))

if __name__=="__main__":main()
