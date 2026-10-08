import pandas as pd
from risk_engine import clean_data,add_risk_label

df=pd.read_csv("data/dcs_student_data.csv")
print("Raw shape:",df.shape)
print("\nMissing values:\n",df.isna().sum()[df.isna().sum()>0])
clean=add_risk_label(clean_data(df))
print("\nShape after exact-duplicate removal:",clean.shape)
print("\nRisk distribution:\n",clean["Risk_Label"].value_counts())
print("\nNormalized departments:\n",clean["Department"].value_counts())
print("\nNormalized genders:\n",clean["Gender"].value_counts())
