from pathlib import Path
import json,joblib,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report
from risk_engine import clean_data,add_risk_label,build_pipeline,FEATURES

df=add_risk_label(clean_data(pd.read_csv("data/dcs_student_data.csv")))
Xtr,Xte,ytr,yte=train_test_split(df[FEATURES],df["Risk_Label"],test_size=.2,stratify=df["Risk_Label"],random_state=42)
model=build_pipeline(); model.fit(Xtr,ytr); pred=model.predict(Xte)
metrics={"accuracy":accuracy_score(yte,pred),"balanced_accuracy":balanced_accuracy_score(yte,pred),"classification_report":classification_report(yte,pred,output_dict=True),"rows_after_cleaning":len(df),"risk_distribution":df["Risk_Label"].value_counts().to_dict()}
Path("models").mkdir(exist_ok=True); joblib.dump(model,"models/risk_model.joblib"); Path("models/metrics.json").write_text(json.dumps(metrics,indent=2))
print(json.dumps({k:metrics[k] for k in ["accuracy","balanced_accuracy","rows_after_cleaning","risk_distribution"]},indent=2))
