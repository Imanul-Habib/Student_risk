import argparse,json,joblib
from risk_engine import predict
p=argparse.ArgumentParser(); p.add_argument('--model',default='models/risk_model.joblib'); p.add_argument('--json',required=True)
a=p.parse_args(); print(json.dumps(predict(joblib.load(a.model),json.loads(a.json)),indent=2))