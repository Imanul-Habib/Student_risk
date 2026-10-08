"""Command-line inference for the trained risk engine."""
import argparse
import json
import joblib
from risk_engine import predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/risk_model.joblib")
    parser.add_argument("--json", required=True, help="Student fields as a JSON object")
    args = parser.parse_args()

    model = joblib.load(args.model)
    result = predict(model, json.loads(args.json))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
