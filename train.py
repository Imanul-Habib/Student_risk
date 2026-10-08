"""Train and evaluate the Student Risk Engine."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from risk_engine import clean_data, add_risk_label, build_pipeline, FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dcs_student_data.csv"
MODEL = ROOT / "models" / "risk_model.joblib"
METRICS = ROOT / "models" / "metrics.json"

def main():
    raw = pd.read_csv(DATA)
    cleaned = clean_data(raw)
    labeled = add_risk_label(cleaned).dropna(subset=[TARGET]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        labeled[FEATURES], labeled[TARGET], test_size=0.20,
        stratify=labeled[TARGET], random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, pred, labels=["High", "Medium", "Low"]).tolist()

    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.named_steps["model"].feature_importances_, index=names)
    source_importance = {}
    for name, value in importances.items():
        name = name.split("__", 1)[-1]
        source = next((c for c in FEATURES if name == c or name.startswith(c + "_")), name)
        source_importance[source] = source_importance.get(source, 0.0) + float(value)

    metrics = {
        "raw_rows": int(len(raw)),
        "rows_after_duplicate_removal": int(len(cleaned)),
        "rows_used_for_training": int(len(labeled)),
        "duplicate_rows_removed": int(len(raw) - len(cleaned)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "classification_report": report,
        "confusion_matrix_labels": ["High", "Medium", "Low"],
        "confusion_matrix": matrix,
        "risk_distribution": labeled[TARGET].value_counts().to_dict(),
        "feature_importance": dict(sorted(source_importance.items(), key=lambda x: x[1], reverse=True)),
        "risk_policy": {
            "High": "Attendance < 75 OR Final_Score < 55",
            "Medium": "75 <= Attendance < 80 OR 55 <= Final_Score < 70",
            "Low": "Attendance >= 80 AND Final_Score >= 70",
        },
    }

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2))

    print(json.dumps({
        "raw_rows": metrics["raw_rows"],
        "rows_after_duplicate_removal": metrics["rows_after_duplicate_removal"],
        "risk_distribution": metrics["risk_distribution"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }, indent=2))

if __name__ == "__main__":
    main()
