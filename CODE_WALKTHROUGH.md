# Code Walkthrough & Technical Interview Notes

This document explains the current implementation file by file and block by block. It describes what the code actually does, why the main choices were made, how the files connect, and what to be ready to explain in an interview.

> **Important project framing:** this implementation predicts a *policy-defined academic risk label*. The label is generated from attendance and final-score thresholds. It is not trained against an observed future outcome such as dropout, graduation, or course failure. The reported high test score mainly shows that the model learned the rules used to create its target; it is not evidence of real-world dropout-prediction accuracy.

---

## 1. System overview

The project has two main paths:

1. **Training path:** CSV → cleaning → risk labels → train/test split → preprocessing + Random Forest → evaluation → saved model and metrics.
2. **Prediction path:** user inputs → DataFrame with the expected feature columns → saved model prediction → hard policy when both core fields are present → result displayed by Streamlit or printed by the CLI.

### File responsibilities

| File | Responsibility |
|---|---|
| `risk_engine.py` | Shared data cleaning, target creation, ML pipeline, rule-based decision, explanations, and prediction helper |
| `train.py` | Loads data, trains and evaluates the model, writes model artifact and metrics |
| `app.py` | Streamlit web interface |
| `predict.py` | Command-line prediction from JSON |
| `eda.py` | Basic reproducible exploratory data analysis (EDA) |
| `requirements.txt` | Python package dependencies |
| `data/dcs_student_data.csv` | Source dataset |
| `models/risk_model.joblib` | Serialized fitted pipeline created by training |
| `models/metrics.json` | Evaluation and experiment details written by training |

### End-to-end flow

```text
                  TRAINING
CSV → clean_data() → add_risk_label() → labeled records
                                      ↓
                           stratified train/test split
                                      ↓
                         preprocessing + Random Forest
                                      ↓
                     evaluation → model + metrics saved

                 PREDICTION
UI form / JSON → prepare_input() → saved sklearn Pipeline
                                      ↓
                     Random Forest class + probabilities
                                      ↓
                  hard policy if Attendance and Final_Score
                            are both available
                                      ↓
                          risk + explanation + UI
```

---

## 2. `risk_engine.py` — shared logic

This is the core module. Both training and inference import functions from it so that they share the same feature list, cleaning rules, and model design.

### 2.1 Imports

```python
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
```

- **Path** builds filesystem paths in a way that works across operating systems.
- **NumPy** provides `NaN` and array-oriented utilities.
- **pandas** represents and transforms tabular data.
- **ColumnTransformer** applies different preprocessing to different column groups.
- **RandomForestClassifier** is the classification algorithm.
- **SimpleImputer** fills missing values.
- **OneHotEncoder** converts categories into numeric indicator columns.
- **Pipeline** chains preprocessing and modeling into one fitted object.
- **joblib** is imported but is not used in the currently shown `risk_engine.py` implementation; it is used elsewhere to save/load the model.

### 2.2 Project root and feature lists

```python
ROOT = Path(__file__).resolve().parent
NUMERIC = [...]
CATEGORICAL = [...]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Risk_Label"
```

- `__file__` identifies the current Python file.
- `.resolve().parent` finds the directory containing the project module.
- `NUMERIC` is the explicit list of numeric model inputs.
- `CATEGORICAL` is the explicit list of categorical inputs.
- `FEATURES` combines both lists in the expected model-input order.
- `TARGET` names the column generated for supervised learning.

**Why explicit lists?** They make the model schema predictable. The code does not accidentally use IDs, names, email, Grade, or Total_Score just because those columns happen to exist in the CSV.

The input features are Age, Attendance, assessment scores, subject scores, Gender, Department, and test-preparation status.

### 2.3 `clean_data(df)`

This function standardizes raw data before target generation and training.

**Copy the DataFrame**

```python
df = df.copy()
```

This avoids mutating the caller's DataFrame as a side effect.

**Trim column names and remove exact duplicate rows**

```python
df.columns = [c.strip() for c in df.columns]
df = df.drop_duplicates().reset_index(drop=True)
```

- Trimming prevents accidental column-name mismatches caused by surrounding spaces.
- `drop_duplicates()` removes rows that are identical across all columns.
- `reset_index(drop=True)` creates a clean sequential index after rows are removed.

This only removes *exact duplicates*; it does not identify two different records that might belong to the same student.

**Normalize categorical fields**

```python
df[c] = df[c].astype("string").str.strip()
```

Converts values to pandas' string dtype and removes surrounding whitespace. This helps avoid treating values such as `"Male"` and `" Male "` as different categories.

Then:
- Gender uses `.str.title()`, so casing is normalized.
- Department uses title casing and maps `"Cs"` to `"Computer Science"` and `"Math"` to `"Mathematics"`.
- `test_preparation_course` uses lowercase.

This is a small, explicit normalization step, not a general fuzzy-matching system. It only handles the variations specified in the code.

**Convert numeric fields**

```python
df[c] = pd.to_numeric(df[c], errors="coerce")
```

Valid numeric strings become numbers. Values that cannot be parsed become `NaN` instead of raising an exception. This is useful when a CSV column contains whitespace or text mixed with numeric values.

**Validate ranges**

- Age outside 15–80 becomes `NaN`.
- Other numeric model fields outside 0–100 become `NaN`.

```python
df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan
```

The function does not delete an entire student row because one feature is invalid. It marks that value missing so the downstream imputer can handle it.

**Interview nuance:** these ranges are project assumptions. In a real deployment, they should be checked against domain requirements and documented. A value converted to missing is not the same as a value proven to be erroneous.

### 2.4 `add_risk_label(df)`

This function creates the target variable from two fields: Attendance and Final_Score.

```python
high = (a < 75) | (f < 55)
medium = ((a >= 75) & (a < 80)) | ((f >= 55) & (f < 70))
```

- `|` means logical **OR**: either high-risk condition is enough.
- `&` means logical **AND**: both sides of that condition must hold.
- `<`, `>=` define the thresholds.

The labels are assigned with `np.select`:

```python
df[TARGET] = np.select(
    [high.fillna(False), medium.fillna(False)],
    ["High", "Medium"],
    default="Low"
)
```

The first matching condition wins, so **High is checked before Medium**. If neither condition matches, the default is Low.

Then:

```python
df.loc[a.isna() | f.isna(), TARGET] = pd.NA
```

If either Attendance or Final_Score is missing, the target is set to missing. That is important: the training process does not invent a policy label when one of the two fields needed by the policy is absent.

**Risk policy**

| Label | Rule |
|---|---|
| High | Attendance < 75 **OR** Final_Score < 55 |
| Medium | No High condition, and Attendance < 80 **OR** Final_Score < 70 (with the relevant lower bounds in the code) |
| Low | Neither High nor Medium condition; in normal complete records this means Attendance ≥ 80 **AND** Final_Score ≥ 70 |

The medium condition in code is explicitly `75 <= Attendance < 80 OR 55 <= Final_Score < 70`. High takes priority when both a high and medium condition could apply.

**Important ML concept:** this creates a *proxy label*. It is not an independently observed outcome. The model is learning to reproduce the policy.

### 2.5 `build_pipeline()`

This creates one scikit-learn Pipeline containing preprocessing followed by the classifier.

**Numeric preprocessing**

```python
SimpleImputer(strategy="median", add_indicator=True)
```

- Missing numeric values are filled with the median learned from the training data.
- `add_indicator=True` adds indicator columns for features that had missing values during fitting, allowing the model to use missingness information as a signal.

**Categorical preprocessing**

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
```

This is a nested pipeline:
1. Fill missing categorical values with the most frequent training value.
2. One-hot encode each category into numeric columns.
3. `handle_unknown="ignore"` avoids an error if a new category appears at inference time. Its unseen category does not activate a known category column.
4. `sparse_output=False` requests a dense encoded array.

**ColumnTransformer**

```python
prep = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC),
    ("cat", categorical_pipeline, CATEGORICAL),
])
```

This routes each column group to the correct transformer and combines the transformed outputs.

**Random Forest configuration**

```python
RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

- `n_estimators=400`: build 400 decision trees and aggregate their predictions.
- `min_samples_leaf=2`: each leaf must contain at least two training samples, which can reduce overly specific leaves.
- `class_weight="balanced"`: adjusts class weights inversely according to class frequencies in the training data.
- `random_state=42`: makes the randomized training procedure reproducible for the same data and software setup.
- `n_jobs=-1`: use available CPU cores for parallelizable work.

Finally:

```python
return Pipeline([("preprocessor", prep), ("model", model)])
```

The key advantage is that calling `.fit()` learns the imputers, category vocabulary, and trees together. At inference, calling `.predict()` on that same pipeline applies the fitted transformations in the correct order. This helps prevent training/serving preprocessing mismatch.

### 2.6 `hard_rule(attendance, final_score)`

This is the deterministic policy implementation for an individual prediction.

- If either value is absent or `NaN`, it returns `None`.
- Otherwise, it applies the High rule first, then Medium, and finally Low.

This rule is separate from the Random Forest. In the current design, when both core inputs are available, the returned **risk label** comes from this rule rather than from the classifier's predicted label.

### 2.7 `prepare_input(record)`

```python
return pd.DataFrame(
    [{c: record.get(c, np.nan) for c in FEATURES}],
    columns=FEATURES
)
```

The UI or CLI supplies a dictionary. This function:
1. Creates exactly one record.
2. Includes every expected feature in the defined order.
3. Fills any omitted feature with `NaN`.
4. Returns a one-row DataFrame, which is the format expected by the scikit-learn pipeline.

Using `record.get` means an omitted optional field does not cause a KeyError.

### 2.8 `explain(record)`

This creates human-readable reasons from the supplied core fields.

- If Attendance is present, it explains the thresholds for attendance.
- If Final_Score is present, it explains the thresholds for the final score.
- If neither is present, it says the model used available fields with imputation.

These are threshold explanations, not SHAP explanations and not a full explanation of the Random Forest's internal decision path. Be precise about that distinction in an interview.

### 2.9 `predict(model, record)`

This is the shared inference function used by the UI and CLI.

1. `prepare_input(record)` formats the input.
2. `model.predict(x)[0]` obtains the Random Forest's predicted class.
3. `model.predict_proba(x)[0]` obtains its probability estimates for each class.
4. `hard_rule(...)` checks whether the deterministic policy can be applied.
5. The return dictionary includes:
   - `risk`: policy result if available, otherwise the model class
   - `model_prediction`: Random Forest class, regardless of policy override
   - `probabilities`: class-to-probability mapping
   - `confidence`: largest Random Forest class probability
   - `rule_risk`: the rule result, or `None`
   - `decision_source`: indicates policy or limited-input model mode
   - `explanation`: generated text reasons

**Confidence caveat:** `max(predict_proba)` is the model's highest class probability, not a guarantee that the prediction is correct. Probability calibration has not been separately evaluated here. Also, if the policy overrides the model class, the displayed confidence still describes the Random Forest output, not the deterministic policy decision.

---

## 3. `train.py` — training and evaluation

### 3.1 Imports and paths

The script imports pandas, JSON, joblib, train/test splitting, evaluation metrics, and shared functions from `risk_engine.py`.

```python
DATA = ROOT / "data" / "dcs_student_data.csv"
MODEL = ROOT / "models" / "risk_model.joblib"
METRICS = ROOT / "models" / "metrics.json"
```

These paths are based on the script location rather than the terminal's current working directory. The output model is saved under `models/`; the metrics are written to JSON.

### 3.2 `main()`: load, clean, label

```python
raw = pd.read_csv(DATA)
cleaned = clean_data(raw)
labeled = add_risk_label(cleaned).dropna(subset=[TARGET]).copy()
```

- Read the CSV into a DataFrame.
- Apply the shared cleaning function.
- Generate the policy target.
- Remove rows where the target is missing, because supervised training requires a known label.

Rows with a missing Attendance or Final_Score cannot be used as labeled examples under this policy. Other missing predictor fields can remain because the pipeline imputes them.

### 3.3 Train/test split

```python
train_test_split(
    labeled[FEATURES], labeled[TARGET],
    test_size=0.20,
    stratify=labeled[TARGET],
    random_state=42
)
```

- `X` contains predictor features.
- `y` contains the target label.
- `test_size=0.20` reserves 20% for evaluation and uses 80% for training.
- `stratify=y` aims to preserve the class proportions in both subsets.
- `random_state=42` makes the split repeatable.

The model is fit only on the training subset. The test subset is used to estimate performance on held-out records.

### 3.4 Fit and predict

```python
model = build_pipeline()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

The pipeline learns imputation values, category encoding, and Random Forest trees using training data. It then predicts labels for test records.

### 3.5 Metrics

The script calculates:
- **Accuracy:** fraction of test predictions that match the labels.
- **Balanced accuracy:** average recall across classes, useful when class sizes differ.
- **Classification report:** per-class precision, recall, F1, and support.
- **Confusion matrix:** counts actual-vs-predicted labels in the fixed order High, Medium, Low.

```python
confusion_matrix(y_test, pred, labels=["High", "Medium", "Low"])
```

When reading the matrix, rows represent actual classes and columns represent predicted classes (as returned by scikit-learn). The fixed label order makes interpretation consistent between runs.

`zero_division=0` tells the classification report what to return if a precision/recall calculation has a zero denominator.

### 3.6 Feature importance

```python
names = model.named_steps["preprocessor"].get_feature_names_out()
importances = pd.Series(
    model.named_steps["model"].feature_importances_,
    index=names
)
```

The fitted preprocessor produces transformed feature names. The fitted Random Forest exposes impurity-based feature importances for those transformed columns.

The script then strips transformer prefixes and sums importance values back to their original source field. This matters for one-hot encoded columns: a source such as Department may have multiple encoded columns, so their importance values are added together.

**Caveat:** impurity-based feature importance is a model-specific summary, not proof of causality. Correlated inputs and the policy-derived target affect its interpretation.

### 3.7 Metrics JSON and model persistence

The script assembles a dictionary with dataset counts, scores, classification report, confusion matrix, risk distribution, feature importances, and the rule definitions.

```python
MODEL.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL)
METRICS.write_text(json.dumps(metrics, indent=2))
```

- Creates the `models` directory if it does not already exist.
- Serializes the **entire fitted pipeline** to a Joblib file.
- Writes metrics as readable, indented JSON.

The script prints a short subset of the results so the user can see the training outcome in the terminal.

### 3.8 Main guard

```python
if __name__ == "__main__":
    main()
```

This runs training only when `train.py` is executed directly. Importing it from another module will not automatically start training.

---

## 4. `app.py` — Streamlit user interface

### 4.1 Imports and page setup

- `joblib` loads the trained pipeline.
- `streamlit` builds the web interface.
- `Path` locates the model file.
- `predict` imports shared inference logic.

```python
st.set_page_config(...)
st.title(...)
st.caption(...)
```

These set the browser-tab title/icon, page layout, heading, and short description.

### 4.2 Check and load the model

```python
model_path = Path("models/risk_model.joblib")
if not model_path.exists():
    st.error(...)
    st.stop()
model = joblib.load(model_path)
```

The app stops early with a useful message if the trained model file is absent. This means the training script must be run before using the app on a fresh checkout.

The path is relative to the current working directory, so launch Streamlit from the project root.

### 4.3 Input form

The form groups inputs into:
- Core risk inputs: Attendance and Final Score.
- Academic information: midterm, assignments, quizzes, participation, projects, and subject scores.
- Student information: age, gender, department, and preparation-course status.

`st.columns()` arranges controls side by side. The `values` dictionary collects optional fields. Empty select-box values are filtered out before prediction.

`with st.form("risk_form")` batches form changes so the assessment is performed when the user clicks the submit button, rather than rerunning inference on every input change.

### 4.4 Validation and prediction

After submission:
1. Attendance and Final Score are added to the dictionary.
2. Missing numeric values and empty strings are removed.
3. The app checks that at least two fields were provided.
4. It calls `predict(model, values)`.

This two-field minimum is a UI validation rule. It does not mean every combination of two fields is equally informative.

### 4.5 Results

The app displays:
- risk level
- model confidence
- number of inputs used
- a High/Medium/Low recommendation message
- decision source
- textual explanation
- class probability bar chart

The app distinguishes policy mode from limited-input mode. If Attendance and Final Score are not both supplied, it warns that the Random Forest is providing a proxy prediction from the available values.

**Code review issue to fix before a live demo:** the current `app.py` accesses `result["inputs_used"]`, but the current `predict()` function in `risk_engine.py` does **not** return an `inputs_used` key. As written, a submitted valid form can raise a `KeyError` at the “Inputs Used” metric. The implementation should either add an `inputs_used` field to `predict()` or calculate the count in `app.py`. It is better to fix and test this rather than overlook it during interview preparation.

Another compatibility point to verify in the target environment is whether the installed Streamlit version supports `st.number_input(..., value=None)` as used by this app. Keep `requirements.txt` aligned with the version actually tested.

---

## 5. `predict.py` — command-line inference

### 5.1 Argument parser

```python
parser.add_argument("--model", default="models/risk_model.joblib")
parser.add_argument("--json", required=True, help="Student fields as a JSON object")
```

- `--model` lets the user choose a model path; otherwise it uses the default.
- `--json` is required and carries one student's fields as a JSON object.

Example:

```bash
python predict.py --json '{"Attendance (%)":62,"Final_Score":51,"math_score":80}'
```

### 5.2 Load, parse, predict, print

```python
model = joblib.load(args.model)
result = predict(model, json.loads(args.json))
print(json.dumps(result, indent=2))
```

- Joblib loads the trained pipeline.
- `json.loads` converts the JSON text to a Python dictionary.
- Shared `predict()` handles the input and returns the result.
- `json.dumps(..., indent=2)` prints formatted JSON.

The CLI and Streamlit interface therefore use the same prediction function, avoiding duplicate business logic.

The main guard ensures this process runs only when the script is executed directly.

---

## 6. `eda.py` — exploratory data analysis

This script performs **printed/tabular EDA**, not chart generation in its current form.

### Imports and data path

It loads pandas and reuses `clean_data()` and `add_risk_label()` so that EDA uses the same cleaning and risk-label definitions as training.

### Main steps

1. Read the CSV into `raw`.
2. Clean it into `clean`.
3. Add risk labels into `labeled`.
4. Print raw shape (rows and columns).
5. Print the count of exact duplicate rows.
6. Print cleaned shape.
7. Print column names and data types.
8. Print risk-label distribution, including missing labels because `dropna=False`.
9. Print numeric correlations with Final_Score, sorted from highest to lowest.

The correlation output is descriptive. Pearson correlation mainly summarizes linear association and does not prove cause and effect. A value near zero means little linear association in this sample; it does not rule out every possible nonlinear relationship.

**Scope note:** although the README describes useful plots to make, the current `eda.py` only prints these summaries. Histograms, boxplots, scatterplots, and a heatmap would need to be added to the script to be generated automatically.

---

## 7. `requirements.txt`

The file lists third-party dependencies and minimum versions:

- pandas: data frames and CSV handling
- NumPy: numeric operations and missing-value markers
- scikit-learn: preprocessing, Random Forest, splitting, and evaluation
- Joblib: model serialization
- Streamlit: web interface

The listed versions are lower bounds. For reproducibility, a production or submitted project may benefit from a tested lock/pinned requirements file so future package releases do not unexpectedly change behavior.

---

## 8. Why the preprocessing belongs inside the Pipeline

This is a common interview topic.

If the imputer or encoder were fitted on the entire dataset before splitting, information from the test set could influence preprocessing. That is a form of data leakage.

Here, the split occurs before `model.fit()`, and the preprocessing is part of the Pipeline. Therefore:
- imputation statistics are learned from `X_train`;
- one-hot category mappings are learned from `X_train`;
- the test set is transformed using those fitted training transformations;
- inference uses the same transformations saved with the model.

This is a sound design pattern for scikit-learn projects.

**Important distinction:** the target itself is derived from Attendance and Final_Score, and those same fields are included as features. That is target construction by design, so the high accuracy is expected and should be explained honestly.

---

## 9. Interview-ready explanation of the project

A concise, honest version:

> “I built a student academic-risk classification application in Python. I first clean the CSV by removing exact duplicates, normalizing categorical strings, converting numeric fields, and marking invalid ranges as missing. I then create a transparent High/Medium/Low policy label from attendance and final-score thresholds. A scikit-learn Pipeline imputes missing numerical and categorical features, one-hot encodes categories, and trains a 400-tree Random Forest. I use a stratified 80/20 train/test split and report accuracy, balanced accuracy, per-class precision/recall/F1, a confusion matrix, and feature importance. The fitted pipeline and metrics are saved with Joblib and JSON. Users can assess a record through Streamlit or pass JSON to a CLI. Because the target is defined by a rule rather than a future observed outcome, I describe this as policy-based risk classification, not validated dropout prediction.”

---

## 10. Interview questions to practise

### Data and preprocessing

**Q: Why remove duplicate rows?**  
A: Exact duplicate records can overrepresent identical observations. I remove exact duplicates, but I do not claim this resolves duplicate students with slightly different records.

**Q: Why use median imputation?**  
A: Median is relatively robust to extreme values. It is fitted on training data through the pipeline, not on the full dataset.

**Q: Why one-hot encode categories?**  
A: The Random Forest expects numeric inputs. One-hot encoding represents categories without imposing an artificial numeric order.

**Q: Why not use names, emails, or Student_ID?**  
A: They are identifiers, not generalizable academic signals, and may introduce noise or privacy concerns.

**Q: What is data leakage?**  
A: Information from evaluation data influences training. Putting imputation and encoding inside the pipeline and fitting after the split helps avoid preprocessing leakage.

### Model and evaluation

**Q: Why Random Forest?**  
A: It can model nonlinear relationships and feature interactions, works with the transformed mixed feature set, and offers feature importance. It is also straightforward to deploy.

**Q: What is stratification?**  
A: It keeps class proportions approximately consistent in train and test subsets.

**Q: Accuracy vs balanced accuracy?**  
A: Accuracy is the overall fraction correct. Balanced accuracy averages recall across classes so performance on a smaller class matters more.

**Q: What does the confusion matrix show?**  
A: It shows how many records of each actual class were predicted as each class, revealing which risk levels are confused.

**Q: Why is accuracy so high?**  
A: The label is generated directly from Attendance and Final_Score, and those same features are inputs. The model can learn the rule. This does not validate real-world future-risk prediction.

**Q: Does feature importance prove causation?**  
A: No. It describes how the fitted Random Forest uses features according to its importance calculation. It is not causal evidence.

### Prediction and design

**Q: Why have both a Random Forest and a hard rule?**  
A: The hard rule makes the policy decision transparent when both defining fields are available. The model provides a fallback classification when those fields are missing. The two outputs should be distinguished clearly.

**Q: What does the confidence mean?**  
A: It is the highest Random Forest probability estimate. It is not a guarantee of correctness and may not correspond to the final policy-overridden risk label.

**Q: What would you improve next?**  
A: Fix and test the missing `inputs_used` result field; add automated tests for threshold boundaries, missing values, and unseen categories; generate the EDA plots in code; pin tested dependency versions; evaluate probability calibration; and, for genuine future-risk prediction, obtain an independent outcome label and use a time-aware validation strategy.

---

## 11. Suggested run order

From the repository root:

```bash
pip install -r requirements.txt
python eda.py
python train.py
python predict.py --json '{"Attendance (%)":62,"Final_Score":51,"math_score":80}'
streamlit run app.py
```

Run training before the CLI or Streamlit app because those tools expect the saved model at `models/risk_model.joblib`.

## 12. Final checklist before the technical interview

- [ ] Explain the risk thresholds and why High is checked first.
- [ ] Explain why rows with missing Attendance or Final_Score cannot receive a policy label.
- [ ] Explain median/mode imputation and one-hot encoding.
- [ ] Explain why preprocessing is fitted only on training data.
- [ ] Explain the difference between accuracy, balanced accuracy, precision, recall, and F1.
- [ ] Explain why the high score is not real-world dropout-prediction validation.
- [ ] Know what is stored in the Joblib model and metrics JSON.
- [ ] Fix the `inputs_used` mismatch and test the Streamlit flow before demoing it.
- [ ] Be clear that `eda.py` currently prints summaries; it does not automatically generate the README's suggested plots.
