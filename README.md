# Student Academic Risk Engine

A university student academic-risk engine built from the supplied `dcs_student_data.csv`.

## Architecture

```
RAW CSV
  ↓
Load / inspect
  ↓
Remove exact duplicates
  ↓
Clean categories + numeric values
  ↓
Invalid values → NaN
  ↓
Median imputation + one-hot encoding
  ↓
Hard policy creates Risk_Label
  ↓
Random Forest
  ↓
Risk + probability + explanation
```

## Risk policy

Rules are evaluated in this order:

- **High:** Attendance < 75 OR Final Score < 55
- **Medium:** 75 <= Attendance < 80 OR 55 <= Final Score < 70
- **Low:** Attendance >= 80 AND Final Score >= 70

The dataset has no independent future outcome such as dropout, failure, or intervention success. Therefore `Risk_Label` is a **policy-defined proxy target**, not a validated future-risk outcome.

## Preprocessing

1. Load the dataset.
2. Inspect shape, columns and data types.
3. Remove exact duplicate rows.
4. Clean string categories.
5. Convert `math_score` and other numeric predictors to numeric.
6. Convert Attendance/scores outside 0–100 to missing values.
7. Convert implausible ages outside 15–80 to missing values.
8. Handle missing numerical values with median imputation inside the ML pipeline.
9. One-hot encode categorical predictors.
10. Exclude identifiers, `Grade`, and `Total_Score` from ML features.
11. Use a stratified 80/20 train/test split.
12. Train a balanced Random Forest.

## Model predictors

Age, Gender, Department, Attendance (%), Midterm_Score, Final_Score, Assignments_Avg, Quizzes_Avg, Participation_Score, Projects_Score, test_preparation_course, math_score, reading_score, writing_score, science_score.

Identifiers excluded from ML:

- Student_ID
- First_Name
- Last_Name
- Email
- Grade
- Total_Score

## Run locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

CLI example:

```bash
python predict.py --json '{"Attendance (%)":62,"Final_Score":51,"math_score":80}'
```

## Interpretation

The Random Forest can obtain very high accuracy because Attendance and Final Score directly define the target label. That must **not** be presented as evidence that the model predicts real-world dropout or failure.

The Streamlit interface accepts a minimum of two fields and up to all available fields. When both Attendance and Final Score are supplied, the transparent hard policy is the authoritative decision. Without both, the app clearly labels the result as a limited-input Random Forest prediction.
