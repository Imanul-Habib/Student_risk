# Student Academic Risk Engine

Built from the supplied dcs_student_data.csv.

## Risk policy
- High: Attendance < 75 OR Final Score < 55
- Medium: 75 <= Attendance < 80 OR 55 <= Final Score < 70
- Low: Attendance >= 80 AND Final Score >= 70

The Random Forest learns this policy-defined label. This is a proxy/policy classifier, not a true dropout prediction model because the dataset has no future outcome label.

## Preprocessing
Remove exact duplicates; normalize Gender and Department; convert math_score to numeric; turn impossible/out-of-range numeric values into missing; impute inside the pipeline; one-hot encode categories; stratified 80/20 split; balanced Random Forest.

## Features
Age, Gender, Department, Attendance (%), Midterm_Score, Final_Score, Assignments_Avg, Quizzes_Avg, Participation_Score, Projects_Score, test_preparation_course, math_score, reading_score, writing_score, science_score.

Identifiers, Grade, and Total_Score are excluded.

## Run
Put the supplied CSV at data/dcs_student_data.csv.

    pip install -r requirements.txt
    python train.py
    streamlit run app.py

CLI:
    python predict.py --json '{"Attendance (%)":62,"Final_Score":51,"math_score":80}'

A near-perfect score is expected because Attendance and Final Score directly define the target; it does not prove real-world predictive power.
