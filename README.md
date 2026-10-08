# Student Academic Risk Engine

## 1. Project Overview

This project implements a student academic-risk identification system using the supplied **DCS student dataset** and a **Random Forest** classifier.

The project follows the AI/ML recruitment task structure:

1. Data Exploration
2. Data Preprocessing
3. Machine Learning Model
4. Model Evaluation
5. Prediction & Recommendation
6. Automation-ready project structure

The task asks the system to identify students who may need academic intervention, define a transparent Low/Medium/High risk criterion, evaluate the model with suitable metrics, and provide a recommendation with the prediction. The supplied task also specifically asks for exploration of useful patterns, the relationship between attendance and marks, suitable visualizations, preprocessing, feature selection, model evaluation, and prediction/recommendation. fileciteturn36file0L32-L43

---

# 2. Dataset

The supplied CSV is stored at:

```
data/dcs_student_data.csv
```

The task description identifies 21 fields covering student identity, demographics, attendance, academic assessments, overall score/grade, preparation-course status, and subject scores. fileciteturn36file0L6-L29

### Dataset columns

| Column | Meaning |
|---|---|
| Student_ID | Unique student identifier |
| First_Name | Student first name |
| Last_Name | Student last name |
| Email | Student email |
| Gender | Student gender |
| Age | Student age |
| Department | Student department |
| Attendance (%) | Attendance percentage |
| Midterm_Score | Midterm examination score |
| Final_Score | Final examination score |
| Assignments_Avg | Average assignment score |
| Quizzes_Avg | Average quiz score |
| Participation_Score | Participation score |
| Projects_Score | Project score |
| Total_Score | Overall score field supplied in the dataset |
| Grade | Overall grade field supplied in the dataset |
| test_preparation_course | Test-preparation-course status |
| math_score | Mathematics score |
| reading_score | Reading score |
| writing_score | Writing score |
| science_score | Science score |

---

# 3. Data Exploration

## 3.1 Initial inspection

The first step was to inspect:

- Dataset shape
- Column names
- Data types
- Duplicate rows
- Missing values
- Categorical values
- Numeric ranges
- Distribution of grades
- Relationships between Total_Score and the other available variables
- Relationship between Attendance and Final_Score/marks

The raw dataset contains:

- **10,030 rows**
- **21 columns**
- **30 exact duplicate rows**

After removing exact duplicate rows, **10,000 unique rows** remain.

This inspection was important because the dataset contains intentionally inconsistent-looking values, missing entries, inconsistent category formatting, and relationships that are not necessarily the same as we would expect in a real academic dataset.

---

## 3.2 Grade vs Total_Score relationship

One of the most important findings from exploration was the relationship between the supplied `Grade` and `Total_Score`.

At first glance, Grade would normally be expected to correspond strongly to Total_Score. However, the supplied dataset does **not** show a clean or reliable relationship.

The observed Total_Score statistics by Grade were:

| Grade | Count | Mean Total_Score | Minimum | Maximum |
|---|---:|---:|---:|---:|
| A | 2,969 | 74.63 | 50.02 | 99.99 |
| B | 1,960 | 75.03 | 50.04 | 99.99 |
| C | 1,637 | 75.49 | 50.02 | 99.98 |
| D | 1,806 | 75.14 | 50.03 | 99.95 |
| F | 1,658 | 75.17 | 50.06 | 99.98 |

The important observation is that **all five grade categories have almost the same average Total_Score and almost the same overall range**.

For example:

- A has a mean Total_Score of about 74.63.
- C has a slightly higher mean of about 75.49.
- F has a mean of about 75.17.

This is not a sensible monotonic Grade → Total_Score relationship.

There is also very large overlap in the Total_Score ranges across all grades. Therefore, `Grade` cannot be treated as a trustworthy numerical transformation of `Total_Score` in this dataset.

### Decision

Because of this inconsistency, **Grade is excluded from the machine-learning predictors**.

This is an important preprocessing decision rather than something that should be hidden. The model should not learn an apparently meaningful relationship from a field that the exploratory analysis shows to be unreliable.

---

# 4. Total_Score vs Other Parameters

The next exploration step was to calculate correlations between Total_Score and the available numerical parameters.

The observed Pearson correlations were extremely close to zero for nearly every predictor.

| Variable | Correlation with Total_Score |
|---|---:|
| math_score | +0.0123 |
| writing_score | +0.0067 |
| Assignments_Avg | +0.0042 |
| reading_score | +0.0028 |
| Midterm_Score | +0.0011 |
| Quizzes_Avg | ~0.0000 |
| Age | -0.0036 |
| test_preparation_course* | -0.0037 |
| science_score | -0.0049 |
| Final_Score | -0.0053 |
| Attendance (%) | -0.0102 |
| Participation_Score | -0.0159 |
| Projects_Score | -0.0171 |

*The preparation-course field is binary in the supplied data, so this correlation is only a simple numerical association and should not be interpreted as a continuous relationship.

### Main finding

There is **no strong linear relationship between Total_Score and the other supplied numerical parameters in this dataset**.

This is especially important because it means we should not make unsupported claims such as:

> "Higher attendance automatically produces higher Total_Score."

or

> "Higher math/reading/writing scores strongly determine Total_Score."

The exploration does not support those claims.

The correct conclusion is simply that **the supplied dataset shows very weak linear correlations between Total_Score and the examined predictors**.

---

# 5. Attendance vs Marks

The recruitment task specifically asks us to check the relationship between attendance and marks. fileciteturn36file0L36-L38

The Pearson correlation between:

- `Attendance (%)`
- `Final_Score`

was approximately:

**-0.0141**

This is effectively a near-zero linear relationship in the supplied data.

Therefore, the exploration does **not** support claiming a meaningful natural positive correlation between attendance and Final_Score.

This finding was used to keep the analysis honest: the risk policy below is a **defined intervention policy**, not a claim that the dataset proves attendance causes academic performance.

### Recommended visualizations

The exploration stage is designed around the following plots:

1. Attendance distribution histogram
2. Final_Score distribution histogram
3. Total_Score distribution histogram
4. Grade frequency bar chart
5. Grade vs Total_Score boxplot
6. Attendance vs Final_Score scatter plot
7. Total_Score vs Attendance scatter plot
8. Total_Score vs Final_Score scatter plot
9. Correlation heatmap for numerical variables
10. Boxplots for important numerical features to identify unusual values/outliers

These visualizations directly support the task requirement to explore useful patterns and create suitable graphs. fileciteturn36file0L34-L38

---

# 6. Data Quality Issues Found

The dataset contains several quality issues that must be handled before machine learning.

### 6.1 Exact duplicate rows

The raw dataset contains **30 exact duplicate rows**.

These rows are removed before model training.

```
Raw rows:                 10,030
Duplicate rows removed:      30
Rows after deduplication: 10,000
```

### 6.2 Missing values

Several numerical and categorical fields contain missing values.

Examples include missing:

- Age
- Attendance
- Midterm_Score
- Final_Score
- Assignments_Avg
- Quizzes_Avg
- Participation_Score
- Projects_Score
- Subject scores

Missing values are not replaced blindly in the raw dataset. They are handled according to their role in preprocessing and model training.

### 6.3 Inconsistent categorical formatting

The Department field contains variants such as:

- `CS`
- `Computer Science`
- `Math`
- `Mathematics`
- different capitalization
- leading/trailing spaces

Gender also contains inconsistent formatting such as:

- `Female`
- ` FEMALE `
- `Male`
- ` MALE `

These are normalized before modeling.

### 6.4 Numeric values stored inconsistently

`math_score` requires numeric conversion because the raw field contains values that may be represented as strings/whitespace.

The numeric predictor columns are converted using safe numeric coercion.

### 6.5 Invalid numeric values

Attendance and academic scores should lie between 0 and 100.

Any value outside this range is treated as invalid and converted to missing.

Age is also checked for plausibility. Ages outside the chosen 15–80 range are treated as invalid.

---

# 7. Data Preprocessing

The preprocessing pipeline follows a deliberate sequence.

```
RAW CSV
   ↓
Inspect shape / columns / data types
   ↓
Remove exact duplicate rows
   ↓
Clean categorical values
   ↓
Convert numerical columns to numeric
   ↓
Detect invalid ranges
   ↓
Invalid values → NaN
   ↓
Create policy-based Risk_Label
   ↓
Select ML features
   ↓
Median imputation for numerical features
   ↓
Most-frequent imputation for categorical features
   ↓
One-hot encoding
   ↓
Train/test split
   ↓
Random Forest
```

## Step 1 — Remove exact duplicates

All completely duplicated rows are removed.

This prevents identical observations from unnecessarily appearing multiple times in the model-training data.

---

## Step 2 — Clean categorical variables

The following categorical variables are normalized:

- Gender
- Department
- test_preparation_course

Examples:

```
" FEMALE " → "Female"
" MALE "   → "Male"
"CS"       → "Computer Science"
"Math"     → "Mathematics"
" engineering " → "Engineering"
```

This prevents the model from treating formatting variations as separate categories.

---

## Step 3 — Convert numerical columns

The numerical fields are converted using safe numeric conversion.

This includes:

- Age
- Attendance (%)
- Midterm_Score
- Final_Score
- Assignments_Avg
- Quizzes_Avg
- Participation_Score
- Projects_Score
- math_score
- reading_score
- writing_score
- science_score

Invalid numeric strings become missing values instead of crashing the pipeline.

---

## Step 4 — Validate numeric ranges

For academic scores and attendance:

```
0 <= value <= 100
```

Values outside this range are considered invalid and converted to `NaN`.

For Age:

```
15 <= Age <= 80
```

Values outside this range are treated as invalid.

This prevents impossible observations from directly influencing the Random Forest.

---

# 8. Target / Risk Definition

The recruitment task explicitly allows us to define our own Low/Medium/High risk criteria and explain the approach. fileciteturn36file0L46-L58

The project uses a transparent intervention policy:

### High Risk

```
Attendance < 75
OR
Final_Score < 55
```

### Medium Risk

```
75 <= Attendance < 80
OR
55 <= Final_Score < 70
```

### Low Risk

```
Attendance >= 80
AND
Final_Score >= 70
```

The High condition is checked first, so if either high-risk condition is satisfied, the student is classified as High Risk.

### Why use these rules?

The dataset does not contain an independent future outcome such as:

- dropout
- course failure in a future semester
- intervention success
- retention
- graduation

Therefore, we cannot honestly call this a validated dropout-prediction model.

Instead, `Risk_Label` is a **policy-defined proxy target** designed for the recruitment-task requirement of identifying students who may need academic intervention.

---

# 9. Feature Selection

The following fields are used as model predictors:

### Numerical

- Age
- Attendance (%)
- Midterm_Score
- Final_Score
- Assignments_Avg
- Quizzes_Avg
- Participation_Score
- Projects_Score
- math_score
- reading_score
- writing_score
- science_score

### Categorical

- Gender
- Department
- test_preparation_course

### Fields excluded

The following fields are deliberately excluded:

- Student_ID
- First_Name
- Last_Name
- Email
- Grade
- Total_Score

### Why exclude identifiers?

Student ID, name and email identify a student but do not represent academic behavior that should generalize to another student.

Using them would introduce noise and could create misleading patterns.

### Why exclude Grade?

The exploration showed that Grade does not have a reliable relationship with Total_Score in this supplied dataset.

### Why exclude Total_Score?

Total_Score is an overall aggregate field and its relationship with the other supplied parameters is unexpectedly weak. More importantly, the project is intended to work from individual academic/attendance inputs rather than rely on a questionable aggregate field.

Keeping these fields out makes the model input more appropriate for the intended prediction interface.

---

# 10. Missing-Value Handling

Missing values are handled inside the ML preprocessing pipeline.

### Numerical variables

Numerical missing values are filled using:

**Median imputation**

Median is preferred here because it is less sensitive to extreme observations than the mean.

### Categorical variables

Categorical missing values are filled using:

**Most-frequent-value imputation**

This keeps the pipeline simple and compatible with unseen categories.

---

# 11. Encoding

Categorical features are transformed using **One-Hot Encoding**.

For example:

```
Department
    Computer Science
    Engineering
    Business
    Mathematics
```

becomes separate binary model features.

`handle_unknown="ignore"` is used so that an unseen category at prediction time does not crash the application.

---

# 12. Machine Learning Model

The selected model is:

## Random Forest Classifier

Configuration:

- 400 trees
- `min_samples_leaf = 2`
- `class_weight = "balanced"`
- `random_state = 42`
- parallel processing enabled

Random Forest was selected because it:

- handles nonlinear decision boundaries
- works well with mixed engineered features
- can capture interactions between inputs
- provides feature importance
- is straightforward to integrate into the Streamlit application

The recruitment task explicitly lists Random Forest as one of the suitable model choices. fileciteturn36file0L46-L55

---

# 13. Train/Test Split

The labeled dataset is split using:

```
80% training
20% testing
```

The split is **stratified** using Risk_Label so that the High, Medium and Low classes remain represented in both sets.

Random state:

```
42
```

This makes the experiment reproducible.

---

# 14. Model Evaluation

The project evaluates the model using:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics align with the evaluation criteria suggested in the recruitment task. fileciteturn36file0L60-L67

The current trained model achieves approximately:

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Balanced Accuracy | 99.91% |

The high score is expected because Attendance and Final_Score directly define the target label.

Therefore:

> **99.95% accuracy must not be presented as 99.95% real-world dropout/failure prediction accuracy.**

The model is primarily learning the defined risk policy.

---

# 15. Feature Importance

The most important model features are:

| Feature | Approx. Importance |
|---|---:|
| Attendance (%) | 47.49% |
| Final_Score | 34.09% |
| Projects_Score | 1.91% |
| Quizzes_Avg | 1.84% |
| Assignments_Avg | 1.81% |
| Midterm_Score | 1.80% |
| Participation_Score | 1.74% |
| math_score | 1.61% |
| science_score | 1.58% |
| writing_score | 1.56% |

This is consistent with the target construction: Attendance and Final_Score are the two variables that explicitly define the risk policy.

---

# 16. Prediction Interface

The Streamlit application allows the user to enter:

- Attendance
- Final Score
- Midterm Score
- Assignments Average
- Quizzes Average
- Participation Score
- Projects Score
- Math
- Reading
- Writing
- Science
- Age
- Gender
- Department
- Test-preparation status

The application requires a minimum of **two supplied fields** and supports using up to all available fields.

### Full-input mode

When both Attendance and Final Score are supplied, the transparent hard-risk policy is used as the authoritative risk decision.

### Limited-input mode

If Attendance and/or Final Score are missing, the Random Forest can still produce a prediction using the available fields and the preprocessing pipeline.

The application clearly labels this as **limited-input mode** because the strongest policy-defining fields are missing.

---

# 17. Prediction & Recommendation

The recruitment task expects the system to return a risk level and recommendation. fileciteturn36file0L69-L83

The application provides:

### High Risk

**Recommendation:** Immediate academic intervention. Review attendance and upcoming assessments and provide targeted academic support.

### Medium Risk

**Recommendation:** Monitor the student's progress and consider targeted support before performance deteriorates further.

### Low Risk

**Recommendation:** No immediate intervention is indicated by the defined policy; continue normal academic monitoring.

The prediction output also includes:

- Risk level
- Model confidence
- Model probabilities
- Decision source
- Explanation of Attendance/Final_Score thresholds when available

---

# 18. Important Interpretation of the Findings

A key part of this project is distinguishing **what the data actually shows** from what we might expect academically.

The exploration found:

1. Grade and Total_Score do not show a sensible monotonic relationship.
2. Total_Score has near-zero Pearson correlation with the examined academic variables.
3. Attendance and Final_Score also have an approximately zero linear correlation in this supplied dataset.
4. Therefore, we do not claim unsupported natural relationships.
5. The risk rules are an explicit intervention policy chosen for the task.
6. The Random Forest learns that policy from the available predictors.

This makes the project more transparent and avoids presenting synthetic/inconsistent dataset relationships as real academic causality.

---

# 19. Project Structure

```
Student_risk/
│
├── app.py
├── risk_engine.py
├── train.py
├── predict.py
├── eda.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── dcs_student_data.csv
│   └── README.md
│
└── models/
    ├── metrics.json
    └── risk_model.joblib
```

---

# 20. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Run a CLI prediction:

```bash
python predict.py --json '{"Attendance (%)":62,"Final_Score":51,"math_score":80}'
```

Run the exploration script:

```bash
python eda.py
```

---

# 21. Files and Responsibilities

### `risk_engine.py`

Contains:

- data cleaning
- risk-label creation
- preprocessing pipeline
- Random Forest definition
- hard-risk policy
- prediction helpers
- explanation logic

### `train.py`

Responsible for:

- loading the dataset
- preprocessing
- generating Risk_Label
- train/test split
- Random Forest training
- evaluation
- saving model metrics
- saving the trained model

### `predict.py`

Provides command-line inference using a JSON student record.

### `app.py`

Provides the Streamlit user interface for interactive risk assessment.

### `eda.py`

Provides reproducible exploratory inspection of:

- dataset shape
- duplicates
- cleaned data
- risk distribution
- numerical relationships

---

# 22. Limitations

This project has several important limitations.

### Policy-derived target

The target is created using rules rather than an observed future outcome.

### Dataset consistency

The Grade/Total_Score relationship is inconsistent, and the numerical correlations are unexpectedly weak.

### No causal conclusions

Correlation results are descriptive only. They do not prove that one academic variable causes another.

### Limited-input predictions

Predictions made without Attendance and Final Score are less directly supported by the project's policy.

### Model evaluation

The near-perfect model score is partly a consequence of the target being explicitly defined by two of the model inputs.

---

# 23. Conclusion

The final system follows the recruitment task from exploration through preprocessing, modeling, evaluation and prediction.

The main analytical conclusion is not simply that one academic variable is strongly correlated with another. Instead, the supplied dataset revealed several inconsistencies:

- Grade does not behave like a reliable representation of Total_Score.
- Total_Score has almost no linear correlation with the other numerical variables.
- Attendance and Final_Score have almost no linear correlation in this dataset.

Because of these findings, the implementation avoids using questionable fields as predictors and uses an explicit, explainable risk policy:

```
High   → Attendance < 75 OR Final_Score < 55
Medium → 75–79.99 attendance OR 55–69.99 Final Score
Low    → Attendance >= 80 AND Final_Score >= 70
```

Random Forest is then used to learn this policy and provide probability-based support for the interactive prediction system.

The resulting application is therefore best described as a **transparent student academic-risk classification engine based on a defined intervention policy**, rather than a validated real-world dropout prediction system.
