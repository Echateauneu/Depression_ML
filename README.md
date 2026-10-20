# Depression_ML

Predicting depression risk from a student well-being survey, comparing three classifiers with proper
handling of class imbalance, plus a small Streamlit demo.

> ⚠️ **Educational project — not a medical tool.** A model trained on one online survey cannot diagnose
> anyone. If you are struggling, talk to a doctor or someone you trust.
> 🇫🇷 **3114** — numéro national de prévention du suicide (free, 24/7).

## Data

[Student Depression survey](https://www.kaggle.com/datasets/anthonytherrien/depression-student-dataset) (Kaggle).
The loader also accepts the larger version of the same questionnaire,
[`hopesb/student-depression-dataset`](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (~28k rows).

Features: age, gender, academic pressure, study satisfaction, sleep duration, study hours, financial stress,
dietary habits, suicidal thoughts, family history of mental illness. Target: `Depression` (yes/no).

### Why only one dataset (and not four merged)?

An earlier version merged four Kaggle datasets into one table. I dropped that, because:

- **Different feature sets.** Each survey asks different questions. After concatenation most columns are
  missing for most rows; median-filling them lets the model learn *which survey a row came from* rather than
  anything about depression.
- **DASS severity is a formula.** The DASS-42 severity is the sum of 14 questionnaire items. Predicting it
  from those items just re-learns a sum, so it's not a useful ML task.
- **"Sought treatment" ≠ depression.** The OSMI tech survey label is about seeking mental-health treatment,
  a different (and broader) target.

## Method

- Cleaning in [`depression_ml/data.py`](depression_ml/data.py): column-name normalisation across dataset
  versions, sleep buckets (`"7-8 hours"`) → hours, yes/no → 0/1, invalid ages → missing.
- Preprocessing inside a scikit-learn `Pipeline` (median imputation + scaling, one-hot encoding), so there
  is no train/test leakage.
- Models: **Logistic Regression** (baseline), **Random Forest**, **Histogram Gradient Boosting**, all with
  `class_weight="balanced"`.
- Selection by 5-fold stratified cross-validated **ROC-AUC** on the training split; the best model is
  scored once on a held-out 20 % test split. Recall is reported too: missing an at-risk person is the costly error.

## Results

Run training to produce `models/metrics.json`:

| Model | CV ROC-AUC | CV recall | CV F1 |
|---|---|---|---|
| Logistic Regression | _run training_ | | |
| Random Forest | | | |
| Gradient Boosting | | | |

## Usage

```bash
pip install -r requirements.txt

# 1. get the data (or download the CSV manually into raw_data/)
python -c "from depression_ml.data import download; download('small')"

# 2. train + compare models
python -m depression_ml.train raw_data/depression_student_dataset.csv

# 3. demo app
streamlit run app.py

# tests (use synthetic data, no download needed)
pytest
```

## Project structure

```
depression_ml/
  data.py    loading + cleaning
  train.py   model comparison, evaluation, saving
app.py       Streamlit demo
tests/       unit + end-to-end tests on synthetic data
```

## Next steps

- SHAP values to explain which answers drive each prediction
- Probability calibration (`CalibratedClassifierCV`) so the demo score is better grounded
- Threshold tuning for a target recall
