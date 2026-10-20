import pandas as pd
import pytest

from depression_ml.data import FEATURES, TARGET, clean, load, parse_sleep


def test_parse_sleep_buckets():
    s = pd.Series(["7-8 hours", "'5-6 hours'", "Less than 5 hours", "Others"])
    assert parse_sleep(s).tolist()[:3] == [7.5, 5.5, 4.5]
    assert pd.isna(parse_sleep(s).iloc[3])


def test_load_produces_all_features(raw_survey):
    df = load(raw_survey)
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df["sleep_hours"].notna().all()


def test_clean_accepts_large_dataset_column_names():
    raw = pd.DataFrame({
        "Gender": ["Female"], "Age": [22], "Academic Pressure": [4], "Study Satisfaction": [2],
        "Sleep Duration": ["'6-7 hours'"], "Dietary Habits": ["Healthy"],
        "Have you ever had suicidal thoughts ?": ["No"], "Work/Study Hours": [8],
        "Financial Stress": [3], "Family History of Mental Illness": ["Yes"], "Depression": [1],
    })
    row = clean(raw).iloc[0]
    assert row["study_hours"] == 8 and row["sleep_hours"] == 6.5 and row["is_female"] == 1


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Missing expected columns"):
        clean(pd.DataFrame({"Age": [20]}))
