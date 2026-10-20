"""Load and clean the Student Depression survey data.

Supported sources (Kaggle, same questionnaire):
  - anthonytherrien/depression-student-dataset   (~500 rows)
  - hopesb/student-depression-dataset            (~28k rows, extra columns)

Download one of them (manually or with the kaggle CLI) into ./raw_data/.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("raw_data")

KAGGLE_DATASETS = {
    "small": ("anthonytherrien/depression-student-dataset", "depression_student_dataset.csv"),
    "large": ("hopesb/student-depression-dataset", "student_depression_dataset.csv"),
}

TARGET = "depression"
NUMERIC_FEATURES = [
    "age",
    "academic_pressure",
    "study_satisfaction",
    "sleep_hours",
    "study_hours",
    "financial_stress",
]
BINARY_FEATURES = ["is_female", "suicidal_thoughts", "family_history"]
CATEGORICAL_FEATURES = ["dietary_habits"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Raw column name -> clean name. Several spellings exist across dataset versions.
COLUMN_ALIASES = {
    "age": ["Age"],
    "gender": ["Gender"],
    "academic_pressure": ["Academic Pressure"],
    "study_satisfaction": ["Study Satisfaction"],
    "sleep_duration": ["Sleep Duration"],
    "dietary_habits": ["Dietary Habits"],
    "suicidal_thoughts": ["Have you ever had suicidal thoughts ?", "Have you ever had suicidal thoughts?"],
    "study_hours": ["Study Hours", "Work/Study Hours"],
    "financial_stress": ["Financial Stress"],
    "family_history": ["Family History of Mental Illness"],
    TARGET: ["Depression"],
}

# Sleep is reported as a text bucket; map each bucket to approximate hours.
SLEEP_BUCKETS = {
    "less than 5 hours": 4.5,
    "5-6 hours": 5.5,
    "6-7 hours": 6.5,
    "7-8 hours": 7.5,
    "more than 8 hours": 8.5,
}


def download(which: str = "small", raw_dir: Path = RAW_DIR) -> Path:
    """Download a dataset with the kaggle CLI (needs ~/.kaggle/kaggle.json)."""
    slug, filename = KAGGLE_DATASETS[which]
    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "--unzip", "-p", str(raw_dir)],
        check=True,
    )
    return raw_dir / filename


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    mapping = {}
    for clean, candidates in COLUMN_ALIASES.items():
        for raw in candidates:
            if raw in df.columns:
                mapping[raw] = clean
                break
    missing = set(COLUMN_ALIASES) - set(mapping.values())
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df[list(mapping)].rename(columns=mapping)


def _yes_no(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map({"yes": 1, "1": 1, "true": 1, "no": 0, "0": 0, "false": 0})


def parse_sleep(series: pd.Series) -> pd.Series:
    """'7-8 hours' -> 7.5. Unknown buckets become NaN (imputed later)."""
    s = series.astype(str).str.strip().str.strip("'\"").str.lower()
    return s.map(SLEEP_BUCKETS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Raw survey dataframe -> clean feature table + `depression` target (0/1)."""
    df = _rename_columns(df)
    out = pd.DataFrame(index=df.index)

    age = pd.to_numeric(df["age"], errors="coerce")
    out["age"] = age.where(age.between(15, 80))
    for col in ["academic_pressure", "study_satisfaction", "study_hours", "financial_stress"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["sleep_hours"] = parse_sleep(df["sleep_duration"])

    out["is_female"] = (df["gender"].astype(str).str.strip().str.lower() == "female").astype(int)
    out["suicidal_thoughts"] = _yes_no(df["suicidal_thoughts"])
    out["family_history"] = _yes_no(df["family_history"])
    out["dietary_habits"] = (
        df["dietary_habits"].astype(str).str.strip().str.strip("'\"").str.lower()
        .where(lambda s: s.isin(["healthy", "moderate", "unhealthy"]), np.nan)
    )

    out[TARGET] = _yes_no(df[TARGET])
    out = out.dropna(subset=[TARGET])
    out[TARGET] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


def load(path: str | Path) -> pd.DataFrame:
    return clean(pd.read_csv(path))
