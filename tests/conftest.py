import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_survey(tmp_path):
    """Synthetic CSV with the same schema as the Kaggle student dataset."""
    rng = np.random.default_rng(0)
    n = 300
    pressure = rng.integers(1, 6, n)
    stress = rng.integers(1, 6, n)
    suicidal = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    logit = 0.8 * (pressure - 3) + 0.6 * (stress - 3) + 1.5 * (suicidal == "Yes") - 0.5
    label = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    df = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 35, n),
        "Academic Pressure": pressure,
        "Study Satisfaction": rng.integers(1, 6, n),
        "Sleep Duration": rng.choice(["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], n),
        "Dietary Habits": rng.choice(["Healthy", "Moderate", "Unhealthy"], n),
        "Have you ever had suicidal thoughts ?": suicidal,
        "Study Hours": rng.integers(0, 12, n),
        "Financial Stress": stress,
        "Family History of Mental Illness": rng.choice(["Yes", "No"], n),
        "Depression": label,
    })
    path = tmp_path / "survey.csv"
    df.to_csv(path, index=False)
    return path
