"""Train and compare classifiers, save the best one.

    python -m depression_ml.train raw_data/depression_student_dataset.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, load

SEED = 42
SCORING = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "balanced_accuracy": "balanced_accuracy"}


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bin", binary, BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_models() -> dict[str, Pipeline]:
    # class_weight="balanced" so the minority class isn't ignored on imbalanced data.
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }
    return {name: Pipeline([("prep", make_preprocessor()), ("model", est)]) for name, est in estimators.items()}


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "f1": f1_score(y, pred),
        "recall": recall_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }


def run(data_path: str | Path, out_dir: str | Path = "models") -> dict:
    df = load(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"{len(df)} rows · positive rate {y.mean():.1%} · train {len(X_train)} / test {len(X_test)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    for name, model in make_models().items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)
        cv_results[name] = {m: float(scores[f"test_{m}"].mean()) for m in SCORING}
        print(f"{name:<22} " + "  ".join(f"{m}={v:.3f}" for m, v in cv_results[name].items()))

    # Model selection on cross-validated ROC-AUC; the test set is only used once, at the end.
    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"])
    best = make_models()[best_name].fit(X_train, y_train)
    test_metrics = {k: float(v) for k, v in evaluate(best, X_test, y_test).items()}
    print(f"\nBest (CV ROC-AUC): {best_name}")
    print("Held-out test: " + "  ".join(f"{m}={v:.3f}" for m, v in test_metrics.items()))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_dir / "model.joblib")
    report = {"best_model": best_name, "cv": cv_results, "test": test_metrics, "n_rows": len(df)}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="path to the raw Kaggle CSV")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
