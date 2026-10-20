import json

from depression_ml.train import run


def test_training_end_to_end(raw_survey, tmp_path):
    report = run(raw_survey, tmp_path / "models")
    assert (tmp_path / "models" / "model.joblib").exists()
    saved = json.loads((tmp_path / "models" / "metrics.json").read_text())
    assert saved["best_model"] in {"logistic_regression", "random_forest", "gradient_boosting"}
    # The synthetic label depends on the features, so every model should beat chance.
    assert report["test"]["roc_auc"] > 0.6
