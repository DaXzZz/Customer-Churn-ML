from pathlib import Path

import pandas as pd

from data_validation import FEATURE_COLUMNS
from train import train_model


def test_training_pipeline_returns_metrics_and_predictions() -> None:
    df = pd.read_csv(Path("data/sample_churn.csv"))

    result = train_model(df, test_size=0.25, random_state=42)

    assert set(result.metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
    }
    assert len(result.predictions) > 0
    assert set(result.predictions["predicted_churn"]).issubset({"No", "Yes"})


def test_training_pipeline_handles_unseen_categories() -> None:
    df = pd.read_csv(Path("data/sample_churn.csv"))
    result = train_model(df, test_size=0.25, random_state=42)

    new_customer = df[FEATURE_COLUMNS].head(1).copy()
    new_customer.loc[:, "payment_method"] = "PayPal"

    prediction = result.model.predict(new_customer)

    assert len(prediction) == 1

