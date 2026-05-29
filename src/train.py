from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_validation import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    validate_churn_data,
)


TARGET_MAPPING = {"No": 0, "Yes": 1}
LABEL_MAPPING = {0: "No", 1: "Yes"}


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict[str, object]
    predictions: pd.DataFrame


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    validate_churn_data(df)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def encode_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].map(TARGET_MAPPING)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_probability: pd.Series,
) -> dict[str, object]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_probability)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def build_prediction_output(
    customer_ids: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_probability: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": customer_ids.to_list(),
            "actual_churn": [LABEL_MAPPING[value] for value in y_true.to_list()],
            "predicted_churn": [LABEL_MAPPING[value] for value in y_pred],
            "churn_probability": [round(float(value), 4) for value in y_probability],
        }
    )


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainingResult:
    validate_churn_data(df)

    x = df[FEATURE_COLUMNS]
    y = encode_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_pred, y_probability)
    predictions = build_prediction_output(
        customer_ids=df.loc[x_test.index, "customer_id"],
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    return TrainingResult(model=model, metrics=metrics, predictions=predictions)


def save_json(data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline churn model.")
    parser.add_argument("--data", default="data/sample_churn.csv", help="Path to training CSV")
    parser.add_argument(
        "--metrics-output",
        default="outputs/sample_metrics.json",
        help="Path for metrics JSON",
    )
    parser.add_argument(
        "--predictions-output",
        default="outputs/sample_predictions.csv",
        help="Path for test prediction CSV",
    )
    parser.add_argument(
        "--model-output",
        default="models/churn_model.joblib",
        help="Path for trained model artifact",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    metrics_path = Path(args.metrics_output)
    predictions_path = Path(args.predictions_output)
    model_path = Path(args.model_output)

    df = load_data(data_path)
    result = train_model(df, test_size=args.test_size, random_state=args.random_state)

    save_json(result.metrics, metrics_path)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(predictions_path, index=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": result.model, "features": FEATURE_COLUMNS}, model_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

