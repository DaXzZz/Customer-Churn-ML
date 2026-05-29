from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data_validation import FEATURE_COLUMNS, validate_columns, validate_numeric_columns
from train import LABEL_MAPPING


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Run: python src/train.py")

    artifact = joblib.load(model_path)
    return artifact["model"], artifact["features"]


def predict_churn(input_path: Path, model_path: Path) -> pd.DataFrame:
    model, feature_columns = load_model(model_path)
    df = pd.read_csv(input_path)

    validate_columns(df, feature_columns)
    validate_numeric_columns(df)

    probabilities = model.predict_proba(df[feature_columns])[:, 1]
    predictions = model.predict(df[feature_columns])

    if "customer_id" in df.columns:
        output = pd.DataFrame({"customer_id": df["customer_id"]})
    else:
        output = pd.DataFrame({"row_id": range(len(df))})

    output["predicted_churn"] = [LABEL_MAPPING[value] for value in predictions]
    output["churn_probability"] = [round(float(value), 4) for value in probabilities]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score customers with a trained churn model.")
    parser.add_argument("--input", default="data/sample_churn.csv", help="Path to input CSV")
    parser.add_argument("--model", default="models/churn_model.joblib", help="Path to model")
    parser.add_argument("--output", default="outputs/new_predictions.csv", help="Path to output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    predictions = predict_churn(input_path=Path(args.input), model_path=Path(args.model))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

