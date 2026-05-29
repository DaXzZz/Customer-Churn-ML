from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TARGET_COLUMN = "churn"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = [
    "customer_id",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
]

TARGET_VALUES = {"No", "Yes"}


def validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def validate_numeric_columns(df: pd.DataFrame) -> None:
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Column '{column}' must contain only numeric values")
        if (values < 0).any():
            raise ValueError(f"Column '{column}' must not contain negative values")


def validate_target(df: pd.DataFrame) -> None:
    values = set(df[TARGET_COLUMN].dropna().astype(str).unique())
    invalid_values = sorted(values - TARGET_VALUES)

    if df[TARGET_COLUMN].isna().any() or invalid_values:
        allowed = ", ".join(sorted(TARGET_VALUES))
        raise ValueError(f"Column '{TARGET_COLUMN}' must contain only: {allowed}")


def validate_churn_data(df: pd.DataFrame) -> None:
    validate_columns(df)
    validate_numeric_columns(df)
    validate_target(df)

