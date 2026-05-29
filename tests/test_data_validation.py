import pandas as pd
import pytest

from data_validation import validate_churn_data


def valid_row() -> dict[str, object]:
    return {
        "customer_id": "C001",
        "tenure_months": 12,
        "monthly_charges": 72.5,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "support_calls": 2,
        "payment_method": "Electronic check",
        "churn": "Yes",
    }


def test_validate_churn_data_accepts_valid_data() -> None:
    df = pd.DataFrame([valid_row()])

    validate_churn_data(df)


def test_validate_churn_data_rejects_missing_column() -> None:
    row = valid_row()
    row.pop("monthly_charges")
    df = pd.DataFrame([row])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_churn_data(df)


def test_validate_churn_data_rejects_invalid_target() -> None:
    row = valid_row()
    row["churn"] = "Maybe"
    df = pd.DataFrame([row])

    with pytest.raises(ValueError, match="must contain only"):
        validate_churn_data(df)


def test_validate_churn_data_rejects_negative_numeric_value() -> None:
    row = valid_row()
    row["support_calls"] = -1
    df = pd.DataFrame([row])

    with pytest.raises(ValueError, match="must not contain negative values"):
        validate_churn_data(df)

