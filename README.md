# Customer Churn ML

A small, complete baseline project for predicting customer churn with Python, pandas, and scikit-learn.

The goal is not to build the most accurate churn model. The goal is to show a clean end-to-end machine learning workflow: load data, validate it, preprocess mixed feature types, train a model, evaluate metrics, and save predictions.

## File Structure

```text
ml-customer-churn-baseline/
  README.md
  requirements.txt
  data/
    sample_churn.csv
  outputs/
    sample_metrics.json
    sample_predictions.csv
  src/
    data_validation.py
    predict.py
    train.py
  tests/
    conftest.py
    test_data_validation.py
    test_training_pipeline.py
```

The `models/` directory is created when training runs. The generated `.joblib` model file is ignored by Git.

## What This Demonstrates

For recruiters, this project demonstrates:

- Practical pandas data loading and validation
- Numeric and categorical preprocessing with `ColumnTransformer`
- A simple scikit-learn training pipeline
- Classification metrics such as accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
- Reproducible train/test splitting
- Saving sample metrics and predictions
- Basic automated tests for data validation and model training behavior

## Setup

Run commands from the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Train the Model

```bash
python src/train.py
```

This reads:

```text
data/sample_churn.csv
```

And writes:

```text
outputs/sample_metrics.json
outputs/sample_predictions.csv
models/churn_model.joblib
```

## Run Prediction

After training:

```bash
python src/predict.py --input data/sample_churn.csv --output outputs/new_predictions.csv
```

The prediction script accepts any CSV with the required feature columns:

```text
tenure_months, monthly_charges, contract_type, internet_service, support_calls, payment_method
```

If a `customer_id` column exists, it is included in the output.

## Sample Input

From `data/sample_churn.csv`:

```csv
customer_id,tenure_months,monthly_charges,contract_type,internet_service,support_calls,payment_method,churn
C001,2,86.5,Month-to-month,Fiber optic,4,Electronic check,Yes
C002,30,55.0,One year,DSL,1,Credit card,No
C003,5,92.0,Month-to-month,Fiber optic,5,Electronic check,Yes
```

## Sample Output

From `outputs/sample_metrics.json`:

```json
{
  "accuracy": 0.9167,
  "precision": 0.8333,
  "recall": 1.0,
  "f1": 0.9091,
  "roc_auc": 1.0,
  "confusion_matrix": [[6, 1], [0, 5]]
}
```

Prediction output format:

```csv
customer_id,actual_churn,predicted_churn,churn_probability
C040,No,Yes,0.6344
C037,No,No,0.069
```

Exact values can change if the data, model, or random seed changes.

## Run Tests

```bash
python -m pytest
```

## Limitations

- The dataset is small and synthetic, so results are not production-quality.
- Logistic regression is used as a readable baseline, not as the only possible model.
- No hyperparameter tuning is included.
- No fairness, drift, or calibration checks are included.
- The target label is simplified to `Yes` and `No`.

## Future Improvements

- Add cross-validation and model comparison.
- Add feature importance or coefficient reporting.
- Add better dataset documentation.
- Add probability calibration.
- Add a simple experiment tracking table.
