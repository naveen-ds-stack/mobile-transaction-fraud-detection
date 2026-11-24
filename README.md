from pathlib import Path

readme = r"""# Mobile Transaction Fraud Detection

An end-to-end machine learning project that analyses mobile financial transactions and classifies them as legitimate or potentially fraudulent. The project compares Logistic Regression, Random Forest and XGBoost, then uses the selected XGBoost model in a Streamlit demonstration application. A Power BI dashboard is included for exploratory and business-focused reporting.

## Project Overview

Fraud detection is an imbalanced binary classification problem: fraudulent transactions are much less common than legitimate transactions. This project explores transaction patterns, engineers account-balance features, compares classification models and presents predictions through a simple Streamlit interface.

**Target variable:** `isFraud`

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

## Dataset

The analysis uses `Fraud_Analysis_Dataset.csv`.

| Item | Details |
|---|---|
| Records | 11,142 transactions |
| Original columns | 10 |
| Legitimate transactions | 10,000 (89.75%) |
| Fraudulent transactions | 1,142 (10.25%) |
| Missing values | No missing values found in the inspected columns |
| Target | `isFraud` |

The dataset includes transaction type and amount, sender and receiver balances before and after the transaction, transaction step, account identifiers and the fraud label.

### Exploratory Data Analysis

Key findings recorded in the notebook:

- The dataset is imbalanced, with 10.25% fraudulent transactions.
- `PAYMENT` is the most frequent transaction type (5,510 records).
- In this dataset, fraud was observed in `TRANSFER` and `CASH_OUT` transactions.
- The observed fraud rates were 38.52% for `TRANSFER` and 30.89% for `CASH_OUT`; the other three transaction types had no fraud records in this sample.
- These patterns describe this dataset and should not be treated as universal fraud rules.

## Data Preparation

The notebook applies the following preparation steps:

1. Checks data types, dataset dimensions and missing values.
2. Retains high-value transaction and balance observations rather than removing them as outliers.
3. Removes `nameOrig` and `nameDest`, which are account identifier fields.
4. Removes `step` from the model features because its relationship with the target appeared highly dataset-specific and could act as a shortcut signal.
5. Creates balance-based features.
6. Applies one-hot encoding to the `type` column with `drop_first=True`.
7. Separates the predictors (`X`) and target (`y`).
8. Uses a stratified 70:30 train-test split with `random_state=42`.

### Engineered Features

| Feature | Calculation | Purpose |
|---|---|---|
| `balance_diff_orig` | `oldbalanceOrg - newbalanceOrig` | Measures the sender balance decrease |
| `balance_diff_dest` | `newbalanceDest - oldbalanceDest` | Measures the receiver balance increase |
| `amount_balance_mismatch` | `amount - balance_diff_orig` | Measures the difference between transaction amount and sender balance decrease |

**Feature availability limitation:** The engineered balance features use post-transaction balances. Their availability must be confirmed before using this model for pre-transaction authorization or real-time blocking.

## Model Development

Three classifiers were trained and evaluated:

- **Logistic Regression** – baseline classifier. Numeric features were standardized using a scaler fitted on the training data.
- **Random Forest** – tree ensemble used to model non-linear relationships.
- **XGBoost** – gradient-boosted tree model selected for the Streamlit demonstration.

The final XGBoost model configuration in the deployment training code:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `eval_metric` | `logloss` |
| `random_state` | 42 |
| `n_jobs` | -1 |

Class weighting was used in the Logistic Regression baseline. The final XGBoost training code uses the configuration above.

## Model Evaluation

The notebook compares Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC (Average Precision), training time and prediction time.

### Random Forest vs. XGBoost

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 0.998504 | **0.998803** |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.985423 | **0.988338** |
| F1-score | 0.992658 | **0.994135** |
| ROC-AUC | 0.995214 | **0.999089** |
| PR-AUC (Average Precision) | **0.999787** | 0.999773 |
| Training time (seconds) | 1.212255 | **0.590202** |
| Prediction time (seconds) | 0.036687 | **0.017281** |

XGBoost was selected based on its slightly higher accuracy, recall, F1-score and ROC-AUC, along with lower recorded training and prediction times. Random Forest had a marginally higher PR-AUC. The PR-AUC difference is very small, so it should not be interpreted as a meaningful practical advantage without further validation.

### XGBoost Confusion Matrix

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 3,000 | 0 |
| Actual fraud | 4 | 339 |

On this held-out test set, XGBoost achieved:

- Accuracy: approximately **99.88%**
- Fraud precision: **100.00%**
- Fraud recall: approximately **98.83%**
- Fraud F1-score: approximately **99.41%**

These are test-set results, not a guarantee of performance on unseen production data.

## XGBoost Feature Importance

The notebook uses the XGBoost model's built-in `feature_importances_` values to inspect which input features contributed most to the fitted model. In the recorded feature-importance analysis, `balance_diff_orig` ranked highest, followed by `newbalanceOrig` and `amount_balance_mismatch`.

Built-in feature importance is model-specific and does not establish causation. It should be interpreted as a diagnostic view of this fitted model.

## Streamlit Application

A Streamlit interface was created to demonstrate the saved XGBoost model. Users enter:

- Transaction amount
- Sender's old and new balances
- Receiver's old and new balances
- Transaction type

The application calculates the engineered balance features, aligns the input columns with the saved model feature list, and displays a fraud probability and predicted class.

The application was tested locally using Streamlit. Public Streamlit Community Cloud deployment should be marked as pending until the deployed URL is available.

## Power BI Dashboard

The repository includes a Power BI dashboard file for visual exploration and reporting:

`powerbi/Fraud_Detection_Dashboard.pbix`

Open the `.pbix` file using Microsoft Power BI Desktop.

## Repository Structure

```text
mobile-transaction-fraud-detection/
│
├── app/
│   ├── app.py
│   ├── fraud_detection_xgboost.joblib
│   ├── model_features.joblib
│   └── requirements.txt
│
├──  Fraud_Analysis_Dataset.csv
│
├── Financial Fraud Detection in Digital Transactions.ipynb  
│
├── Fraud_Detection_Dashboard.pbix
│   
└── README.md
```

## Run the Streamlit Application Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/naveen-ds-stack/mobile-transaction-fraud-detection.git
   cd mobile-transaction-fraud-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r app/requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app/app.py
   ```

Ensure the model files are in the same `app/` directory as `app.py`, and that the code loads them using paths relative to that directory.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Power BI
- Joblib
- Jupyter Notebook

## Limitations and Further Work

- The evaluation uses a stratified random train-test split; time-based validation on later transactions is still needed.
- The dataset is a limited sample and may not represent changing real-world fraud patterns.
- The balance-based engineered features use post-transaction values. A pre-transaction fraud system would need features available at authorization time.
- The decision threshold should be selected based on the business cost of missed fraud and false alerts.
- Further validation, monitoring, threshold analysis and testing on independent data are required before production use.

## Author

**Naveen Prasath R**

GitHub: [naveen-ds-stack](https://github.com/naveen-ds-stack)
"""

out = Path("/mnt/data/README.md")
out.write_text(readme, encoding="utf-8")
print(f"Created {out} ({out.stat().st_size} bytes)")
