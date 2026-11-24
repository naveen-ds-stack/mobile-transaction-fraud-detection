import streamlit as st
import pandas as pd
import joblib

# 1. Load trained XGBoost model and feature names
model = joblib.load("fraud_detection_xgboost.joblib")
features = joblib.load("model_features.joblib")

st.set_page_config(
    page_title="Fraud Detection App",
    layout="centered"
)

st.title("💳 - Mobile Transaction Fraud Detection - Naveen")
st.write("Enter transaction details to predict fraud risk.")

# 2. User inputs (step removed)
amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

txn_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

# 3. Feature Engineering
balance_diff_orig = oldbalanceOrg - newbalanceOrig
balance_diff_dest = newbalanceDest - oldbalanceDest

amount_balance_mismatch = amount - balance_diff_orig

# 4. Create input dictionary
input_data = {
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'balance_diff_orig': balance_diff_orig,
    'balance_diff_dest': balance_diff_dest,
    'amount_balance_mismatch': amount_balance_mismatch
}

# 5. One-hot encode transaction type
for col in [
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]:
    input_data[col] = int(txn_type == col.replace("type_", ""))

# 6. Create DataFrame and align with training features
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=features, fill_value=0)

# 7. Prediction
if st.button("Predict Fraud"):

    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"Fraud Probability: **{prob:.2%}**")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")