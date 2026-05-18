from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "random_forest_fraud_model.pkl"


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="card",
    layout="centered",
)


st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }
        .result-safe {
            padding: 18px;
            border-radius: 8px;
            background: #dcfce7;
            color: #166534;
            border: 1px solid #16a34a;
            font-size: 22px;
            font-weight: 700;
            text-align: center;
        }
        .result-fraud {
            padding: 18px;
            border-radius: 8px;
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #dc2626;
            font-size: 22px;
            font-weight: 700;
            text-align: center;
        }
        .small-note {
            color: #475569;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_frame(values):
    return pd.DataFrame([values])


if not MODEL_PATH.exists():
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()


model = load_model()

st.title("Credit Card Fraud Detection System")
st.write("Mini project using Random Forest algorithm and Streamlit.")

st.subheader("Enter Transaction Details")

demo_type = st.radio(
    "Quick fill option",
    ["Manual entry", "Example safe transaction", "Example fraud transaction"],
    horizontal=True,
)

defaults = {
    "amount": 2500.0,
    "hour": 14,
    "previous_transactions_24h": 2,
    "failed_attempts": 0,
    "distance_from_home_km": 8.0,
    "account_age_days": 950,
    "card_present": "Yes",
    "international_transaction": "No",
    "transaction_type": "POS",
    "merchant_category": "Grocery",
}

if demo_type == "Example fraud transaction":
    defaults.update(
        {
            "amount": 42000.0,
            "hour": 2,
            "previous_transactions_24h": 15,
            "failed_attempts": 4,
            "distance_from_home_km": 230.0,
            "account_age_days": 45,
            "card_present": "No",
            "international_transaction": "Yes",
            "transaction_type": "Online",
            "merchant_category": "Electronics",
        }
    )
elif demo_type == "Example safe transaction":
    defaults.update(
        {
            "amount": 1200.0,
            "hour": 11,
            "previous_transactions_24h": 1,
            "failed_attempts": 0,
            "distance_from_home_km": 5.0,
            "account_age_days": 1300,
            "card_present": "Yes",
            "international_transaction": "No",
            "transaction_type": "POS",
            "merchant_category": "Grocery",
        }
    )

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0, value=defaults["amount"], step=100.0)
    hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, value=defaults["hour"])
    previous_transactions_24h = st.number_input(
        "Previous Transactions in 24 Hours",
        min_value=0,
        value=defaults["previous_transactions_24h"],
    )
    failed_attempts = st.number_input("Failed Login/Payment Attempts", min_value=0, value=defaults["failed_attempts"])
    distance_from_home_km = st.number_input(
        "Distance From Home (km)",
        min_value=0.0,
        value=defaults["distance_from_home_km"],
        step=1.0,
    )

with col2:
    account_age_days = st.number_input("Account Age (days)", min_value=1, value=defaults["account_age_days"])
    card_present = st.selectbox("Card Present", ["Yes", "No"], index=["Yes", "No"].index(defaults["card_present"]))
    international_transaction = st.selectbox(
        "International Transaction",
        ["No", "Yes"],
        index=["No", "Yes"].index(defaults["international_transaction"]),
    )
    transaction_type = st.selectbox(
        "Transaction Type",
        ["POS", "Online", "ATM", "Transfer"],
        index=["POS", "Online", "ATM", "Transfer"].index(defaults["transaction_type"]),
    )
    merchant_category = st.selectbox(
        "Merchant Category",
        ["Grocery", "Fuel", "Electronics", "Travel", "Jewellery", "Restaurant", "Other"],
        index=["Grocery", "Fuel", "Electronics", "Travel", "Jewellery", "Restaurant", "Other"].index(
            defaults["merchant_category"]
        ),
    )

threshold = st.slider("Fraud Detection Threshold", 0.10, 0.90, 0.50, 0.05)

input_values = {
    "amount": amount,
    "hour": hour,
    "previous_transactions_24h": previous_transactions_24h,
    "failed_attempts": failed_attempts,
    "distance_from_home_km": distance_from_home_km,
    "account_age_days": account_age_days,
    "card_present": card_present,
    "international_transaction": international_transaction,
    "transaction_type": transaction_type,
    "merchant_category": merchant_category,
}

if st.button("Predict Transaction"):
    input_data = build_input_frame(input_values)
    fraud_probability = float(model.predict_proba(input_data)[0][1])
    prediction = 1 if fraud_probability >= threshold else 0

    st.subheader("Prediction Result")
    if prediction == 1:
        st.markdown('<div class="result-fraud">Fraud Transaction Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-safe">Safe Transaction</div>', unsafe_allow_html=True)

    st.write(f"Fraud Probability: **{fraud_probability * 100:.2f}%**")
    st.progress(fraud_probability)

    st.subheader("Entered Transaction")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    '<p class="small-note">Algorithm used: Random Forest Classifier. This mini project is for academic demonstration.</p>',
    unsafe_allow_html=True,
)
