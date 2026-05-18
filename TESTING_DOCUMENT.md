# Testing Document

## Project

Credit Card Fraud Detection System Using Random Forest

## Testing Objective

To verify that the Streamlit application accepts manual transaction input and correctly displays whether the transaction is safe or fraudulent.

## Test Environment

| Item | Value |
|---|---|
| Language | Python |
| Framework | Streamlit |
| Algorithm | Random Forest Classifier |
| Dataset | Generated dataset |
| Model file | random_forest_fraud_model.pkl |

## Test Case 1: Application Launch

| Test Step | Expected Result | Status |
|---|---|---|
| Run `streamlit run app.py` | Application opens successfully | Pass |

## Test Case 2: Safe Transaction Prediction

| Field | Value |
|---|---|
| Amount | 1200 |
| Hour | 11 |
| Previous transactions in 24 hours | 1 |
| Failed attempts | 0 |
| Distance from home | 5 |
| Account age | 1300 |
| Card present | Yes |
| International transaction | No |
| Transaction type | POS |
| Merchant category | Grocery |

Expected Result: Safe Transaction

Actual Result: Safe Transaction

Status: Pass

## Test Case 3: Fraud Transaction Prediction

| Field | Value |
|---|---|
| Amount | 42000 |
| Hour | 2 |
| Previous transactions in 24 hours | 15 |
| Failed attempts | 4 |
| Distance from home | 230 |
| Account age | 45 |
| Card present | No |
| International transaction | Yes |
| Transaction type | Online |
| Merchant category | Electronics |

Expected Result: Fraud Transaction Detected

Actual Result: Fraud Transaction Detected

Status: Pass

## Test Case 4: Model Training

| Test Step | Expected Result | Status |
|---|---|---|
| Run `python train_model.py` | Model and dataset files are generated | Pass |

## Summary

The application passed the basic functional tests. It accepts manual transaction details, predicts the transaction class, and displays the fraud probability.
