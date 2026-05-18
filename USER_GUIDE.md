# User Guide

## Project Name

Credit Card Fraud Detection System Using Random Forest

## How to Open the Project

1. Extract the project zip file.
2. Open the extracted folder.
3. Install the required Python packages.
4. Train the model if needed.
5. Run the Streamlit application.

## Installation Steps

Open a terminal or command prompt inside the project folder and run:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train_model.py
```

This creates:

- `fraud_transactions.csv`
- `random_forest_fraud_model.pkl`
- `model_metrics.json`

## Run the Application

Run:

```bash
streamlit run app.py
```

The Streamlit app will open in the browser.

## How to Use the App

1. Enter transaction amount.
2. Enter transaction hour.
3. Enter previous transactions in 24 hours.
4. Enter failed attempts.
5. Enter distance from home.
6. Enter account age.
7. Select whether card is present.
8. Select whether transaction is international.
9. Select transaction type.
10. Select merchant category.
11. Click `Predict Transaction`.

## Output

The app displays:

- `Safe Transaction`
- `Fraud Transaction Detected`
- Fraud probability percentage

## Example Safe Input

| Field | Value |
|---|---|
| Amount | 1200 |
| Hour | 11 |
| Previous transactions | 1 |
| Failed attempts | 0 |
| Distance from home | 5 |
| Account age | 1300 |
| Card present | Yes |
| International transaction | No |
| Transaction type | POS |
| Merchant category | Grocery |

## Example Fraud Input

| Field | Value |
|---|---|
| Amount | 42000 |
| Hour | 2 |
| Previous transactions | 15 |
| Failed attempts | 4 |
| Distance from home | 230 |
| Account age | 45 |
| Card present | No |
| International transaction | Yes |
| Transaction type | Online |
| Merchant category | Electronics |
