# Credit Card Fraud Detection System

Final year mini project using Random Forest algorithm and Streamlit.

## Project Objective

The objective of this project is to predict whether a credit card transaction is safe or fraudulent. The user enters transaction details manually in the Streamlit app, and the trained Random Forest model displays the prediction result.

## Algorithm Used

Random Forest Classifier

## Features

- Manual transaction input form
- Fraud or safe transaction prediction
- Fraud probability score
- Streamlit web interface
- Model training script
- Self-contained sample dataset generation

## Project Files

```text
app.py                         Streamlit application
train_model.py                 Random Forest model training script
random_forest_fraud_model.pkl  Trained model
fraud_transactions.csv         Generated training dataset
model_metrics.json             Model accuracy and report
requirements.txt               Required Python packages
README.md                      Project documentation
PROJECT_REPORT.md              Complete project report in Markdown
USER_GUIDE.md                  User and setup guide
TESTING_DOCUMENT.md            Testing documentation
Credit_Card_Fraud_Detection_Project_Report.docx
                              Submission-ready Word project report
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Input Fields

- Transaction amount
- Transaction hour
- Previous transactions in 24 hours
- Failed attempts
- Distance from home
- Account age
- Card present
- International transaction
- Transaction type
- Merchant category

## Output

The app displays:

- Safe Transaction
- Fraud Transaction Detected
- Fraud probability percentage

## Documentation

This project includes complete documentation for academic submission:

- `Credit_Card_Fraud_Detection_Project_Report.docx`
- `PROJECT_REPORT.md`
- `USER_GUIDE.md`
- `TESTING_DOCUMENT.md`

## Note

This project is created as an academic mini project. The dataset is generated inside `train_model.py` for easy demonstration without needing an external dataset.
