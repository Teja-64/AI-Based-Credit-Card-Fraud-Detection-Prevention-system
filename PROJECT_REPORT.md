# Credit Card Fraud Detection System Using Random Forest

## 1. Title

Credit Card Fraud Detection System Using Random Forest Algorithm

## 2. Abstract

Credit card fraud is a major issue in the digital payment system. As online transactions increase, fraudsters try to perform unauthorized payments using stolen card details or suspicious transaction patterns. This project presents a mini machine learning system that predicts whether a credit card transaction is safe or fraudulent.

The proposed system uses the Random Forest Classifier algorithm. A sample transaction dataset is generated inside the project for academic demonstration. The model is trained using transaction details such as amount, transaction hour, number of previous transactions, failed attempts, distance from home, account age, card availability, international transaction status, transaction type, and merchant category.

The final output is displayed using a Streamlit web application. The user manually enters transaction details and the system predicts whether the transaction is safe or fraudulent. It also displays the fraud probability score.

## 3. Introduction

Credit card usage has become common in online shopping, banking, travel booking, and point-of-sale payments. With this growth, financial fraud has also increased. Traditional rule-based systems may fail because fraud patterns change over time. Machine learning can learn patterns from data and help identify suspicious transactions.

This project is designed as a final-year mini project. It demonstrates the complete workflow of a machine learning application:

- Dataset creation
- Data preprocessing
- Model training
- Model evaluation
- Model saving
- Streamlit-based user interface
- Manual transaction prediction

## 4. Problem Statement

The aim of this project is to develop a simple credit card fraud detection system that can classify a transaction as safe or fraudulent based on manually entered transaction details.

The main problem is:

How can a machine learning model identify suspicious credit card transactions and display the result to the user through a simple web interface?

## 5. Objectives

- To study credit card fraud detection using machine learning.
- To build a Random Forest based fraud detection model.
- To create a generated transaction dataset for demonstration.
- To train and evaluate the model.
- To create a Streamlit application for manual transaction input.
- To display fraud or safe transaction output clearly.
- To make the project simple, understandable, and suitable for academic submission.

## 6. Scope of the Project

This project focuses on academic demonstration of fraud detection. The system predicts fraud using transaction features entered by the user. It is not connected to a real banking network or payment gateway.

The project can be extended in the future by using real bank transaction data, database storage, user login, SMS alerts, email alerts, and live transaction monitoring.

## 7. Existing System

In many simple systems, fraud is detected using fixed rules such as:

- Block transaction if the amount is above a limit.
- Block transaction if it happens at midnight.
- Block transaction if multiple failed attempts occur.

Such systems are easy to implement but not very flexible. They may generate false alerts or miss new fraud patterns.

## 8. Proposed System

The proposed system uses the Random Forest Classifier algorithm to classify transactions. Instead of depending only on fixed rules, the model learns from patterns in the dataset.

The user opens the Streamlit application, enters transaction details, and clicks the prediction button. The trained model processes the input and displays:

- Safe Transaction
- Fraud Transaction Detected
- Fraud probability percentage

## 9. Hardware and Software Requirements

### Hardware Requirements

| Component | Minimum Requirement |
|---|---|
| Processor | Intel i3 or above |
| RAM | 4 GB or above |
| Storage | 500 MB free space |
| Display | Standard monitor |

### Software Requirements

| Software | Purpose |
|---|---|
| Python 3.10+ | Programming language |
| Streamlit | Web application interface |
| scikit-learn | Machine learning model |
| pandas | Data handling |
| numpy | Numerical operations |
| joblib | Model saving/loading |

## 10. Technologies Used

- Python
- Streamlit
- Random Forest Classifier
- pandas
- numpy
- scikit-learn
- joblib

## 11. Dataset Description

The dataset is generated inside `train_model.py`. This makes the project self-contained and easy to run without downloading external datasets.

The generated dataset contains transaction details and a target column named `is_fraud`.

### Input Features

| Feature | Description |
|---|---|
| amount | Transaction amount |
| hour | Hour of transaction from 0 to 23 |
| previous_transactions_24h | Number of transactions in the last 24 hours |
| failed_attempts | Failed login or payment attempts |
| distance_from_home_km | Distance from user's usual location |
| account_age_days | Age of the account in days |
| card_present | Whether the physical card was present |
| international_transaction | Whether the transaction is international |
| transaction_type | POS, Online, ATM, or Transfer |
| merchant_category | Type of merchant |

### Output Feature

| Feature | Description |
|---|---|
| is_fraud | 0 means safe, 1 means fraud |

## 12. Algorithm Used: Random Forest

Random Forest is a supervised machine learning algorithm used for classification and regression. It creates multiple decision trees and combines their results to make a final prediction.

In this project, Random Forest is used for binary classification:

- Class 0: Safe Transaction
- Class 1: Fraud Transaction

### Advantages of Random Forest

- Handles both numerical and categorical data after preprocessing.
- Reduces overfitting compared to a single decision tree.
- Gives good accuracy for classification problems.
- Works well for tabular datasets.
- Can model complex fraud patterns.

## 13. System Architecture

The system follows this flow:

1. Generate transaction dataset.
2. Split dataset into training and testing data.
3. Preprocess numerical and categorical features.
4. Train Random Forest Classifier.
5. Save trained model as `random_forest_fraud_model.pkl`.
6. Load model in Streamlit application.
7. Accept manual input from user.
8. Predict fraud probability.
9. Display final result.

## 14. Module Description

### 14.1 Data Generation Module

The data generation module creates sample transaction records using realistic fraud-related conditions such as high amount, late-night transaction, multiple failed attempts, international transaction, and long distance from home.

File: `train_model.py`

### 14.2 Preprocessing Module

Numerical fields are standardized using `StandardScaler`. Categorical fields are converted using `OneHotEncoder`.

### 14.3 Model Training Module

The Random Forest Classifier is trained using the preprocessed dataset. The trained pipeline includes both preprocessing and model logic.

### 14.4 Model Saving Module

The final trained model is saved using joblib.

Output file: `random_forest_fraud_model.pkl`

### 14.5 Streamlit User Interface Module

The Streamlit application accepts transaction details from the user and displays prediction output.

File: `app.py`

## 15. Implementation Details

The project contains two main Python files:

### train_model.py

This file:

- Generates the dataset.
- Splits data into train and test sets.
- Applies preprocessing.
- Trains the Random Forest model.
- Evaluates the model.
- Saves model and metrics.

### app.py

This file:

- Loads the trained model.
- Shows manual input fields.
- Accepts transaction details.
- Predicts fraud probability.
- Displays whether the transaction is safe or fraudulent.

## 16. Testing

The project was tested using example safe and fraud transactions.

### Test Case 1: Safe Transaction

| Field | Value |
|---|---|
| Amount | 1200 |
| Hour | 11 |
| Previous transactions | 1 |
| Failed attempts | 0 |
| Distance from home | 5 km |
| Account age | 1300 days |
| Card present | Yes |
| International transaction | No |
| Transaction type | POS |
| Merchant category | Grocery |

Expected output: Safe Transaction

### Test Case 2: Fraud Transaction

| Field | Value |
|---|---|
| Amount | 42000 |
| Hour | 2 |
| Previous transactions | 15 |
| Failed attempts | 4 |
| Distance from home | 230 km |
| Account age | 45 days |
| Card present | No |
| International transaction | Yes |
| Transaction type | Online |
| Merchant category | Electronics |

Expected output: Fraud Transaction Detected

## 17. Results

The Random Forest model was trained successfully. The generated metrics are stored in `model_metrics.json`.

During testing:

- Safe example was predicted as safe.
- Fraud example was predicted as fraud.
- Streamlit application started successfully.
- Manual input form worked correctly.

## 18. Advantages

- Simple and easy to understand.
- Uses Random Forest algorithm.
- Accepts manual input from the user.
- Shows fraud probability.
- Does not require external dataset download.
- Suitable for final-year mini project demonstration.

## 19. Limitations

- The dataset is generated for academic use.
- It is not connected to real banking systems.
- It does not process live payment transactions.
- Prediction accuracy may vary if used with real-world data.

## 20. Future Enhancements

- Use real credit card transaction dataset.
- Add database support.
- Add user login and admin dashboard.
- Add transaction history.
- Add email or SMS fraud alerts.
- Deploy the app online.
- Add more algorithms for comparison.
- Add charts and visual analytics.

## 21. Conclusion

This project successfully implements a credit card fraud detection system using the Random Forest algorithm. The model predicts whether a transaction is safe or fraudulent based on manually entered transaction details. The Streamlit interface makes the system easy to use and suitable for academic demonstration.

The project covers the complete machine learning workflow from data generation to model deployment through a simple web application.

## 22. References

- scikit-learn documentation
- Streamlit documentation
- pandas documentation
- Machine learning classification concepts
