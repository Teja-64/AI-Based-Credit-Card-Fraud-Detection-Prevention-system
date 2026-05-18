import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def create_dataset(rows=6000):
    rng = np.random.default_rng(RANDOM_STATE)

    amount = np.round(rng.gamma(shape=2.2, scale=1800, size=rows) + 50, 2)
    hour = rng.integers(0, 24, size=rows)
    previous_transactions_24h = rng.poisson(lam=3, size=rows)
    failed_attempts = rng.poisson(lam=0.35, size=rows)
    distance_from_home_km = np.round(rng.gamma(shape=1.8, scale=18, size=rows), 2)
    account_age_days = rng.integers(10, 3000, size=rows)
    card_present = rng.choice(["Yes", "No"], size=rows, p=[0.72, 0.28])
    international_transaction = rng.choice(["No", "Yes"], size=rows, p=[0.88, 0.12])
    transaction_type = rng.choice(
        ["POS", "Online", "ATM", "Transfer"],
        size=rows,
        p=[0.48, 0.32, 0.12, 0.08],
    )
    merchant_category = rng.choice(
        ["Grocery", "Fuel", "Electronics", "Travel", "Jewellery", "Restaurant", "Other"],
        size=rows,
        p=[0.22, 0.15, 0.16, 0.10, 0.07, 0.18, 0.12],
    )

    risk_score = np.zeros(rows)
    risk_score += amount > 12000
    risk_score += amount > 25000
    risk_score += np.isin(hour, [0, 1, 2, 3, 4])
    risk_score += previous_transactions_24h >= 8
    risk_score += failed_attempts >= 2
    risk_score += distance_from_home_km > 75
    risk_score += account_age_days < 90
    risk_score += card_present == "No"
    risk_score += international_transaction == "Yes"
    risk_score += np.isin(transaction_type, ["Online", "Transfer"])
    risk_score += np.isin(merchant_category, ["Electronics", "Travel", "Jewellery"])

    probability = 1 / (1 + np.exp(-(risk_score - 4.2)))
    fraud = rng.binomial(1, probability)

    # Add strong fraud-like examples so the model learns obvious suspicious cases.
    fraud_indices = rng.choice(rows, size=500, replace=False)
    amount[fraud_indices] = np.round(rng.uniform(18000, 85000, size=len(fraud_indices)), 2)
    hour[fraud_indices] = rng.choice([0, 1, 2, 3, 4, 23], size=len(fraud_indices))
    previous_transactions_24h[fraud_indices] = rng.integers(8, 24, size=len(fraud_indices))
    failed_attempts[fraud_indices] = rng.integers(2, 8, size=len(fraud_indices))
    distance_from_home_km[fraud_indices] = np.round(rng.uniform(80, 500, size=len(fraud_indices)), 2)
    account_age_days[fraud_indices] = rng.integers(10, 180, size=len(fraud_indices))
    card_present[fraud_indices] = "No"
    international_transaction[fraud_indices] = "Yes"
    transaction_type[fraud_indices] = rng.choice(["Online", "Transfer"], size=len(fraud_indices))
    merchant_category[fraud_indices] = rng.choice(
        ["Electronics", "Travel", "Jewellery"],
        size=len(fraud_indices),
    )
    fraud[fraud_indices] = 1

    data = pd.DataFrame(
        {
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
            "is_fraud": fraud,
        }
    )
    return data


def main():
    data = create_dataset()
    data.to_csv(PROJECT_DIR / "fraud_transactions.csv", index=False)

    target = "is_fraud"
    numeric_features = [
        "amount",
        "hour",
        "previous_transactions_24h",
        "failed_attempts",
        "distance_from_home_km",
        "account_age_days",
    ]
    categorical_features = [
        "card_present",
        "international_transaction",
        "transaction_type",
        "merchant_category",
    ]

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "algorithm": "Random Forest Classifier",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "features": list(X.columns),
    }

    joblib.dump(pipeline, PROJECT_DIR / "random_forest_fraud_model.pkl")
    with open(PROJECT_DIR / "model_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print("Training completed.")
    print(f"Accuracy: {metrics['accuracy']}")
    print("Saved random_forest_fraud_model.pkl")
    print("Saved fraud_transactions.csv")
    print("Saved model_metrics.json")


if __name__ == "__main__":
    main()
