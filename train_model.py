# Import libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("creditcard_2023.csv")

print("Dataset Loaded")

# Separate fraud and legit
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Under sampling
legit_sample = legit.sample(n=len(fraud))

balanced_data = pd.concat([legit_sample, fraud], axis=0)

print("Balanced Dataset")
print(balanced_data["Class"].value_counts())

# Features and target
X = balanced_data[["Amount","V1"]]
y = balanced_data["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))

# Save model
pickle.dump(model, open("model.pkl","wb"))

print("Model saved successfully")