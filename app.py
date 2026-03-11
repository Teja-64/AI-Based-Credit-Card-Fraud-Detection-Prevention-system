import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("AI Credit Card Fraud Detection System")

st.write("Enter transaction details to check if it is Fraud or Legitimate")

# Inputs
amount = st.number_input("Enter Transaction Amount")
v1 = st.number_input("Enter V1 Value")

# Prediction button
if st.button("Check Transaction"):

    input_data = np.array([[amount, v1]])

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("Legitimate Transaction")
    else:
        st.error("Fraudulent Transaction")