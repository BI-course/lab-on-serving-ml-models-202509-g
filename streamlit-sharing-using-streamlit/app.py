import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("./model/decisiontree_classifier_baseline.pkl")

# Streamlit page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn using a Decision Tree Classifier.")

# Input form
with st.form("prediction_form"):
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, step=1.0)
    customer_age = st.number_input("Customer Age", min_value=0, step=1)
    support_calls = st.number_input("Support Calls", min_value=0, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    X = np.array([[monthly_fee, customer_age, support_calls]])
    prediction = model.predict(X)
    st.success(f"### Predicted Class: {int(prediction[0])}")
