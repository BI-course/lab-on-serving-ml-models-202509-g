import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("./model/decisiontree_classifier_baseline.pkl")

# Define prediction function
def predict(monthly_fee, customer_age, support_calls):
    X = np.array([[monthly_fee, customer_age, support_calls]])
    prediction = model.predict(X)
    return int(prediction[0])

# Build Gradio interface
demo = gr.Interface(
    title="Customer Churn Prediction",
    description="Predict whether a customer will churn using a Decision Tree Classifier.",
    fn=predict,
    inputs=[
        gr.Number(label="Monthly Fee"),
        gr.Number(label="Customer Age"),
        gr.Number(label="Support Calls")
    ],
    outputs=gr.Number(label="Predicted Class")
)

if __name__ == "__main__":
    demo.launch()
