import gradio as gr
import numpy as np
import joblib
import os

MODEL_PATH = "./model/decisiontree_classifier_baseline.pkl"

def predict(monthly_fee, customer_age, support_calls):
    # Ensure correct shape
    X = np.array([[monthly_fee, customer_age, support_calls]])
    # Load the model (load once if possible, not every call)
    if not hasattr(predict, "model"):
        if not os.path.exists(MODEL_PATH):
            return "Model file not found!"
        predict.model = joblib.load(MODEL_PATH)
    pred = predict.model.predict(X)
    return f"Predicted Class: {int(pred[0])}"

# Gradio Inputs/Outputs
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Monthly Fee", value=0.0),
        gr.Number(label="Customer Age", value=18),
        gr.Number(label="Support Calls", value=0),
    ],
    outputs="text",
    title="Customer Churn Prediction",
    description="Predict customer churn using a Decision Tree Classifier."
)

if __name__ == "__main__":
    demo.launch()