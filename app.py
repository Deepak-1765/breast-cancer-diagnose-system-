import gradio as gr
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Load Model & Scaler
# ===============================

model = load_model("breast_cancer_model.h5")
scaler = joblib.load("breast_cancer_scaler.pkl")

# ===============================
# Prediction Function
# ===============================

def predict(
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,
):

    data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        result = "🔴 Malignant (Cancer Detected)"
        confidence = probability * 100
    else:
        result = "🟢 Benign (No Cancer)"
        confidence = (1 - probability) * 100

    return result, f"{confidence:.2f}%"

# ===============================
# Interface
# ===============================

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Radius Mean"),
        gr.Number(label="Texture Mean"),
        gr.Number(label="Perimeter Mean"),
        gr.Number(label="Area Mean"),
        gr.Number(label="Smoothness Mean"),
        gr.Number(label="Compactness Mean"),
        gr.Number(label="Concavity Mean"),
        gr.Number(label="Concave Points Mean"),
        gr.Number(label="Symmetry Mean"),
        gr.Number(label="Fractal Dimension Mean"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
    ],
    title="Breast Cancer Detection",
    description="Predict whether the tumor is Benign or Malignant using a Deep Learning model.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
