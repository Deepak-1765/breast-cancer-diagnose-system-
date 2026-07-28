# app.py
import os
import joblib
import traceback
import numpy as np
import gradio as gr
import tensorflow as tf

try:
    scaler = joblib.load("breast_cancer_scaler.pkl")
    deployed_nn = tf.keras.models.load_model("breast_cancer_model.h5")
except Exception as e:
    print(e)
    scaler=None
    deployed_nn=None

DEFAULTS=[
17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.0787,
1.095,0.9053,8.589,153.4,0.0064,0.0490,0.0537,0.0159,0.0300,0.0062,
25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
]

labels=[
"Mean Radius","Mean Texture","Mean Perimeter","Mean Area","Mean Smoothness",
"Mean Compactness","Mean Concavity","Mean Concave Points","Mean Symmetry","Mean Fractal Dimension",
"Radius Error","Texture Error","Perimeter Error","Area Error","Smoothness Error",
"Compactness Error","Concavity Error","Concave Points Error","Symmetry Error","Fractal Dimension Error",
"Worst Radius","Worst Texture","Worst Perimeter","Worst Area","Worst Smoothness",
"Worst Compactness","Worst Concavity","Worst Concave Points","Worst Symmetry","Worst Fractal Dimension"
]

def predict_cancer(*features):
    try:
        vals=[float(v) for v in features]
        if scaler is None or deployed_nn is None:
            return "Model not loaded."
        x=scaler.transform(np.array([vals]))
        p=deployed_nn.predict(x,verbose=0)[0][0]
        if p>=0.5:
            return f"🟢 BENIGN\nConfidence: {p:.2%}"
        return f"🔴 MALIGNANT\nConfidence: {(1-p):.2%}"
    except Exception:
        return traceback.format_exc()

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as app:
    gr.Markdown("# 🔬 Breast Cancer Detection")
    comps=[]
    with gr.Tabs():
        names=["Mean Metrics","Error Metrics","Worst Metrics"]
        for t in range(3):
            with gr.Tab(names[t]):
                with gr.Row():
                    for c in range(2):
                        with gr.Column():
                            for i in range(5):
                                idx=t*10+c*5+i
                                comps.append(gr.Number(label=labels[idx],value=DEFAULTS[idx]))
    out=gr.Textbox(label="Result",lines=4)
    with gr.Row():
        run=gr.Button("🔍 Run Neural Network Analysis",variant="primary")
        clear=gr.ClearButton(components=comps+[out],value="🗑 Clear All")
    run.click(predict_cancer,inputs=comps,outputs=out)
    gr.Markdown("### 👨‍💻 Developer\nReplace with your GitHub/LinkedIn links.")

if __name__=="__main__":
    port=int(os.environ.get("PORT",10000))
    app.launch(server_name="0.0.0.0",server_port=port)
