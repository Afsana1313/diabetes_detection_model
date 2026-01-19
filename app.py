import gradio as gr
import numpy as np
import pickle
#main logic
with open("pipeline.pkl","rb") as f:
    model = pickle.load(f)

def predict(*inputs):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Age")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="ML Model Prediction App",
    description="Enter input features to get model prediction"
)

interface.launch()
