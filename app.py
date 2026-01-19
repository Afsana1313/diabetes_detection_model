import gradio as gr
import pandas as pd
import pickle
#main logic
with open("pipeline.pkl","rb") as f:
    model = pickle.load(f)

def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):

    BMI_Age_Risk = BMI * Age
    Glucose_BMI = Glucose * BMI

    data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                          DiabetesPedigreeFunction, Age, BMI_Age_Risk, Glucose_BMI]],
                        columns=[
                            "Pregnancies",
                            "Glucose",
                            "BloodPressure",
                            "SkinThickness",
                            "Insulin",
                            "BMI",
                            "DiabetesPedigreeFunction",
                            "Age",
                            "BMI_Age_Risk",
                            "Glucose_BMI"
                        ])

    prediction = model.predict(data)[0]

    return "Diabetic" if prediction == 1 else "Not Diabetic"
    
  


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="DiabetesPedigreeFunction"),
          gr.Number(label="Age")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="ML Model Prediction App",
    description="Enter input features to get model prediction"
)

interface.launch()
