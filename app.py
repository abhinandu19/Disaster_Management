import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load trained models
cat_clf = joblib.load("catboost_classifier.pkl")
cat_reg = joblib.load("catboost_regssion.pkl")

# Load preprocessing objects
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def preprocess_input(df):
    df = df.copy()

    # ---- Date handling ----
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df.drop("date", axis=1, inplace=True)

    # ---- Encoding ----
    for col in df.columns:
        if df[col].dtype == "object":
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
            else:
                df[col] = 0  # fallback for unseen category

    # ---- Scaling ----
    num_cols = scaler.feature_names_in_
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def predict(file):
    df = pd.read_csv(file.name)

    processed = preprocess_input(df)

    # ---- Prediction ----
    pred_class = cat_clf.predict(processed).ravel()
    pred_reg = cat_reg.predict(processed).ravel()

    df["Pred_Response_Efficiency"] = pred_class
    df["Pred_Recovery_Days"] = pred_reg

    return df


# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload Test CSV"),
    outputs=gr.Dataframe(label="Prediction Output"),
    title="Disaster Response Prediction System",
    description="Upload a CSV file to predict response efficiency and recovery days"
)

interface.launch()