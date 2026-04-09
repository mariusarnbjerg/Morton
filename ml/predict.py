"""
predict.py — ASA Score Prediction Inference Script
Run: python ml/predict.py
"""
import json, sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

MODEL_PATH    = "ml/models/asa_model.joblib"
METADATA_PATH = "ml/models/model_metadata.json"

EXAMPLE_PATIENT = {
    "age": 67, "bmi": 29.4, "weight_kg": 85.0, "height_cm": 170.0,
    "sex": "Male", "smoking_status": "Former",
    "diabetes": "Type 2", "hypertension": "Yes", "surgery_type": "Cardiac",
}

EXAMPLE_BATCH = [
    {"age":45,"bmi":22.1,"weight_kg":68.0,"height_cm":175.0,"sex":"Female","smoking_status":"Never","diabetes":"No","hypertension":"No","surgery_type":"General"},
    {"age":72,"bmi":31.8,"weight_kg":95.0,"height_cm":173.0,"sex":"Male","smoking_status":"Current","diabetes":"Type 2","hypertension":"Yes","surgery_type":"Cardiac"},
    {"age":58,"bmi":None,"weight_kg":80.0,"height_cm":168.0,"sex":"Female","smoking_status":"Unknown","diabetes":"No","hypertension":"Yes","surgery_type":"Orthopaedic"},
]

def load_model(model_path, metadata_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nRun Colab first and place the .joblib file in ml/models/")
    pipeline = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"✓ Model loaded: {metadata['model_name']} (trained: {metadata['trained_on']})")
    return pipeline, metadata

def format_input(patient_data, metadata):
    if isinstance(patient_data, dict):
        patient_data = [patient_data]
    df = pd.DataFrame(patient_data)
    for col in metadata["all_features"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[metadata["all_features"]]

def predict(pipeline, input_df, metadata):
    class_names = metadata["class_names"]
    preds = pipeline.predict(input_df)
    results = pd.DataFrame({"predicted_class": preds, "predicted_label": [class_names[p] for p in preds]})
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(input_df)
            for i, name in enumerate(class_names):
                results[f"prob_{name.replace(' ','_')}"] = probs[:, i].round(3)
        except: pass
    return results

if __name__ == "__main__":
    pipeline, metadata = load_model(MODEL_PATH, METADATA_PATH)
    print("\nRunning batch example...")
    input_df = format_input(EXAMPLE_BATCH, metadata)
    results  = predict(pipeline, input_df, metadata)
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(results.to_string())
    print("="*50)
