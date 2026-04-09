"""
ml/api/main.py — FastAPI endpoint for ASA prediction
Install: pip install fastapi uvicorn
Run:     uvicorn ml.api.main:app --reload --port 8000
Docs:    http://localhost:8000/docs
"""
import json
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH    = "ml/models/asa_model.joblib"
METADATA_PATH = "ml/models/model_metadata.json"

app = FastAPI(title="ASA Score Prediction API", description="Proof-of-concept — not for clinical use.", version="0.1.0")
pipeline = None
metadata = None

@app.on_event("startup")
def load_model():
    global pipeline, metadata
    try:
        pipeline = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        print(f"✓ Model loaded: {metadata['model_name']}")
    except FileNotFoundError:
        print("⚠ Model not found. Run Colab pipeline first and place files in ml/models/")

class PatientInput(BaseModel):
    age:            Optional[float] = Field(None, example=67)
    bmi:            Optional[float] = Field(None, example=29.4)
    weight_kg:      Optional[float] = Field(None, example=85.0)
    height_cm:      Optional[float] = Field(None, example=170.0)
    sex:            Optional[str]   = Field(None, example="Male")
    smoking_status: Optional[str]   = Field(None, example="Former")
    diabetes:       Optional[str]   = Field(None, example="Type 2")
    hypertension:   Optional[str]   = Field(None, example="Yes")
    surgery_type:   Optional[str]   = Field(None, example="Cardiac")
    class Config:
        extra = "allow"

def run_inference(patient_dict):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place asa_model.joblib in ml/models/ first.")
    df = pd.DataFrame([patient_dict])
    for col in metadata["all_features"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[metadata["all_features"]]
    pred_class = int(pipeline.predict(df)[0])
    pred_label = metadata["class_names"][pred_class]
    probs = {}
    if hasattr(pipeline, "predict_proba"):
        try:
            prob_array = pipeline.predict_proba(df)[0]
            probs = {metadata["class_names"][i]: round(float(p), 4) for i, p in enumerate(prob_array)}
        except: pass
    missing = [f for f in metadata["all_features"] if patient_dict.get(f) is None]
    return {"predicted_class": pred_class, "predicted_label": pred_label, "probabilities": probs, "model_name": metadata["model_name"], "warning": f"Imputed missing: {missing}" if missing else None}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.get("/model-info")
def model_info():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return metadata

@app.post("/predict")
def predict_single(patient: PatientInput):
    return run_inference(patient.dict())

@app.post("/predict-batch")
def predict_batch(patients: list[PatientInput]):
    if len(patients) > 500:
        raise HTTPException(status_code=400, detail="Max 500 patients per request.")
    return {"predictions": [run_inference(p.dict()) for p in patients], "n_patients": len(patients)}
