"""
ASA Predictor — loads a trained sklearn pipeline and predicts ASA scores.

Takes a features dict (extracted by the LLM from a chatbot transcript),
calculates BMI, builds a DataFrame matching the training format, and
returns the predicted ASA class with probabilities.
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict
from pathlib import Path


# Column names — must match the training pipeline exactly
FEATURE_COLUMNS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "AlcoholConsumption_X",
    "NumberOfAllergies",
    "NumberOfDiagnoses",
    "Sex",
    "SmokingStatus_Simplified",
    "NeuroPsyk_flag",
    "Respiratorisk_flag",
    "Kardiovaskulært_flag",
    "GI/Lever/Nyre_flag",
    "Endo/Andet_flag",
    "Bevægeapparat_flag",
]

ASA_LABELS = {0: "ASA-I", 1: "ASA-II", 2: "ASA-III"}


class ASAPredictor:
    """Loads a trained .joblib model and predicts ASA scores from extracted features."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"✅ ASA model loaded from {self.model_path.name}")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict ASA score from extracted features.

        Args:
            features: Dict with keys matching FEATURE_SCHEMA from feature_extractor.py.
                      BMI should NOT be included — it's calculated here.

        Returns:
            {
                "asa_class": "ASA-II",
                "asa_numeric": 2,
                "probabilities": {"ASA-I": 0.12, "ASA-II": 0.71, "ASA-III": 0.17},
                "confidence": 0.71,
                "features_used": { ... the full feature dict including BMI ... }
            }
        """
        # Calculate BMI from height and weight
        height_cm = features.get("Height_cm")
        weight_kg = features.get("Weight_kg")

        if height_cm and weight_kg and height_cm > 0:
            bmi = weight_kg / (height_cm / 100) ** 2
        else:
            bmi = np.nan

        # Build a single-row DataFrame with the exact columns the model expects
        row = {
            "Age": features.get("Age", np.nan),
            "Height_cm": height_cm if height_cm else np.nan,
            "Weight_kg": weight_kg if weight_kg else np.nan,
            "BMI": bmi,
            "AlcoholConsumption_X": features.get("AlcoholConsumption_X", np.nan),
            "NumberOfAllergies": features.get("NumberOfAllergies", 0),
            "NumberOfDiagnoses": features.get("NumberOfDiagnoses", 0),
            "Sex": features.get("Sex", "Mand"),
            "SmokingStatus_Simplified": features.get("SmokingStatus_Simplified", "Unknown"),
            "NeuroPsyk_flag": features.get("NeuroPsyk_flag", 0),
            "Respiratorisk_flag": features.get("Respiratorisk_flag", 0),
            "Kardiovaskulært_flag": features.get("Kardiovaskulært_flag", 0),
            "GI/Lever/Nyre_flag": features.get("GI/Lever/Nyre_flag", 0),
            "Endo/Andet_flag": features.get("Endo/Andet_flag", 0),
            "Bevægeapparat_flag": features.get("Bevægeapparat_flag", 0),
        }

        # Handle alcohol sentinel value (-1 means not mentioned → treat as missing)
        if row["AlcoholConsumption_X"] == -1:
            row["AlcoholConsumption_X"] = np.nan

        for key in ["Age", "Height_cm", "Weight_kg", "BMI", "AlcoholConsumption_X",
                    "NumberOfAllergies", "NumberOfDiagnoses",
                    "NeuroPsyk_flag", "Respiratorisk_flag", "Kardiovaskulært_flag",
                    "GI/Lever/Nyre_flag", "Endo/Andet_flag", "Bevægeapparat_flag"]:
            if row[key] is not None and not (isinstance(row[key], float) and np.isnan(row[key])):
                row[key] = float(row[key])

        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # Predict
        predicted_class = int(self.model.predict(df)[0])
        probabilities = self.model.predict_proba(df)[0]

        # Build probability dict
        prob_dict = {
            ASA_LABELS[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        asa_label = ASA_LABELS.get(predicted_class, f"ASA-{predicted_class + 1}")
        confidence = round(float(probabilities[predicted_class]), 4)

        # Clean NaN values for JSON serialization
        clean_features = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in row.items()
        }

        return {
            "asa_class": asa_label,
            "asa_numeric": predicted_class + 1,
            "probabilities": prob_dict,
            "confidence": confidence,
            "features_used": clean_features,
        }