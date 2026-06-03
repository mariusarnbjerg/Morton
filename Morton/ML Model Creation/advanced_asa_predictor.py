import re
import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

class SafeTfidfVectorizer(TfidfVectorizer):
    def fit(self, raw_documents, y=None):
        raw_documents = pd.Series(raw_documents).fillna("").astype(str)
        try:
            super().fit(raw_documents)
            self._empty = False
        except ValueError:
            self._empty = True
            self.vocabulary_ = {}
        return self

    def fit_transform(self, raw_documents, y=None):
        raw_documents = pd.Series(raw_documents).fillna("").astype(str)
        try:
            result = super().fit_transform(raw_documents)
            self._empty = False
            return result
        except ValueError:
            self._empty = True
            self.vocabulary_ = {}
            return sparse.csr_matrix((len(raw_documents), 0))

    def transform(self, raw_documents):
        raw_documents = pd.Series(raw_documents).fillna("").astype(str)
        if getattr(self, "_empty", False):
            return sparse.csr_matrix((len(raw_documents), 0))
        return super().transform(raw_documents)

    def get_feature_names_out(self, input_features=None):
        if getattr(self, "_empty", False):
            return np.array([], dtype=object)
        return super().get_feature_names_out(input_features)
# =============================================================================
# Leakage prevention (same as training)
# =============================================================================

_PLAN_SPLIT = re.compile(
    r"\bPlan\s+for\s+an[æa]stesi(?:en)?\s*[:\-]",
    flags=re.IGNORECASE,
)

_BACKUP_PLAN = re.compile(
    r"\b(?:Planlagt\s+an[æa]stesitype|Luftvejsplan|Anæstesiplan|"
    r"Planlagt\s+luftvej|Monitorering|Induktion|Samtykke)\s*[:\-]",
    flags=re.IGNORECASE,
)

_ASA_PATTERNS = [
    r"\bASA[\s:\-]*(?:klasse|score|status|ps|class)?[\s:\-]*[1-5IVX]+\b",
    r"\bASA[\s:\-]*[1-5]\b",
    r"\bASA[\s:\-]*(?:I{1,3}|IV|V)\b",
    r"\bASA\s*scor[a-z]*\s*[:\s]*[1-5IVX]+\b",
    r"\bASA\s*(?:klasse|score|status|ps|class)?\b",
]


def clean_anamnese(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    match = _PLAN_SPLIT.search(text)
    if match:
        text = text[:match.start()]
    else:
        backup = _BACKUP_PLAN.search(text)
        if backup:
            text = text[:backup.start()]
    for pat in _ASA_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# Predictor
# =============================================================================

class ASAPredictor:
    CLASS_NAMES = ["ASA 1", "ASA 2", "ASA 3"]

    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)

    def predict(
        self,
        anamnese: str = "",
        age=None,
        sex=None,
        bmi=None,
        alcohol=None,
        smoking: str = "Unknown",
        cave: str = "",
        n_allergies: int = 0,
        sks_codes: str = "",
        diagnoses: str = "",
        n_diagnoses: int = 0,
    ) -> dict:
        """
        Predict ASA score from raw patient data.

        Args match the dataset columns:
            anamnese:     Free text (Anamnese column)
            age:          Numeric (Age column)
            sex:          "Mand" or "Kvinde" (Sex column)
            bmi:          Numeric (BMI column)
            alcohol:      Units/week or None (AlcoholConsumption_X column)
            smoking:      One of "Current smoker", "Former smoker",
                          "Never smoker", "Unknown" (SmokingStatus_Simplified)
            cave:         Pipe-separated allergies or "" (CAVE column)
            n_allergies:  Count (NumberOfAllergies column)
            sks_codes:    Pipe-separated codes (SKS_codes column)
            diagnoses:    Pipe-separated names (Diagnoses column)
            n_diagnoses:  Count (NumberOfDiagnoses column)
        """
        # Build feature row matching training format
        sex_val = {"mand": 1, "kvinde": 0}.get(
            str(sex).strip().lower(), np.nan
        ) if sex else np.nan

        row = {
            "age": float(age) if age is not None else np.nan,
            "bmi": float(bmi) if bmi is not None else np.nan,
            "alcohol": float(alcohol) if alcohol is not None else np.nan,
            "n_allergies": int(n_allergies),
            "n_diagnoses": int(n_diagnoses),
            "sex": sex_val,
            "smoking_status": str(smoking) if smoking else "Unknown",
            "anamnese_clean": clean_anamnese(anamnese),
            "diagnoses_text": str(diagnoses) if diagnoses else "",
            "sks_text": str(sks_codes) if sks_codes else "",
            "cave_text": str(cave) if cave else "",
        }

        df = pd.DataFrame([row])

        prediction = self.pipeline.predict(df)[0]
        probabilities = self.pipeline.predict_proba(df)[0]

        return {
            "predicted_class": int(prediction),
            "predicted_label": self.CLASS_NAMES[prediction],
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(self.CLASS_NAMES, probabilities)
            },
        }

    def predict_from_row(self, row: pd.Series) -> dict:
        """
        Predict from a raw Excel row using the original column names.
        """
        def get(col, default=None):
            return row[col] if col in row.index and pd.notna(row[col]) else default

        alcohol = get("AlcoholConsumption_X")
        if isinstance(alcohol, str) and alcohol.upper() == "NULL":
            alcohol = None

        cave = get("CAVE", "")
        if isinstance(cave, str) and cave.upper() == "NULL":
            cave = ""

        return self.predict(
            anamnese=get("Anamnese", ""),
            age=get("Age"),
            sex=get("Sex"),
            bmi=get("BMI"),
            alcohol=alcohol,
            smoking=get("SmokingStatus_Simplified", "Unknown"),
            cave=cave,
            n_allergies=get("NumberOfAllergies", 0),
            sks_codes=get("SKS_codes", ""),
            diagnoses=get("Diagnoses", ""),
            n_diagnoses=get("NumberOfDiagnoses", 0),
        )

    def predict_from_excel(self, filepath: str, limit: int = None) -> pd.DataFrame:
        """
        Run predictions on an entire Excel file.
        Returns a DataFrame with original data plus predictions.
        """
        df = pd.read_excel(filepath)
        if limit:
            df = df.head(limit)

        predictions = []
        for idx, row in df.iterrows():
            try:
                result = self.predict_from_row(row)
                predictions.append({
                    "row": idx,
                    "predicted_label": result["predicted_label"],
                    **result["probabilities"],
                })
            except Exception as e:
                predictions.append({
                    "row": idx,
                    "predicted_label": f"ERROR: {e}",
                })

        pred_df = pd.DataFrame(predictions)

        # If ASA Score exists, add comparison
        if "ASA Score" in df.columns:
            pred_df["actual_asa"] = df["ASA Score"].values[:len(pred_df)]

        return pred_df


# =============================================================================
# Usage
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    PROJECT_DIR = Path(__file__).parent
    OUTPUT_DIR = PROJECT_DIR / "advanced_ml_outputs (LightGBM_balanced)"

    # Find the latest model file
    model_files = sorted(OUTPUT_DIR.glob("asa_model_*.joblib"))
    if not model_files:
        print(f"No model found in {OUTPUT_DIR}. Run asa_training.py first.")
        exit(1)

    model_path = model_files[-1]
    print(f"Loading model: {model_path.name}\n")
    predictor = ASAPredictor(str(model_path))

    # Demo prediction
    result = predictor.predict(
        anamnese="Anamnese Der er ikke dokumenteret nogle diagnoser af betydning for denne anæstesi. Anamnese: Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:  Psykiatrisk sygehistorie med angst Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.    Bevægeapparat:  I.a    ",
        age=27,
        sex="Mand",
        bmi=19.3,
        alcohol=None,
        smoking="Unknown",
        cave="ANDRE ANTIINFLAM./ANTIRHEUM. MIDLER, NON-STEROIDE",
        n_allergies=1,
        sks_codes="",
        diagnoses="",
        n_diagnoses=1,
    )
    print(f"Prediction: {result['predicted_label']}")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")