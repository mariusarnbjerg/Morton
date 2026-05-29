"""
================================================================================
  ASA SCORE PREDICTOR — INFERENCE
================================================================================

  Load a trained model and predict ASA scores from raw patient data.
  Input format matches the dataset columns directly.

  Usage:
    predictor = ASAPredictor("ml_outputs/asa_model_xxx.joblib")
    result = predictor.predict(
        anamnese="Full anamnese text...",
        age=67, sex="Mand", bmi=29.0,
        alcohol=10, smoking="Current smoker",
        cave="Penicillin|Latex", n_allergies=2,
        sks_codes="DJ440|DI10", diagnoses="KOL|Hypertension",
        n_diagnoses=2,
    )

    # From Excel file
    predictor.predict_from_excel("data.xlsx")
================================================================================
"""

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
    OUTPUT_DIR = PROJECT_DIR / "advanced_ml_outputs"

    # Find the latest model file
    model_files = sorted(OUTPUT_DIR.glob("asa_model_*.joblib"))
    if not model_files:
        print(f"No model found in {OUTPUT_DIR}. Run asa_training.py first.")
        exit(1)

    model_path = model_files[-1]
    print(f"Loading model: {model_path.name}\n")
    predictor = ASAPredictor(str(model_path))

    # Demo prediction
    result_5693 = predictor.predict(
        anamnese="Anamnese   Der er ikke dokumenteret nogle diagnoser af betydning for denne anæstesi         Anamnese:   Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:  Psykiatrisk sygehistorie med angst    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.    Bevægeapparat:  I.a    ",
        age=19,
        sex="Kvinde",
        bmi=19.3,
        alcohol=None,
        smoking="Unknown",
        cave="ANDRE ANTIINFLAM./ANTIRHEUM. MIDLER, NON-STEROIDE",
        n_allergies=1,
        sks_codes="DD486",
        diagnoses="Ikke spec. tumor i mamma (DD486)",
        n_diagnoses=1,
    )
    print(f"Prediction 5693: {result_5693['predicted_label']}")
    for cls, prob in result_5693["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_28435 = predictor.predict(
        anamnese="  Anamnese           Respiratorisk:   Rygestatus: Aldrig vurderet    GI/Lever/Nyre:   Ingen gastroøsofageal refluks  Bevægeapparat:      Objektiv undersøgelse:       Højde og vægt:   Højde: 166,0 cm d. 13-03-2018  Vægt: 52,0 kg d. 13-03-2018    Luftvej:   Mallampati: I  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed    Tandstatus:   I.a.         ",
        age=83,
        sex="Kvinde",
        bmi=18.9,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DM179",
        diagnoses="Knæledsartrose UNS (DM179)",
        n_diagnoses=1,
    )
    print(f"Prediction 28435: {result_28435['predicted_label']}")
    for cls, prob in result_28435["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_40179 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 167,0 cm d. 18-05-2022  Vægt: 61,0 kg d. 18-05-2022    Kardiovaskulært:   I.a.      Respiratorisk:   I.a.      Abdominalt:   I.a.      Ryg:   I.a.         ",
        age=49,
        sex="Kvinde",
        bmi=21.9,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DM199",
        diagnoses="Artrose UNS (DM199)",
        n_diagnoses=1,
    )
    print(f"Prediction 28435: {result_40179['predicted_label']}")
    for cls, prob in result_40179["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_2269 = predictor.predict(
        anamnese="  Anamnese   rask      Anamnese:   Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 168,0 cm d. 24-02-2025  Vægt: 80,0 kg d. 24-02-2025    Kardiovaskulært:   I.a.      Respiratorisk:   I.a.      Abdominalt:   I.a.      Ryg:   I.a.         ",
        age=61,
        sex="Kvinde",
        bmi=28.3,
        alcohol=None,
        smoking="Unknown",
        cave="TRAMADOL| SULFAMETHIZOL",
        n_allergies=2,
        sks_codes="DZ928F1 | DN811 | DN393",
        diagnoses="Anamnese m. op. for uterovaginal prolaps i samme kompartment (DZ928F1) | Cystocele hos kvinde (DN811) | Stress-inkontinens (DN393)",
        n_diagnoses=3,
    )
    print(f"Prediction 2269: {result_2269['predicted_label']}")
    for cls, prob in result_2269["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_5573 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   Astma  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:       Højde og vægt:   Højde: 169,0 cm d. 09-06-2022  Vægt: 80,0 kg d. 09-06-2022    Luftvej:   Mallampati: I  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed    Tandstatus:   I.a.         ",
        age=39,
        sex="Kvinde",
        bmi=28,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DO021",
        diagnoses="Graviditet med dødt retineret foster (missed abortion)(DO021)",
        n_diagnoses=1,
    )
    print(f"Prediction 5573: {result_5573['predicted_label']}")
    for cls, prob in result_5573["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_13048 = predictor.predict(
        anamnese="Anamnese   Andet   (+) Primær dobbeltsidig knæledsartrose (DM170)         Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:       Højde og vægt:   Højde: 175,0 cm d. 29-06-2021  Vægt: 82,0 kg d. 29-06-2021    Luftvej:   Mallampati: II  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed    Kardiovaskulært:   I.a.      Tandstatus:   I.a.        Respiratorisk:   I.a.      Abdominalt:   I.a.      Ryg:   I.a.         ",
        age=74,
        sex="Mand",
        bmi=26.8,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DM170",
        diagnoses="Primær dobbeltsidig knæledsartrose (DM170)",
        n_diagnoses=1,
    )
    print(f"Prediction 13048: {result_13048['predicted_label']}")
    for cls, prob in result_13048["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_14653 = predictor.predict(
        anamnese="  Anamnese         Anamnese:   Patient har ikke tidligere haft anæstesi komplikationer  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 179,0 cm d. 19-08-2024  Vægt: 84,0 kg d. 19-08-2024    Luftvej:   Mallampati: II  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed    Kardiovaskulært:   I.a.      Tandstatus:   I.a.      Respiratorisk:   I.a.         ",
        age=43,
        sex="Mand",
        bmi=26.2,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DT939",
        diagnoses="Følgetilstand efter læsion af underekstremitet UNS (DT939)",
        n_diagnoses=1,
    )
    print(f"Prediction 14653: {result_14653['predicted_label']}")
    for cls, prob in result_14653["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_6997 = predictor.predict(
        anamnese="  Anamnese         Anamnese:   Patient har ikke tidligere haft anæstesi komplikationer  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 179,0 cm d. 19-08-2024  Vægt: 84,0 kg d. 19-08-2024    Luftvej:   Mallampati: II  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed    Kardiovaskulært:   I.a.      Tandstatus:   I.a.      Respiratorisk:   I.a.         ",
        age=56,
        sex="Mand",
        bmi=26.9,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DK829",
        diagnoses="Sygdom i galdeblæren UNS (DK829)",
        n_diagnoses=1,
    )
    print(f"Prediction 6997: {result_6997['predicted_label']}")
    for cls, prob in result_6997["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_10293 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Patient har ikke tidligere haft anæstesi komplikationer  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  I.a.    Bevægeapparat:  forkortet rygmarv, jf. mor. må ikke få neuroaksiel blokade.      Objektiv undersøgelse:     Neurologisk:   I.a.    GCS: 15    Højde og vægt:   Højde: 192,0 cm d. 25-06-2021  Vægt: 100,0 kg d. 25-06-2021    Luftvej:   Mallampati: I  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed    Tandstatus:   I.a.         ",
        age=24,
        sex="Mand",
        bmi=27.1,
        alcohol=None,
        smoking="Unknown",
        cave="MORPHIN",
        n_allergies=1,
        sks_codes="DS921",
        diagnoses="Fraktur af talus (DS921)",
        n_diagnoses=1,
    )
    print(f"Prediction 10293: {result_10293['predicted_label']}")
    for cls, prob in result_10293["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_13812 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Patient har ikke tidligere haft anæstesi komplikationer  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.    Bevægeapparat:      Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 153,0 cm d. 19-12-2023  Vægt: 61,0 kg d. 19-12-2023    Luftvej:   Mallampati: I  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed    Kardiovaskulært:   Hjerterytme: regelmæssig    Tandstatus:   I.a.         ",
        age=44,
        sex="Kvinde",
        bmi=26.1,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DN920",
        diagnoses="Menoragi eller polymenoré (DN920)",
        n_diagnoses=1,
    )
    print(f"Prediction 13812: {result_13812['predicted_label']}")
    for cls, prob in result_13812["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_4162 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Pt.journal set, men patienten er ikke til stede    Respiratorisk:   Rygestatus: Aldrig vurderet  Bevægeapparat:      Objektiv undersøgelse:       Højde og vægt:   Højde: 158,0 cm d. 17-08-2023  Vægt: 76,0 kg d. 17-08-2023    Luftvej:   BMI 30-35      Kardiovaskulært:   EKG gennemgået  SR    Abdominalt:   P.t. er adipøs       ",
        age=26,
        sex="Kvinde",
        bmi=30.4,
        alcohol=None,
        smoking="Unknown",
        cave="OXYCODON",
        n_allergies=1,
        sks_codes="DM758 | DS461C",
        diagnoses="Anden skulderlidelse (DM758) | Læsion af SLAP (superior labrum anterior posterior) (DS461C)",
        n_diagnoses=2,
    )
    print(f"Prediction 4162: {result_4162['predicted_label']}")
    for cls, prob in result_4162["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")


    result_7884 = predictor.predict(
        anamnese="Anamnese           Respiratorisk:   Rygestatus: Aldrig vurderet    Kardiovaskulært:   Går til 2. sal uden pause (4 Mets)    GI/Lever/Nyre:   Ingen gastroøsofageal refluks  Bevægeapparat:      Objektiv undersøgelse:       Højde og vægt:   Højde: 165,0 cm d. 11-03-2024  Vægt: 95,0 kg d. 11-03-2024    Luftvej:   Mallampati: III  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed  BMI 30-35      Tandstatus:   I.a.         ",
        age=57,
        sex="Mand",
        bmi=34.9,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DK802",
        diagnoses="Sten i galdeblæren uden kolecystitis (DK802)",
        n_diagnoses=1,
    )
    print(f"Prediction 7884: {result_7884['predicted_label']}")
    for cls, prob in result_7884["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_8983 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Pt. er i fast opioidbehandling med:  tramadol (DOLATRAMYL) 150 mg depottablet  Neuro/Psyk - øvrigt:   I.a.  Nedsat hørelse, bruger høreapparat  Tramadol 300mg svt morfin 60 mg i alt/dg (knæ og rygsmerter)     Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.    Går til 2. sal uden pause (4 Mets)    GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  I.a.    Bevægeapparat:  I.a  AKTUELT: venstresidig re-TKA       Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 170,0 cm d. 06-06-2023  Vægt: 91,2 kg d. 06-06-2023    Luftvej:   Mallampati: IV  Mundåbning: >4 cm  Underbid: normal  TM afstand: >3 FB  fuld nakkebevægelighed  BMI 30-35    Akkurat 4 cm mundåbning    Kardiovaskulært:   St.p. et c.: Ia iflg jnl     Tandstatus:   Fuldprotese i overmund    Abdominalt:   I.a.      Ryg:   I.a.         ",
        age=76,
        sex="Mand",
        bmi=31.6,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DZ966B",
        diagnoses="Tilstand med knæledsprotese (DZ966B)",
        n_diagnoses=1,
    )
    print(f"Prediction 8983: {result_8983['predicted_label']}")
    for cls, prob in result_8983["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_6417 = predictor.predict(
        anamnese="Procedure: ORKIEKTOMI (ENKELTSIDIG) (Højre)  TESTISBIOPSI (Venstre)    Anæstesianamnese       Pt.journal set, men patienten er ikke til stede      Neuro/Psyk:   Har epilepsi    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.      Bevægeapparat:  I.a      Objektiv undersøgelse:       Højde og vægt:   Højde: 180,0 cm d. 22-06-2018  Vægt: 63,0 kg d. 22-06-2018    Kardiovaskulært:   Stet c ia jf journal    Respiratorisk:   Stet p ia jf journal         ",
        age=39,
        sex="Mand",
        bmi=19.4,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DZ031T",
        diagnoses="Obs. pga mistanke om kræft i testis (DZ031T)",
        n_diagnoses=1,
    )
    print(f"Prediction 6417: {result_6417['predicted_label']}")
    for cls, prob in result_6417["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_6574 = predictor.predict(
        anamnese="Anamnese   Andet   (+) AL amyloidose (DE858A)   (+) Dyb flebitis i underekstremitet UNS (DI803C)   (+) Hjertehypertrofi (DI517C)   (+) Hjertesvigt UNS (DI509)         Anamnese:   Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   Rygestatus: Aldrig vurderet    Kardiovaskulært:   Antikoagulerende behandling.    GI/Lever/Nyre:   Kronisk nyresygdom    Endo/Andet:  Hæmatologi:  Bevægeapparat:  I.a      Objektiv undersøgelse:       Højde og vægt:   Højde: 183,0 cm d. 29-04-2022  Vægt: 94,0 kg d. 29-04-2022    Kardiovaskulært:   I.a.      Respiratorisk:   I.a.         ",
        age=61,
        sex="Mand",
        bmi=28.1,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DH492",
        diagnoses="Abducensparese (DH492) | Abducensparese",
        n_diagnoses=1,
    )
    print(f"Prediction 6574: {result_6574['predicted_label']}")
    for cls, prob in result_6574["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_11614 = predictor.predict(
        anamnese="  Anamnese         Anamnese:   Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:       Højde og vægt:   Højde: 186,0 cm d. 10-09-2024  Vægt: 106,6 kg d. 10-09-2024    Luftvej:   BMI 30-35         ",
        age=24,
        sex="Mand",
        bmi=30.8,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DK409 | DS901A",
        diagnoses="Ingvinalhernie UNS uden ileus eller gangræn (DK409) | Kontusion af tå UNS (DS901A)",
        n_diagnoses=2,
    )
    print(f"Prediction 11614: {result_11614['predicted_label']}")
    for cls, prob in result_11614["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_108759 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Pt.journal set, men patienten er ikke til stede  Neuro/Psyk - øvrigt:  Svær hjerneskade, multihandicappet,  udviklingshæmmet, udadreagerende     Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   Colitis ulcerosa    Endo/Andet:  I.a.    Bevægeapparat:  I.a      Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 150,0 cm d. 11-08-2023  Vægt: 45,3 kg d. 11-08-2023    Luftvej:   Kan ikke kooperere    Kardiovaskulært:   I.a.      Tandstatus:   ?    Respiratorisk:   I.a.      Abdominalt:   I.a.      Ryg:   I.a.         ",
        age=29,
        sex="Mand",
        bmi=20.1,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DN309",
        diagnoses="Cystitis UNS (DN309)",
        n_diagnoses=1,
    )
    print(f"Prediction 108759: {result_108759['predicted_label']}")
    for cls, prob in result_108759["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_91578 = predictor.predict(
        anamnese="  Anamnese   HYSTEROSKOPISK POLYPRESEKTION - _      Anamnese:   Pt.har aldrig tidligere været i GA  Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   Hypertension (Losartan), velreguleret    GI/Lever/Nyre:   Gastroøsofageal refluks (PPI, fast forebyggende pga. ibuprofen), velreguleret    Endo/Andet:  Wegovy for vægttab.    Bevægeapparat:  2 diskusprolaps i lænden (Gabapentin 900 + 900 + 600 mg/dgl)      Objektiv undersøgelse:     Neurologisk:   I.a.    GCS: 15    Højde og vægt:   Højde: 163,0 cm d. 19-03-2025  Vægt: 111,0 kg d. 19-03-2025    Luftvej:   Mallampati: I  Mundåbning: >4 cm  Underbid: normal  fuld nakkebevægelighed  BMI >40      Kardiovaskulært:   Stet. P & C i.a. i flg. AOP    Tandstatus:   I.a.           Der er følgende mangler i journalen inden det samlede Prænotat kan afsluttes:  Relevante prøvesvar    ",
        age=55,
        sex="Kvinde",
        bmi=41.8,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DN840A | DN950 | DN841",
        diagnoses="Endometriepolyp i livmoderen (DN840A) | Postmenopausal metroragi (DN950) | Polyp i livmoderhalsen (DN841)",
        n_diagnoses=3,
    )
    print(f"Prediction 91578: {result_91578['predicted_label']}")
    for cls, prob in result_91578["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_1353055 = predictor.predict(
        anamnese="  Anæstesianamnese       Pt.journal set, men patienten er ikke til stede      Neuro/Psyk:    I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  C mam.    Bevægeapparat:  I.a      Objektiv undersøgelse:     CNS:   I.a.      Højde og vægt:   Højde: 165,0 cm d. 06-06-2025  Vægt: 152,8 kg d. 06-06-2025    Luftvej:   Mallampati: II  Mundåbning: >4 cm  Underbid: normal  TM afstand: >3 FB  fuld nakkebevægelighed  BMI >40      Kardiovaskulært:   I.a.      Tandstatus:   I.a.      Respiratorisk:   I.a.         ",
        age=35,
        sex="Kvinde",
        bmi=56.1,
        alcohol=None,
        smoking="Unknown",
        cave="PLASTER",
        n_allergies=1,
        sks_codes="DC509",
        diagnoses="Brystkræft UNS, venstresidig (DC509 + TUL2)",
        n_diagnoses=1,
    )
    print(f"Prediction 1353055: {result_1353055['predicted_label']}")
    for cls, prob in result_1353055["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_79713 = predictor.predict(
        anamnese="Procedure: Ureteronefroskopi + cystoskopi m. biopsi (Venstre)  CYSTOSKOPI MED BIOPSI (_)    Anæstesianamnese       Pt.journal set, men patienten er ikke til stede      Neuro/Psyk:   Lamotrigin  cymbalta  kroniske mavesmerter   Psykiatrisk sygehistorie med depression    Respiratorisk:   Rygestatus: Aldrig vurderet  Under udredning for ILS.seneste nrmal spirometri men DLCO 70 % (jan 2025)  har tid i lungeamb efter operationsdag. dette bør fremrykkes.  forpustet ved trappegang    GI/Lever/Nyre:   Colitis ulcerosa  primær sl´kleroserende kolangit  mesalazin  mirikizymab    Endo/Andet:  Hypotyroidisme, dysreguleret    Bevægeapparat:  endometriose    fibromyalgi  spl      Objektiv undersøgelse:       Højde og vægt:   Højde: 169,0 cm d. 01-10-2025  Vægt: 57,0 kg d. 01-10-2025    Respiratorisk:   FEV 1 (L) 4,07  FEV 1 (%) 116  FVC (L) 4,62  FVC (%) 111  FEV 1 / FVC 88  RV (%) 127  TLC (%) 122  DLCO (%) 81  KCO (DLCO/VA) (%) 73  DLCOc (%) 81  Undersøgelse: udvidet lungefunktion           Under udredning for ILS. DLCO 70 % i jan 2025. ny LFU inden OP (FUS læge er kontaktet)    10/10-25: Normal LFU    ",
        age=28,
        sex="Kvinde",
        bmi=20,
        alcohol=None,
        smoking="Unknown",
        cave="MORPHIN| LEVOTHYROXINNATRIUM| INFLIXIMAB",
        n_allergies=3,
        sks_codes="DZ031H",
        diagnoses="Obs. pga mistanke om kræft i urinveje (DZ031H)",
        n_diagnoses=1,
    )
    print(f"Prediction 79713: {result_79713['predicted_label']}")
    for cls, prob in result_79713["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_4477 = predictor.predict(
        anamnese="Ambulant skopi mislykkedes. Angst og vanskelig at stikke  Anamnese         Anamnese:   Patient har ikke tidligere haft anæstesi komplikationer  Neuro/Psyk - øvrigt:  Psykiatrisk sygehistorie med angst  Zertralin     Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   I.a.      Endo/Andet:  Diabetes Type 2  Bevægeapparat:      Objektiv undersøgelse:     Neurologisk:   I.a.      Højde og vægt:   Højde: 170,0 cm d. 28-11-2024  Vægt: 129,0 kg d. 28-11-2024    Luftvej:   Mallampati: II  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed  BMI >40      Kardiovaskulært:   I.a.      Tandstatus:   I.a.      Respiratorisk:   I.a.      Abdominalt:   P.t. er adipøs       ",
        age=44,
        sex="Kvinde",
        bmi=44.6,
        alcohol=None,
        smoking="Unknown",
        cave="NIKKEL",
        n_allergies=1,
        sks_codes="DK573 | DZ031D",
        diagnoses="Divertikulose eller divertikulit i tyktarm u perf. el absces (DK573) | Obs. pga mistanke om kræft i tyktarmen eller endetarmen (DZ031D)",
        n_diagnoses=2,
    )
    print(f"Prediction 4477: {result_4477['predicted_label']}")
    for cls, prob in result_4477["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_2341 = predictor.predict(
        anamnese="Anamnese         Anamnese:   Patient har ikke tidligere haft anæstesi komplikationer  Neuro/Psyk - øvrigt:  PTSD      Respiratorisk:   Rygestatus: Aldrig vurderet    Kardiovaskulært:   Hypertension, velreguleret    GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  Diabetes (HbA1C=65) Type 2  Bevægeapparat:      Objektiv undersøgelse:       Højde og vægt:   Højde: 177,0 cm d. 20-11-2023  Vægt: 100,0 kg d. 20-11-2023    Luftvej:   Mallampati: II  Underbid: normal  TM afstand: >6,5 cm  BMI 30-35      Tandstatus:   I.a.         ",
        age=56,
        sex="Mand",
        bmi=31.9,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DK429",
        diagnoses="Umbilikalhernie uden ileus eller gangræn (DK429)",
        n_diagnoses=1,
    )
    print(f"Prediction 2341: {result_2341['predicted_label']}")
    for cls, prob in result_2341["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_26219 = predictor.predict(
        anamnese="Anamnese         Neuro/Psyk - øvrigt:   I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   Pt har pacemaker  Arytmier: atrieflimmer/flagren  Velkompenseret.    GI/Lever/Nyre:   I.a.      Endo/Andet:  Prostatacancer  Bevægeapparat:  I.a      Objektiv undersøgelse:       Højde og vægt:   Højde: 178,0 cm d. 07-03-2022  Vægt: 77,5 kg d. 07-03-2022    Luftvej:   Mallampati: I    Kardiovaskulært:   EKG gennemgået  SR, OK    Tandstatus:   Delprotese i overmund  Delprotese i undermund       ",
        age=88,
        sex="Mand",
        bmi=24.5,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DC187 | DI480 | DZ933",
        diagnoses="Kræft i colon sigmoideum (DC187) | Paroksysmatisk atrieflimren (DI480) | Tilstand med kolostomi (DZ933)",
        n_diagnoses=3,
    )
    print(f"Prediction 26219: {result_26219['predicted_label']}")
    for cls, prob in result_26219["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")

    result_2771500 = predictor.predict(
        anamnese="Procedure: URETEROSKOPISK STENFJERNELSE (Venstre)    Anæstesianamnese       Patient har tidligere haft anæstesi komplikationer (PONV - opkast efter diskusprolaps operation)  Pt. er i fast opioidbehandling   med:  oxycodon (OXYCODONE TEVA) 5 mg kapsel    Neuro/Psyk:    I.a.    Respiratorisk:   I.a.  Rygestatus: Aldrig vurderet    Kardiovaskulært:   I.a.      GI/Lever/Nyre:   Ingen gastroøsofageal refluks    Endo/Andet:  I.a.    Wegovy beh - tabt 12 kg seneste 3 mdr    Bevægeapparat:  I.a      Objektiv undersøgelse:       Højde og vægt:   Højde: 185,0 cm d. 13-10-2025  Vægt: 149,0 kg d. 13-10-2025    Luftvej:   Mallampati: II  Mundåbning: >4 cm  Underbid: normal  TM afstand: >6,5 cm  fuld nakkebevægelighed  BMI >40    Fuldskæg    Kardiovaskulært:   Stet c ia jf journal    Tandstatus:   I.a.      Respiratorisk:   Stet p ia jf journal         ",
        age=33,
        sex="Mand",
        bmi=43.5,
        alcohol=None,
        smoking="Unknown",
        cave="",
        n_allergies=0,
        sks_codes="DN202",
        diagnoses="Nyresten med uretersten UNS (DN202)",
        n_diagnoses=1,
    )
    print(f"Prediction 2771500: {result_2771500['predicted_label']}")
    for cls, prob in result_2771500["probabilities"].items():
        print(f"  {cls}: {prob:.1%}")