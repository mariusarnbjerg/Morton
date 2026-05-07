"""
================================================================================
  ASA SCORE PREDICTION — MASTER'S THESIS ML PIPELINE
  Preoperative Anesthesiology AI — Proof of Concept
================================================================================

  Author      : Marius Arnbjerg & Mikkel Trolle
  Institution : Aalborg University

  PURPOSE:
    Supervised classification of ASA physical status scores from structured
    preoperative patient data and preoperative text.

  IMPORTANT DESIGN CHOICES:
    - ASA is used as the target label.
    - ASA is removed from the text features before training, so the model does
      not simply read the answer from the note.
    - DurableKey is used for joins and patient-level splitting, but NOT as a
      feature. This avoids patient ID memorisation.
    - Train, validation, and test splits are done by patient/DurableKey to reduce
      leakage between notes from the same patient.

  DATA SOURCES:
    - Patientinfo: alder, køn, rygning, alkohol, CAVE
    - Diagnoser: active SKS-koder at ydelsesdato
    - Anæstesiprætilsynsnotatet: preoperative text sections, height, weight, ASA

  EXCLUDED:
    - Postoperative or treatment-related sheets are ignored to avoid leakage.

  USAGE:
    Run this file in PyCharm.
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost ikke tilgængeligt.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM ikke tilgængeligt.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "Dataset" / "Endelige_udtræk_INC6299351.xlsx"
OUTPUT_DIR = PROJECT_DIR / "ml_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "asa_score"
ID_COLUMNS = ["durable_key", "ydelsesdato", "fodselsdato"]

# Core structured features
NUMERICAL_FEATURES = [
    "alder",
    "alkohol_per_uge",
    "hoejde",
    "vaegt",
    "bmi",
    "n_cave",
    "n_diagnoser",
    "mallampati",
    "n_text_chars",
]

BINARY_FEATURES = [
    "sex",
    "neuro_psyk",
    "respiratorisk",
    "kardiovaskulaert",
    "gi_lever_nyre",
    "endo_andet",
    "bevaegeapparat",
    "has_hypertension",
    "has_diabetes",
    "has_asthma",
    "has_kol",
    "has_sleep_apnea",
    "has_cardiac_disease",
    "has_reflux",
    "has_renal_disease",
    "has_liver_disease",
    "has_psychiatric",
    "has_alcohol_abuse",
    "has_drug_abuse",
    "has_obesity_text",
    "has_pregnancy",
    "has_cancer",
    "has_previous_anaesthesia_complication",
]

CATEGORICAL_FEATURES = [
    "smoking_status",
    "planned_anaesthesia_type",
    "airway_plan",
]

TEXT_FEATURES = [
    "tekst_clean",
    "diagnose_text",
    "sks_text",
    "cave_text",
]

LIST_FEATURES = ["cave", "diagnoseliste", "diagnosenavne"]
ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES

TASK_TYPE = "multiclass"
BINARY_MAPPING = {1: 0, 2: 0, 3: 1}

EXCLUDE_ASA_V = True
EXCLUDE_ASA_IV = True
EXCLUDE_UNDER_18 = True

ALCOHOL_AGE_BINS = [18, 30, 50, 70, 120]
ALCOHOL_AGE_LABELS = ["18-29", "30-49", "50-69", "70+"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15
CV_FOLDS = 3
TUNE_MODELS = False
RUN_SHAP = False
SPLIT_BY_PATIENT = True

# Final optimization focus.
# LightGBM performed best in the baseline comparison, so the next experiment
# only trains LightGBM with stronger regularization to reduce overfitting.
RUN_XGBOOST = False
RUN_EXTRA_TREES = False
RUN_LIGHTGBM = True
USE_VOTING_ENSEMBLE = False

# Controls whether the same patient can contribute multiple training rows.
# Recommended for thesis experiments:
#   "all_notes"          = keep all preoperative notes as separate encounters
#   "patient_asa_unique" = keep max one row per patient per ASA class
# This reduces repeated rows from the same patient, e.g. if DurableKey 121 has
# several ASA 3 notes, only one ASA 3 row is kept. If the same patient also has
# ASA 2, one ASA 2 row is kept as well.
PATIENT_ROW_MODE = "patient_asa_unique"

# Text feature configuration
TEXT_MAX_FEATURES = 4000
DIAG_MAX_FEATURES = 1500
SKS_MAX_FEATURES = 1000
CAVE_MAX_FEATURES = 500


# =============================================================================
# REGEX PATTERNS AND PARSING HELPERS
# =============================================================================

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}

ASA_PATTERNS = [
    r"\bASA[\s:]*([1-5])\b",
    r"\bASA[\s:]*(I{1,3}|IV|V)\b(?!\w)",
]

# Removes ASA label from text features only. Target extraction still uses raw text.
ASA_REMOVE_PATTERN = r"\bASA\s*:?\s*(?:[1-5]|I{1,3}|IV|V)\b"

HOJDE_PATTERN = r"H[øo]jde[\s:]+(\d{2,3}(?:[.,]\d)?)"
VAEGT_PATTERN = r"V[æa]gt[\s:]+(\d{2,3}(?:[.,]\d)?)"
MALLAMPATI_PATTERN = r"Mallampati[\s:]*(?:klasse\s*)?([1-4]|I{1,3}|IV)\b"

BINARY_SECTIONS = [
    ("neuro_psyk", r"Neuro/Psyk\s*-\s*øvrigt"),
    ("respiratorisk", r"Respiratorisk"),
    ("kardiovaskulaert", r"Kardiovaskulært"),
    ("gi_lever_nyre", r"GI/Lever/Nyre"),
    ("endo_andet", r"Endo/Andet"),
    ("bevaegeapparat", r"Bevægeapparat"),
]

ALL_HEADERS = [
    "Anamnese",
    r"Neuro/Psyk\s*-\s*øvrigt",
    "Respiratorisk",
    "Rygestatus",
    "Kardiovaskulært",
    "GI/Lever/Nyre",
    "Endo/Andet",
    "Bevægeapparat",
    "Objektiv undersøgelse",
    "Højde og vægt",
    "Højde",
    "Vægt",
    "Luftvej",
    "Mallampati",
    "Mundåbning",
    "Underbid",
    "Plan for anæstesi",
    "Planlagt anæstesitype",
    "Induktion",
    "Luftvejsplan",
    "Monitorering",
    "Samtykke",
    "ASA",
    "Neurologisk",
    "GCS",
]
NEXT_HEADER_RE = "(?:" + "|".join(ALL_HEADERS) + r")\s*:"

HEIGHT_RANGE = (40, 230)
WEIGHT_RANGE = (2, 300)
BMI_RANGE = (10, 70)


KEYWORD_PATTERNS = {
    "has_hypertension": r"\b(hypertension|forhøjet blodtryk|htn)\b",
    "has_diabetes": r"\b(diabetes|dm1|dm2|type\s*1|type\s*2)\b",
    "has_asthma": r"\b(astma|asthma)\b",
    "has_kol": r"\b(kol|copd)\b",
    "has_sleep_apnea": r"\b(søvnapn[øo]e|sleep\s*apnea|cpap)\b",
    "has_cardiac_disease": r"\b(iskæmisk|ami|myokardieinfarkt|hjertesvigt|atrieflimren|pacemaker|angina|aortastenose|arytmi)\b",
    "has_reflux": r"\b(refluks|reflux|gastro[øo]sofageal)\b",
    "has_renal_disease": r"\b(nyresvigt|dialyse|kronisk\s*nyre|renal)\b",
    "has_liver_disease": r"\b(levercirrose|cirrose|hepatitis|leversvigt)\b",
    "has_psychiatric": r"\b(depression|angst|skizofren|bipolar|psykiatrisk)\b",
    "has_alcohol_abuse": r"\b(abusus\s*spir|alkoholmisbrug|overforbrug|storforbrug|antabus|thiamin)\b",
    "has_drug_abuse": r"\b(stofmisbrug|narkotika|opioidmisbrug|misbrug)\b",
    "has_obesity_text": r"\b(adipositas|overvægt|fedme|obesitas|bmi\s*[>≥]\s*30)\b",
    "has_pregnancy": r"\b(gravid|graviditet|gestationsuge)\b",
    "has_cancer": r"\b(cancer|malign|tumor|karcinom|metastase|kemoterapi)\b",
    "has_previous_anaesthesia_complication": r"\b(tidligere.*anæstesi.*komplikation|svær\s*intubation|vanskelig\s*luftvej|ponv|postoperativ\s*kvalme)\b",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_colname(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def extract_asa(text):
    if not isinstance(text, str):
        return np.nan
    for pat in ASA_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).upper()
            if val.isdigit():
                v = int(val)
                if 1 <= v <= 5:
                    return float(v)
            if val in ROMAN:
                return float(ROMAN[val])
    return np.nan


def remove_asa_from_text(text):
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(ASA_REMOVE_PATTERN, " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_first_numeric(text, pattern, valid_range):
    if not isinstance(text, str):
        return np.nan
    lo, hi = valid_range
    for m in re.finditer(pattern, text, re.IGNORECASE):
        try:
            val = float(m.group(1).replace(",", "."))
            if lo <= val <= hi:
                return val
        except (ValueError, IndexError):
            continue
    return np.nan


def extract_mallampati(text):
    if not isinstance(text, str):
        return np.nan
    m = re.search(MALLAMPATI_PATTERN, text, re.IGNORECASE)
    if not m:
        return np.nan
    val = m.group(1).upper()
    if val.isdigit():
        return float(val)
    return float(ROMAN.get(val, np.nan))


def extract_section_content(text, header_pattern):
    if not isinstance(text, str):
        return None
    pattern = re.compile(
        rf"{header_pattern}\s*:\s*(.*?)(?={NEXT_HEADER_RE}|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def section_to_binary(content):
    if content is None:
        return np.nan
    c = content.strip()
    if not c:
        return np.nan
    norm = re.sub(r"[.\s]", "", c).lower()
    normal_values = {
        "ia",
        "i.a",
        "ingenanmerkninger",
        "ingenanm",
        "ingen",
        "nej",
        "normal",
        "normalt",
    }
    if norm in normal_values:
        return 0
    return 1


def simplify_text_category(value):
    if value is None or pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not s or s in {"ia", "i.a.", "null", "nan"}:
        return np.nan
    return s[:80]


def parse_note(text):
    out = {}
    raw_text = text if isinstance(text, str) else ""

    for col, pat in BINARY_SECTIONS:
        out[col] = section_to_binary(extract_section_content(raw_text, pat))

    out["hoejde"] = extract_first_numeric(raw_text, HOJDE_PATTERN, HEIGHT_RANGE)
    out["vaegt"] = extract_first_numeric(raw_text, VAEGT_PATTERN, WEIGHT_RANGE)
    out["mallampati"] = extract_mallampati(raw_text)
    out["asa_score"] = extract_asa(raw_text)
    out["tekst_clean"] = remove_asa_from_text(raw_text)
    out["n_text_chars"] = len(raw_text)

    planned = extract_section_content(raw_text, r"Planlagt anæstesitype")
    airway = extract_section_content(raw_text, r"Luftvejsplan")
    out["planned_anaesthesia_type"] = simplify_text_category(planned)
    out["airway_plan"] = simplify_text_category(airway)

    text_lower = raw_text.lower()
    for col, pattern in KEYWORD_PATTERNS.items():
        out[col] = int(bool(re.search(pattern, text_lower, flags=re.IGNORECASE)))

    return out


def parse_yyyymmdd(val):
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, pd.Timestamp) or hasattr(val, "year"):
        return pd.Timestamp(val)
    try:
        s = str(int(float(val)))
        if len(s) == 8:
            ts = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            if pd.notna(ts):
                return ts
    except (ValueError, TypeError):
        pass
    return pd.to_datetime(val, errors="coerce", dayfirst=True)


def parse_dato_series(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    sample = s.dropna().head(20).astype(str)
    if len(sample) > 0 and all(len(x.split(".")[0]) == 8 and x.split(".")[0].isdigit() for x in sample):
        return s.apply(parse_yyyymmdd)
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def simplify_smoking(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"null", "nan", "", "aldrig vurderet", "ukendt", "unknown"}:
        return np.nan
    if "hver dag" in s or "nogle dage" in s or "current" in s or "aktiv" in s or s.startswith("ryger") or "dagligt" in s:
        return "Current"
    if "tidlig" in s or "former" in s or s.startswith("ex") or "ophørt" in s:
        return "Former"
    if s == "aldrig" or "never" in s or "aldrig ryger" in s:
        return "Never"
    return np.nan


def parse_cave(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s.upper() in {"NULL", "NAN", "INGEN", ""}:
        return []
    parts = re.split(r"[,;/+]| og ", s)
    return [p.strip().lower() for p in parts if p.strip()]


def calc_age_at(birth, ref):
    age = ref.dt.year - birth.dt.year
    not_yet = (ref.dt.month < birth.dt.month) | ((ref.dt.month == birth.dt.month) & (ref.dt.day < birth.dt.day))
    return (age - not_yet.astype(int)).astype("Int64")


def impute_alcohol_by_age_group(df):
    df = df.copy()
    df["_age_grp"] = pd.cut(
        df["alder"].astype(float),
        bins=ALCOHOL_AGE_BINS,
        labels=ALCOHOL_AGE_LABELS,
        right=False,
        include_lowest=True,
    )

    def fill_with_mode(s):
        m = s.mode()
        return s.fillna(m.iloc[0]) if len(m) > 0 else s

    df["alkohol_per_uge"] = df.groupby("_age_grp", group_keys=False, observed=True)["alkohol_per_uge"].transform(fill_with_mode)
    return df.drop(columns="_age_grp")


def safe_join_list(values):
    if isinstance(values, list):
        return " ".join([str(v).strip().lower().replace(" ", "_") for v in values if str(v).strip()])
    return ""


def sks_prefix_tokens(values):
    if not isinstance(values, list):
        return ""
    tokens = []
    for code in values:
        c = str(code).strip().upper()
        if not c:
            continue
        tokens.append(c)
        if len(c) >= 2:
            tokens.append(c[:2])
        if len(c) >= 3:
            tokens.append(c[:3])
    return " ".join(tokens)


def find_column(df, candidates):
    mapping = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in mapping:
            return mapping[key]
    return None


def build_diagnoselister(notes_df, diagnoser_df):
    diag = diagnoser_df.copy()

    col_key = find_column(diag, ["DurableKey", "durable_key"])
    col_sks = find_column(diag, ["SKS-kode", "SKS kode", "sks_kode"])
    col_diag = find_column(diag, ["Diagnose", "diagnose"])
    col_start = find_column(diag, ["Diagnose start", "diagnose_start"])
    col_slut = find_column(diag, ["Diagnose slut", "diagnose_slut"])

    if col_key is None or col_sks is None:
        raise KeyError("Kunne ikke finde DurableKey eller SKS-kode i Diagnoser sheet.")

    if col_start is None:
        diag["dx_start"] = pd.Timestamp.min
    else:
        diag["dx_start"] = parse_dato_series(diag[col_start])

    if col_slut is None:
        diag["dx_slut"] = pd.Timestamp.max
    else:
        diag["dx_slut"] = parse_dato_series(diag[col_slut]).fillna(pd.Timestamp.max)

    notes_idx = notes_df[["DurableKey", "Ydelsesdato"]].reset_index().rename(columns={"index": "_note_idx"})

    keep_cols = [col_key, col_sks, "dx_start", "dx_slut"]
    if col_diag is not None:
        keep_cols.append(col_diag)

    diag_small = diag[keep_cols].rename(columns={
        col_key: "DurableKey",
        col_sks: "SKS-kode",
        col_diag: "Diagnose" if col_diag is not None else col_diag,
    })

    merged = notes_idx.merge(diag_small, on="DurableKey", how="left")
    mask = (
        merged["dx_start"].notna()
        & (merged["dx_start"] <= merged["Ydelsesdato"])
        & (merged["Ydelsesdato"] <= merged["dx_slut"])
    )
    active = merged.loc[mask].dropna(subset=["SKS-kode"])

    sks_grouped = active.groupby("_note_idx")["SKS-kode"].apply(lambda s: sorted(set(s.astype(str))))

    if "Diagnose" in active.columns:
        diag_grouped = active.groupby("_note_idx")["Diagnose"].apply(lambda s: sorted(set(s.dropna().astype(str))))
    else:
        diag_grouped = pd.Series(dtype=object)

    sks_lists = notes_df.index.to_series().map(sks_grouped).apply(lambda v: v if isinstance(v, list) else [])
    diag_lists = notes_df.index.to_series().map(diag_grouped).apply(lambda v: v if isinstance(v, list) else [])

    return sks_lists, diag_lists


# =============================================================================
# GROUP SPLITTING
# =============================================================================

def make_group_stratified_splits(X, y, groups, test_size=0.20, val_size=0.15, random_state=42):
    """
    Creates train, validation, and test splits with no patient overlap.
    Uses StratifiedGroupKFold as an approximation to the requested proportions.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    n_test_splits = max(2, round(1 / test_size))
    sgkf_test = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(sgkf_test.split(X, y, groups))

    X_trainval = X.iloc[trainval_idx].reset_index(drop=True)
    y_trainval = y.iloc[trainval_idx].reset_index(drop=True)
    groups_trainval = groups.iloc[trainval_idx].reset_index(drop=True)

    relative_val_size = val_size / (1 - test_size)
    n_val_splits = max(2, round(1 / relative_val_size))
    sgkf_val = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=random_state + 1)
    train_idx_rel, val_idx_rel = next(sgkf_val.split(X_trainval, y_trainval, groups_trainval))

    X_train = X_trainval.iloc[train_idx_rel].reset_index(drop=True)
    X_val = X_trainval.iloc[val_idx_rel].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    y_train = y_trainval.iloc[train_idx_rel].reset_index(drop=True)
    y_val = y_trainval.iloc[val_idx_rel].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    groups_train = groups_trainval.iloc[train_idx_rel].reset_index(drop=True)
    groups_val = groups_trainval.iloc[val_idx_rel].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)

    overlap_train_val = set(groups_train).intersection(set(groups_val))
    overlap_train_test = set(groups_train).intersection(set(groups_test))
    overlap_val_test = set(groups_val).intersection(set(groups_test))

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError("Patient overlap detected between train, validation, and test splits.")

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train


# =============================================================================
# MAIN DATA BUILDER
# =============================================================================

def build_note_dataframe(filepath):
    print(f"\n{'=' * 60}\n  Loader: {Path(filepath).name}\n{'=' * 60}")
    xls = pd.ExcelFile(filepath)
    print(f"  Sheets fundet: {xls.sheet_names}")
    print("  → Bruger KUN: Patientinfo, Diagnoser, Anæstesiprætilsynsnotatet")
    print("  → Andre sheets ignoreres pga. label leakage / post-op data\n")

    sheet_map = {n.lower().strip(): n for n in xls.sheet_names}

    def pick(*candidates):
        for c in candidates:
            if c.lower() in sheet_map:
                return sheet_map[c.lower()]
        raise KeyError(f"Ingen af {candidates} fundet. Sheets: {xls.sheet_names}")

    df_info = pd.read_excel(xls, sheet_name=pick("Patientinfo"))
    df_diag = pd.read_excel(xls, sheet_name=pick("Diagnoser"))
    df_note = pd.read_excel(xls, sheet_name=pick("Anæstesiprætilsynsnotatet", "Anæstesipraetilsynsnotatet"))

    # STEP 1: Parse preoperative notes
    print("  → STEP 1: Parser fritekst fra prætilsynsnotater...")
    df_note = df_note.copy()
    df_note["Ydelsesdato"] = df_note["Ydelsesdato"].apply(parse_yyyymmdd)
    n_before = len(df_note)
    df_note = df_note.dropna(subset=["Ydelsesdato"]).reset_index(drop=True)
    if n_before != len(df_note):
        print(f"    ⚠ Fjernet {n_before - len(df_note)} notater uden Ydelsesdato")

    parsed = df_note["Tekst"].apply(parse_note).apply(pd.Series)
    notes = pd.concat(
        [
            df_note[["DurableKey", "Ydelsesdato"]].reset_index(drop=True),
            parsed.reset_index(drop=True),
        ],
        axis=1,
    )
    print(f"    ✓ {len(notes):,} notater parset")
    print(f"      ASA udfyldt: {notes['asa_score'].notna().sum():,} ({notes['asa_score'].notna().mean() * 100:.1f}%)")

    # STEP 2: Patientinfo
    print("  → STEP 2: Join Patientinfo...")
    pi = df_info[["DurableKey", "Birthdate", "Sex", "SmokingStatus", "AlcoholConsumption_X", "CAVE"]].copy()
    pi["fodselsdato"] = parse_dato_series(pi["Birthdate"])
    pi["sex"] = (
        pi["Sex"].astype(str).str.strip().str.lower()
        .map({"mand": 1, "male": 1, "m": 1, "kvinde": 0, "female": 0, "f": 0, "k": 0})
        .astype("Int64")
    )
    pi["smoking_status"] = pi["SmokingStatus"].apply(simplify_smoking)
    pi["alkohol_per_uge"] = pd.to_numeric(pi["AlcoholConsumption_X"].replace({"NULL": np.nan, "null": np.nan}), errors="coerce")
    pi["cave"] = pi["CAVE"].apply(parse_cave)

    pi_keep = (
        pi[["DurableKey", "fodselsdato", "sex", "smoking_status", "alkohol_per_uge", "cave"]]
        .drop_duplicates(subset="DurableKey", keep="last")
        .reset_index(drop=True)
    )
    print(f"    Patientinfo rå: {len(pi):,}, efter dedup: {len(pi_keep):,}")
    print(f"    Smoking: {pi_keep['smoking_status'].value_counts(dropna=False).to_dict()}")

    df = notes.merge(pi_keep, on="DurableKey", how="left")
    df["cave"] = df["cave"].apply(lambda v: v if isinstance(v, list) else [])
    print(f"    ✓ Merged: {len(df):,} notater")

    # STEP 3: Age and BMI
    print("  → STEP 3: Beregn alder + BMI...")
    df["alder"] = calc_age_at(df["fodselsdato"], df["Ydelsesdato"])
    df["bmi"] = df["vaegt"] / (df["hoejde"] / 100) ** 2
    df["bmi"] = df["bmi"].where(df["bmi"].between(*BMI_RANGE))
    print(f"    ✓ Alder range: {df['alder'].min()}–{df['alder'].max()}, BMI udfyldt: {df['bmi'].notna().sum():,}")

    # STEP 4: Alcohol imputation
    print("  → STEP 4: Mode-imputation af alkohol...")
    n_miss_before = df["alkohol_per_uge"].isna().sum()
    df = impute_alcohol_by_age_group(df)
    n_miss_after = df["alkohol_per_uge"].isna().sum()
    print(f"    ✓ Imputeret {n_miss_before - n_miss_after:,} værdier")

    # STEP 5: Diagnosis lists
    print("  → STEP 5: Diagnoseliste pr. notat...")
    df["diagnoseliste"], df["diagnosenavne"] = build_diagnoselister(df, df_diag)
    print(f"    ✓ Median {df['diagnoseliste'].apply(len).median():.0f} diagnoser/notat")

    # STEP 6: Exclusions
    print("  → STEP 6: Eksklusion...")
    n0 = len(df)
    if EXCLUDE_ASA_V:
        df = df[df["asa_score"].fillna(99) < 5]
        print(f"    ✓ ASA V udelukket: {n0 - len(df)}")
        n0 = len(df)
    if EXCLUDE_ASA_IV:
        df = df[df["asa_score"].fillna(99) < 4]
        print(f"    ✓ ASA IV udelukket: {n0 - len(df)}")
        n0 = len(df)
    if EXCLUDE_UNDER_18:
        df = df[df["alder"].fillna(0) >= 18]
        print(f"    ✓ Alder < 18 udelukket: {n0 - len(df)}")
    df = df.reset_index(drop=True)

    # STEP 7: Count and text encodings
    df["n_cave"] = df["cave"].apply(len).astype(int)
    df["n_diagnoser"] = df["diagnoseliste"].apply(len).astype(int)
    df["cave_text"] = df["cave"].apply(safe_join_list)
    df["sks_text"] = df["diagnoseliste"].apply(sks_prefix_tokens)
    df["diagnose_text"] = df["diagnosenavne"].apply(safe_join_list)

    # Convert nullable integer columns to float/int friendly values
    for col in ["alder", "sex"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure all engineered columns exist
    for col in NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES:
        if col not in df.columns:
            df[col] = np.nan if col not in TEXT_FEATURES else ""

    for col in TEXT_FEATURES:
        df[col] = df[col].fillna("").astype(str)

    # STEP 8: Final columns
    df = df.rename(columns={"DurableKey": "durable_key", "Ydelsesdato": "ydelsesdato"})
    final_cols = ["durable_key", "ydelsesdato", "fodselsdato", TARGET_COLUMN] + ALL_FEATURES + LIST_FEATURES
    df = df[[c for c in final_cols if c in df.columns]]

    print(f"\n{'=' * 60}\n  ✓ Færdig")
    print(f"    Rækker: {len(df):,}, Patienter: {df['durable_key'].nunique():,}")
    print(f"    ASA fordeling: {df['asa_score'].value_counts().sort_index().to_dict()}")
    print(f"{'=' * 60}\n")
    return df


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def main():
    print("\n" + "█" * 60)
    print("  ASA SCORE PREDICTION — MASTER'S THESIS PIPELINE")
    print("█" * 60)
    print(f"  Datafil      : {DATA_PATH.name}")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"  Task         : {TASK_TYPE}")
    print(f"  CV folds     : {CV_FOLDS}")
    print(f"  Tune models  : {TUNE_MODELS}")
    print(f"  Patient split: {SPLIT_BY_PATIENT}")
    print(f"  Patient row mode: {PATIENT_ROW_MODE}")
    print()

    # ============= LOAD + FEATURE ENGINEERING =============
    df_raw = build_note_dataframe(DATA_PATH)
    df_raw = df_raw.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    # ============= OPTIONAL PATIENT-LEVEL ROW REDUCTION =============
    # Prevents patients with many repeated notes from dominating the model.
    # This is different from patient-level splitting: splitting avoids leakage,
    # while row reduction avoids overweighting repeated patients.
    if PATIENT_ROW_MODE == "patient_asa_unique":
        n_before = len(df_raw)
        patients_before = df_raw["durable_key"].nunique()

        # Count how many rows each patient has per ASA class before reduction.
        repeat_summary = (
            df_raw.groupby(["durable_key", TARGET_COLUMN])
            .size()
            .reset_index(name="n_rows_same_patient_same_asa")
            .sort_values("n_rows_same_patient_same_asa", ascending=False)
        )
        repeat_summary_path = OUTPUT_DIR / "patient_asa_repeat_summary_before_reduction.csv"
        repeat_summary.to_csv(repeat_summary_path, index=False, encoding="utf-8-sig")

        # Keep one representative row per patient per ASA class.
        df_sorted = df_raw.sort_values(["durable_key", TARGET_COLUMN, "ydelsesdato"])
        kept_index = df_sorted.drop_duplicates(
            subset=["durable_key", TARGET_COLUMN],
            keep="first",
        ).index

        removed_repeated_rows = df_raw.drop(index=kept_index).copy()
        removed_repeated_rows_path = OUTPUT_DIR / "removed_repeated_patient_asa_rows.csv"
        removed_repeated_rows.to_csv(removed_repeated_rows_path, index=False, encoding="utf-8-sig")

        df_raw = (
            df_raw.loc[kept_index]
            .sort_values(["durable_key", TARGET_COLUMN, "ydelsesdato"])
            .reset_index(drop=True)
        )

        print()
        print("=" * 55)
        print("  PATIENT-LEVEL ROW REDUCTION")
        print("=" * 55)
        print("  Mode: patient_asa_unique")
        print(f"  Rows before: {n_before:,}")
        print(f"  Rows after : {len(df_raw):,}")
        print(f"  Removed    : {n_before - len(df_raw):,}")
        print(f"  Patients   : {patients_before:,}")
        print("  Rule       : Max one row per patient per ASA class")
        print(f"  Saved repeat summary: {repeat_summary_path}")
        print(f"  Saved removed repeated rows: {removed_repeated_rows_path}")
        print()
        print("  Top repeated patient/ASA combinations before reduction:")
        print(repeat_summary.head(10).to_string(index=False))

    elif PATIENT_ROW_MODE == "all_notes":
        print()
        print("  Patient row mode: keeping all notes as separate encounters")
    else:
        raise ValueError("PATIENT_ROW_MODE must be either 'all_notes' or 'patient_asa_unique'")

    # ============= EDA =============
    print("=" * 55, "\n  EXPLORATORY DATA ANALYSIS\n", "=" * 55, sep="")
    print(f"\nDataset : {df_raw.shape[0]:,} notater × {df_raw.shape[1]} kolonner")
    print(f"Patienter unikke: {df_raw['durable_key'].nunique():,}")

    target_counts = df_raw[TARGET_COLUMN].value_counts().sort_index()
    target_pct = (target_counts / len(df_raw) * 100).round(1)
    print(f"\nTarget '{TARGET_COLUMN}':")
    print(pd.DataFrame({"Count": target_counts, "%": target_pct}).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    target_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Class counts")
    axes[0].set_xlabel("ASA")
    axes[0].tick_params(axis="x", rotation=0)
    axes[1].pie(
        target_counts,
        labels=[f"ASA {int(c)}" for c in target_counts.index],
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Blues_d", len(target_counts)),
    )
    axes[1].set_title("Class proportions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 01_target_distribution.png")

    # ============= TARGET ENCODING =============
    print("\n" + "=" * 55, "\n  TARGET ENCODING\n", "=" * 55, sep="")
    df = df_raw.copy().dropna(subset=[TARGET_COLUMN])

    if TASK_TYPE == "binary":
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(BINARY_MAPPING).astype(int)
        class_names = ["Low risk", "High risk"]
    else:
        unique_classes = sorted(df[TARGET_COLUMN].unique())
        class_map = {c: i for i, c in enumerate(unique_classes)}
        inverse_class_map = {i: c for c, i in class_map.items()}
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(class_map).astype(int)
        class_names = [f"ASA {int(c)}" for c in unique_classes]
        print(f"  Encoding: {class_map}")

    n_classes = df[TARGET_COLUMN].nunique()
    print(f"  Classes: {class_names}, Total: {len(df):,}")

    # ============= FEATURE ASSIGNMENT =============
    num_features = [f for f in NUMERICAL_FEATURES if f in df.columns]
    bin_features = [f for f in BINARY_FEATURES if f in df.columns]
    cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    text_features = [f for f in TEXT_FEATURES if f in df.columns]

    numeric_plus_binary = num_features + bin_features
    all_features = numeric_plus_binary + cat_features + text_features

    print(f"\n  Numerical: {len(num_features)}")
    print(f"  Binary: {len(bin_features)}")
    print(f"  Categorical: {len(cat_features)}")
    print(f"  Text: {len(text_features)}")
    print(f"  Total feature columns before vectorization: {len(all_features)}")

    X = df[all_features].copy()
    y = df[TARGET_COLUMN].copy()
    groups = df["durable_key"].copy()

    # ============= EXPORT FIRST 10 ROWS FOR REPORT =============
    # Important: df[TARGET_COLUMN] has already been encoded here:
    #   0 = ASA 1, 1 = ASA 2, 2 = ASA 3
    # Therefore, we add asa_original for readability in the thesis/report.
    preview_cols = [
        "durable_key",
        "ydelsesdato",
        "alder",
        "sex",
        "smoking_status",
        "alkohol_per_uge",
        "hoejde",
        "vaegt",
        "bmi",
        "cave",
        "cave_text",
        "diagnoseliste",
        "diagnosenavne",
        "sks_text",
        "planned_anaesthesia_type",
        "airway_plan",
        "neuro_psyk",
        "respiratorisk",
        "kardiovaskulaert",
        "gi_lever_nyre",
        "endo_andet",
        "bevaegeapparat",
        "has_hypertension",
        "has_diabetes",
        "has_asthma",
        "has_kol",
        "has_sleep_apnea",
        "has_cardiac_disease",
        "has_alcohol_abuse",
        "has_obesity_text",
        "tekst_clean",
        TARGET_COLUMN,
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    training_preview = df[preview_cols].head(10).copy()

    if TASK_TYPE == "multiclass":
        # Make the Excel preview look clinically correct:
        #   asa_score = real ASA score shown in the report, e.g. 1, 2, 3
        #   asa_encoded = internal model label, e.g. 0, 1, 2
        training_preview["asa_encoded"] = training_preview[TARGET_COLUMN]
        training_preview[TARGET_COLUMN] = training_preview["asa_encoded"].map(inverse_class_map)
        ordered_cols = [c for c in training_preview.columns if c != "asa_encoded"]
        training_preview = training_preview[ordered_cols + ["asa_encoded"]]

    # Shorten very long text/list columns so Excel is readable for the report.
    def shorten_for_preview(value, max_len=250):
        if isinstance(value, list):
            value = "; ".join([str(v) for v in value])
        value = "" if pd.isna(value) else str(value)
        return value[:max_len] + "..." if len(value) > max_len else value

    for col in ["tekst_clean", "diagnosenavne", "diagnoseliste", "cave", "sks_text", "cave_text"]:
        if col in training_preview.columns:
            training_preview[col] = training_preview[col].apply(shorten_for_preview)

    print("\n" + "=" * 80)
    print("  FIRST 10 ROWS OF FINAL TRAINING DATAFRAME")
    print("  Note: asa_score is the real ASA class; asa_encoded is the internal model label.")
    print("=" * 80)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 120)
    print(training_preview.to_string(index=False))

    preview_csv_path = OUTPUT_DIR / "training_dataframe_first_10_rows.csv"
    preview_excel_path = OUTPUT_DIR / "training_dataframe_first_10_rows.xlsx"

    training_preview.to_csv(preview_csv_path, index=False, encoding="utf-8-sig")

    try:
        training_preview.to_excel(preview_excel_path, index=False)
        print(f"\n  ✓ Saved CSV preview to: {preview_csv_path}")
        print(f"  ✓ Saved Excel preview to: {preview_excel_path}")
    except PermissionError:
        # This usually happens if the Excel file is already open in Excel/PyCharm.
        # Save a timestamped copy instead, so the pipeline can continue.
        timestamp_preview = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_excel_path_alt = OUTPUT_DIR / f"training_dataframe_first_10_rows_{timestamp_preview}.xlsx"
        training_preview.to_excel(preview_excel_path_alt, index=False)
        print(f"\n  ✓ Saved CSV preview to: {preview_csv_path}")
        print(f"  ⚠ Could not overwrite Excel preview because the file is open or locked:")
        print(f"    {preview_excel_path}")
        print(f"  ✓ Saved Excel preview as timestamped copy instead:")
        print(f"    {preview_excel_path_alt}")

    # ============= SPLIT =============
    print("\n" + "=" * 55, "\n  TRAIN/VAL/TEST SPLIT\n", "=" * 55, sep="")

    if SPLIT_BY_PATIENT:
        X_train, X_val, X_test, y_train, y_val, y_test, groups_train = make_group_stratified_splits(
            X,
            y,
            groups,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_STATE,
        )
        cv_strategy = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_groups = groups_train
    else:
        from sklearn.model_selection import train_test_split

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        val_frac = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_frac,
            random_state=RANDOM_STATE,
            stratify=y_trainval,
        )
        cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_groups = None

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print("\n  Train target distribution:")
    print(y_train.value_counts(normalize=True).sort_index().round(3).to_string())
    print("  Val target distribution:")
    print(y_val.value_counts(normalize=True).sort_index().round(3).to_string())
    print("  Test target distribution:")
    print(y_test.value_counts(normalize=True).sort_index().round(3).to_string())

    # ============= PREPROCESSING =============
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20)),
        ]
    )

    # Danish/clinical text. Token pattern keeps words, SKS-like codes, and short tokens such as ia.
    note_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        max_features=TEXT_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        token_pattern=r"(?u)\b[\wæøåÆØÅ.-]{2,}\b",
    )

    diag_tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=DIAG_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=3,
        token_pattern=r"(?u)\b[\wæøåÆØÅ.-]{2,}\b",
    )

    sks_tfidf = TfidfVectorizer(
        lowercase=False,
        max_features=SKS_MAX_FEATURES,
        ngram_range=(1, 1),
        min_df=2,
        token_pattern=r"(?u)\b[A-Z0-9]{2,}\b",
    )

    cave_tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=CAVE_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        token_pattern=r"(?u)\b[\wæøåÆØÅ.-]{2,}\b",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numeric_plus_binary),
            ("cat", categorical_transformer, cat_features),
            ("text", note_tfidf, "tekst_clean"),
            ("diag_text", diag_tfidf, "diagnose_text"),
            ("sks", sks_tfidf, "sks_text"),
            ("cave", cave_tfidf, "cave_text"),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    # ============= MODELS =============
    print("\n" + "=" * 55, "\n  MODEL DEFINITIONS\n", "=" * 55, sep="")

    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=18,
                    min_samples_split=30,
                    min_samples_leaf=15,
                    max_features="sqrt",
                    max_samples=0.75,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    et_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=600,
                    min_samples_split=8,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {}

    if RUN_EXTRA_TREES:
        models["Extra Trees"] = et_pipeline

    if RUN_XGBOOST and XGBOOST_AVAILABLE:
        xgb_params = dict(
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss" if TASK_TYPE == "multiclass" else "logloss",
        )
        if TASK_TYPE == "multiclass":
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = n_classes
        models["XGBoost"] = Pipeline([("preprocessor", preprocessor), ("classifier", XGBClassifier(**xgb_params))])

    if RUN_LIGHTGBM and LIGHTGBM_AVAILABLE:
        models["LightGBM"] = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LGBMClassifier(
                        # More conservative LightGBM than the baseline model.
                        # Goal: reduce train-validation gap while keeping F1 macro high.
                        n_estimators=600,
                        learning_rate=0.03,
                        num_leaves=31,
                        max_depth=8,
                        min_child_samples=70,
                        subsample=0.75,
                        subsample_freq=1,
                        colsample_bytree=0.75,
                        reg_alpha=0.5,
                        reg_lambda=4.0,
                        min_split_gain=0.05,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    print(f"  Modeller: {list(models.keys())}")

    # ============= CROSS-VALIDATION =============
    print("\n" + "=" * 55, f"\n  {CV_FOLDS}-FOLD CROSS-VALIDATION\n", "=" * 55, sep="")
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"}

    cv_results_summary = {}
    for model_name, pipeline in models.items():
        print(f"\n  Training {model_name}...")
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_strategy,
            groups=cv_groups,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
        )
        cv_results_summary[model_name] = {
            metric: {
                "train_mean": cv_scores[f"train_{metric}"].mean(),
                "test_mean": cv_scores[f"test_{metric}"].mean(),
                "test_std": cv_scores[f"test_{metric}"].std(),
            }
            for metric in scoring
        }
        f1_t = cv_results_summary[model_name]["f1_macro"]["train_mean"]
        f1_v = cv_results_summary[model_name]["f1_macro"]["test_mean"]
        gap = f1_t - f1_v
        flag = "⚠ OVERFIT" if gap > 0.10 else ("moderate overfit" if gap > 0.05 else "ok")
        print(f"    F1 train: {f1_t:.3f} | F1 val: {f1_v:.3f} | gap: {gap:+.3f} ({flag})")

    rows = []
    for name, results in cv_results_summary.items():
        row = {"Model": name}
        for metric, vals in results.items():
            row[metric] = f"{vals['test_mean']:.3f} ± {vals['test_std']:.3f}"
        row["Train-Val gap (F1m)"] = f"{results['f1_macro']['train_mean'] - results['f1_macro']['test_mean']:+.3f}"
        rows.append(row)
    cv_summary_df = pd.DataFrame(rows).set_index("Model")
    print("\n  Summary:\n" + cv_summary_df.to_string())

    # ============= LIGHTGBM TUNING =============
    # Keep TUNE_MODELS = False for the first regularized run.
    # If the regularized LightGBM looks promising, set TUNE_MODELS = True later.
    if TUNE_MODELS and "LightGBM" in models:
        print()
        print("=" * 55)
        print("  LIGHTGBM HYPERPARAMETER TUNING")
        print("=" * 55)

        lgbm_tuned = RandomizedSearchCV(
            models["LightGBM"],
            param_distributions={
                "classifier__n_estimators": [400, 600, 800],
                "classifier__learning_rate": [0.02, 0.03, 0.04],
                "classifier__num_leaves": [15, 31, 47],
                "classifier__max_depth": [5, 7, 8, 10],
                "classifier__min_child_samples": [50, 70, 100, 150],
                "classifier__subsample": [0.65, 0.75, 0.85],
                "classifier__colsample_bytree": [0.65, 0.75, 0.85],
                "classifier__reg_alpha": [0.2, 0.5, 1.0],
                "classifier__reg_lambda": [2.0, 4.0, 6.0],
                "classifier__min_split_gain": [0.0, 0.05, 0.1],
            },
            n_iter=20,
            cv=cv_strategy,
            scoring="f1_macro",
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=2,
        )

        if cv_groups is not None:
            lgbm_tuned.fit(X_train, y_train, groups=cv_groups)
        else:
            lgbm_tuned.fit(X_train, y_train)

        print(f"    Best params: {lgbm_tuned.best_params_}")
        print(f"    Best CV F1 macro: {lgbm_tuned.best_score_:.4f}")
        models["LightGBM (tuned)"] = lgbm_tuned.best_estimator_

    # ============= FIT ALL ON FULL TRAIN =============
    print("\n" + "=" * 55, "\n  FITTING ON FULL TRAIN\n", "=" * 55, sep="")
    fitted_models = {}
    for name, pipeline in models.items():
        if "(tuned)" in name:
            fitted_models[name] = pipeline
            print(f"  ✓ {name} already fitted")
            continue
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        print(f"  ✓ {name} fitted")

    # Optional voting ensemble. Disabled by default to keep runtime down.
    if USE_VOTING_ENSEMBLE:
        print()
        print("  → Bygger Voting Ensemble...")
        voting_estimators = []
        for k in ["Random Forest", "Extra Trees", "LightGBM"]:
            if k in fitted_models:
                safe_name = k.lower().replace(" ", "_")
                voting_estimators.append((safe_name, fitted_models[k]))
        if len(voting_estimators) >= 2:
            voting_clf = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=-1)
            voting_clf.fit(X_train, y_train)
            fitted_models["Voting Ensemble"] = voting_clf
            print(f"    ✓ Voting Ensemble ({len(voting_estimators)} modeller)")
        else:
            print("    Ikke nok modeller til ensemble")

    # ============= VALIDATION =============
    print("\n" + "=" * 55, "\n  VALIDATION SET EVALUATION\n", "=" * 55, sep="")
    avg_strategy = "binary" if TASK_TYPE == "binary" else "macro"
    val_results = {}
    for name, pipeline in fitted_models.items():
        y_pred = pipeline.predict(X_val)
        result = {
            "Accuracy": round(accuracy_score(y_val, y_pred), 4),
            "Precision(macro)": round(precision_score(y_val, y_pred, average=avg_strategy, zero_division=0), 4),
            "Recall(macro)": round(recall_score(y_val, y_pred, average=avg_strategy, zero_division=0), 4),
            "F1(macro)": round(f1_score(y_val, y_pred, average=avg_strategy, zero_division=0), 4),
            "F1(weighted)": round(f1_score(y_val, y_pred, average="weighted", zero_division=0), 4),
        }
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_val)
                if TASK_TYPE == "binary":
                    result["ROC-AUC"] = round(roc_auc_score(y_val, y_proba[:, 1]), 4)
                else:
                    result["ROC-AUC"] = round(roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro"), 4)
            except Exception:
                result["ROC-AUC"] = None
        val_results[name] = result

    val_results_df = pd.DataFrame(val_results).T.sort_values("F1(macro)", ascending=False)
    print("\n" + val_results_df.to_string())

    best_model_name = val_results_df["F1(macro)"].astype(float).idxmax()
    best_model = fitted_models[best_model_name]
    print(f"\n  ★ Best model: {best_model_name}")

    # ============= TEST EVALUATION =============
    print("\n" + "=" * 55, f"\n  TEST SET EVALUATION: {best_model_name}\n", "=" * 55, sep="")
    y_pred_test = best_model.predict(X_test)
    y_proba_test = None
    if hasattr(best_model, "predict_proba"):
        try:
            y_proba_test = best_model.predict_proba(X_test)
        except Exception:
            pass

    print("\n" + classification_report(y_test, y_pred_test, target_names=class_names, digits=3, zero_division=0))

    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1m = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    test_f1w = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  F1 macro : {test_f1m:.4f}")
    print(f"  F1 weight: {test_f1w:.4f}")

    auc_score = None
    if y_proba_test is not None:
        try:
            if TASK_TYPE == "binary":
                auc_score = roc_auc_score(y_test, y_proba_test[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_proba_test, multi_class="ovr", average="macro")
            print(f"  ROC-AUC  : {auc_score:.4f}")
        except Exception as e:
            print(f"  ROC-AUC  : could not compute ({e})")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_test)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✓ Saved: 06_confusion_matrix.png")

    # Save classification report
    report_dict = classification_report(y_test, y_pred_test, target_names=class_names, digits=4, zero_division=0, output_dict=True)

    # ============= FEATURE IMPORTANCE =============
    print("\n" + "=" * 55, "\n  FEATURE IMPORTANCE\n", "=" * 55, sep="")
    inner = best_model.estimators_[0][1] if isinstance(best_model, VotingClassifier) else best_model

    imp_series = None
    try:
        fitted_pp = inner.named_steps["preprocessor"]
        feat_names = fitted_pp.get_feature_names_out()
        clf = inner.named_steps["classifier"]

        if hasattr(clf, "feature_importances_"):
            imp_series = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
        elif hasattr(clf, "coef_"):
            coef = np.abs(clf.coef_)
            if coef.ndim == 2:
                coef = coef.mean(axis=0)
            imp_series = pd.Series(coef, index=feat_names).sort_values(ascending=False)

        if imp_series is not None:
            top_n = min(30, len(imp_series))
            fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.30)))
            imp_series.head(top_n).sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
            ax.set_xlabel("Importance")
            ax.set_title(f"Top {top_n} Features — {best_model_name}", style="italic", color="grey")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "08_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✓ Saved: 08_feature_importance.png")
            print(f"\n  Top 20:\n{imp_series.head(20).to_string()}")
        else:
            print("  Feature importance not available for this model.")
    except Exception as e:
        print(f"  Feature importance skipped: {e}")

    # ============= EXPORT =============
    print("\n" + "=" * 55, "\n  EXPORTING RESULTS\n", "=" * 55, sep="")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    val_results_df.to_csv(OUTPUT_DIR / f"validation_results_{timestamp}.csv")
    cv_summary_df.to_csv(OUTPUT_DIR / f"cv_results_{timestamp}.csv")

    with open(OUTPUT_DIR / f"classification_report_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    test_metrics = {
        "model": best_model_name,
        "task_type": TASK_TYPE,
        "n_classes": int(n_classes),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1m, 4),
        "test_f1_weighted": round(test_f1w, 4),
        "test_roc_auc": round(auc_score, 4) if auc_score else None,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "features_used": all_features,
        "split_by_patient": SPLIT_BY_PATIENT,
        "patient_row_mode": PATIENT_ROW_MODE,
        "timestamp": timestamp,
    }
    with open(OUTPUT_DIR / f"test_metrics_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    if imp_series is not None:
        imp_series.to_csv(OUTPUT_DIR / f"feature_importance_{timestamp}.csv", header=["importance"])

    model_filename = f"asa_model_{best_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.joblib"
    joblib.dump(best_model, OUTPUT_DIR / model_filename)
    print(f"  ✓ Saved model: {model_filename}")

    metadata = {
        "model_name": best_model_name,
        "task_type": TASK_TYPE,
        "class_names": class_names,
        "all_features": all_features,
        "text_features": text_features,
        "important_note": "ASA was extracted as target but removed from text_clean before training.",
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1m, 4),
        "test_roc_auc": round(auc_score, 4) if auc_score else None,
        "data_sources": ["Patientinfo", "Diagnoser", "Anæstesiprætilsynsnotatet"],
        "split_by_patient": SPLIT_BY_PATIENT,
        "patient_row_mode": PATIENT_ROW_MODE,
        "trained_on": timestamp,
    }
    with open(OUTPUT_DIR / f"model_metadata_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ============= FINAL =============
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Best model : {best_model_name}")
    print(f"  Accuracy   : {test_acc:.4f}")
    print(f"  F1 macro   : {test_f1m:.4f}")
    if auc_score:
        print(f"  ROC-AUC    : {auc_score:.4f}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("█" * 60)


if __name__ == "__main__":
    main()
