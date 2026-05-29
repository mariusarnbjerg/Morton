"""
================================================================================
  ASA SCORE PREDICTION — SIMPLIFIED TRAINING PIPELINE
================================================================================

  Features:
    Structured (7):  age, sex, bmi, alcohol, smoking_status, n_allergies, n_diagnoses
    Text (4, TF-IDF): anamnese, diagnoses, sks_codes, cave

  Dataset columns expected:
    DurableKey, Age, Sex, BMI, AlcoholConsumption_X, SmokingStatus_Simplified,
    NumberOfAllergies, NumberOfDiagnoses, Anamnese, Diagnoses, SKS_codes, CAVE,
    ASA Score

  Usage:
    Run this file directly in your IDE or terminal.
================================================================================
"""

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

from scipy import sparse

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15
CV_FOLDS = 10

# TF-IDF limits
ANAMNESE_MAX_FEATURES = 4000
DIAG_MAX_FEATURES = 1500
SKS_MAX_FEATURES = 1000
CAVE_MAX_FEATURES = 500

# ASA classes to keep (drop rare ASA 4-6)
KEEP_ASA = [1, 2, 3]

# Paths
PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "Dataset" / "Final_raw_data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "advanced_ml_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# LEAKAGE PREVENTION — strip ASA mentions and plan sections from anamnese
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
    """Remove plan sections and ASA mentions to prevent label leakage."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Truncate at plan section
    match = _PLAN_SPLIT.search(text)
    if match:
        text = text[:match.start()]
    else:
        backup = _BACKUP_PLAN.search(text)
        if backup:
            text = text[:backup.start()]

    # Strip ASA mentions
    for pat in _ASA_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# SAFE TF-IDF (handles empty vocabularies in small CV folds)
# =============================================================================

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
# DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print(f"  Loading: {Path(filepath).name}")
    print(f"{'=' * 60}")

    df = pd.read_excel(filepath)
    print(f"  Raw rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}\n")

    # --- Normalize NULL strings to actual NaN ---
    df.replace("NULL", np.nan, inplace=True)
    df.replace("null", np.nan, inplace=True)

    # --- ASA target ---
    df["asa_score"] = pd.to_numeric(df["ASA Score"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["asa_score"]).reset_index(drop=True)
    print(f"  Dropped {n_before - len(df)} rows without ASA score")

    df["asa_score"] = df["asa_score"].astype(int)
    n_before = len(df)
    df = df[df["asa_score"].isin(KEEP_ASA)].reset_index(drop=True)
    print(f"  Dropped {n_before - len(df)} rows with ASA outside {KEEP_ASA}")

    # --- Structured features ---
    df["age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["sex"] = df["Sex"].map({"Mand": 1, "Kvinde": 0}).astype("Int64")
    df["bmi"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["alcohol"] = pd.to_numeric(df["AlcoholConsumption_X"], errors="coerce")
    df["smoking_status"] = df["SmokingStatus_Simplified"].fillna("Unknown").astype(str)
    df["n_allergies"] = pd.to_numeric(df["NumberOfAllergies"], errors="coerce").fillna(0).astype(int)
    df["n_diagnoses"] = pd.to_numeric(df["NumberOfDiagnoses"], errors="coerce").fillna(0).astype(int)

    # --- Text features ---
    df["anamnese_clean"] = df["Anamnese"].apply(clean_anamnese)
    df["diagnoses_text"] = df["Diagnoses"].fillna("").astype(str)
    df["sks_text"] = df["SKS_codes"].fillna("").astype(str)
    df["cave_text"] = df["CAVE"].fillna("").astype(str)

    # --- Patient key ---
    df["patient_id"] = df["DurableKey"]

    # --- Deduplicate: one row per patient per ASA score ---
    n_before = len(df)
    df = df.sort_values(["patient_id", "asa_score"]).drop_duplicates(
        subset=["patient_id", "asa_score"], keep="first"
    ).reset_index(drop=True)
    print(f"  Deduplicated: {n_before:,} → {len(df):,} rows")

    # --- Exclude under 18 ---
    n_before = len(df)
    df = df[df["age"].fillna(0) >= 18].reset_index(drop=True)
    print(f"  Dropped {n_before - len(df)} rows with age < 18")

    # --- Summary ---
    print(f"\n  Final dataset: {len(df):,} rows, {df['patient_id'].nunique():,} patients")
    print(f"  ASA distribution: {df['asa_score'].value_counts().sort_index().to_dict()}")
    print(f"{'=' * 60}\n")

    return df


# =============================================================================
# TRAIN / VAL / TEST SPLIT (by patient)
# =============================================================================

def split_by_patient(X, y, groups, test_size=0.20, val_size=0.15, random_state=42):
    groups = pd.Series(groups).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Train+Val / Test split
    sgkf = StratifiedGroupKFold(
        n_splits=max(2, round(1 / test_size)),
        shuffle=True, random_state=random_state,
    )
    tv_idx, test_idx = next(sgkf.split(X, y, groups))

    X_tv, X_test = X.iloc[tv_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_tv, y_test = y.iloc[tv_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    g_tv = groups.iloc[tv_idx].reset_index(drop=True)

    # Train / Val split
    sgkf2 = StratifiedGroupKFold(
        n_splits=max(2, round(1 / (val_size / (1 - test_size)))),
        shuffle=True, random_state=random_state + 1,
    )
    tr_idx, val_idx = next(sgkf2.split(X_tv, y_tv, g_tv))

    X_train = X_tv.iloc[tr_idx].reset_index(drop=True)
    X_val = X_tv.iloc[val_idx].reset_index(drop=True)
    y_train = y_tv.iloc[tr_idx].reset_index(drop=True)
    y_val = y_tv.iloc[val_idx].reset_index(drop=True)
    g_train = g_tv.iloc[tr_idx].reset_index(drop=True)

    # Verify no patient leakage
    g_test = groups.iloc[test_idx].reset_index(drop=True)
    g_val = g_tv.iloc[val_idx].reset_index(drop=True)
    assert not (set(g_train) & set(g_val)), "Patient leakage: train/val overlap"
    assert not (set(g_train) & set(g_test)), "Patient leakage: train/test overlap"
    assert not (set(g_val) & set(g_test)), "Patient leakage: val/test overlap"

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    output_dir = OUTPUT_DIR

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    df = load_data(str(DATA_PATH))

    NUMERICAL = ["age", "bmi", "alcohol", "n_allergies", "n_diagnoses"]
    BINARY = ["sex"]
    CATEGORICAL = ["smoking_status"]
    TEXT = ["anamnese_clean", "diagnoses_text", "sks_text", "cave_text"]
    ALL_FEATURES = NUMERICAL + BINARY + CATEGORICAL + TEXT

    X = df[ALL_FEATURES].copy()
    y = df["asa_score"].copy()
    groups = df["patient_id"].copy()

    # Encode target: ASA 1→0, ASA 2→1, ASA 3→2
    class_map = {c: i for i, c in enumerate(sorted(y.unique()))}
    inverse_map = {i: c for c, i in class_map.items()}
    class_names = [f"ASA {int(c)}" for c in sorted(y.unique())]
    y = y.map(class_map).astype(int)

    print(f"  Features: {len(ALL_FEATURES)} ({len(NUMERICAL)} num, {len(BINARY)} bin, "
          f"{len(CATEGORICAL)} cat, {len(TEXT)} text)")
    print(f"  Classes: {class_names}")
    print(f"  Encoding: {class_map}\n")

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, g_train = split_by_patient(
        X, y, groups, TEST_SIZE, VAL_SIZE, RANDOM_STATE
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = ", ".join(
            f"ASA {int(inverse_map[k])}: {v:.1%}"
            for k, v in yy.value_counts(normalize=True).sort_index().items()
        )
        print(f"  {name}: {dist}")

    # ------------------------------------------------------------------
    # 3. Preprocessor
    # ------------------------------------------------------------------
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL + BINARY),
            ("cat", cat_transformer, CATEGORICAL),
            ("anamnese", SafeTfidfVectorizer(
                max_features=ANAMNESE_MAX_FEATURES, ngram_range=(1, 2),
                min_df=5, max_df=0.95, lowercase=True,
                token_pattern=r"(?u)\b[\wæøåÆØÅ.-]{2,}\b",
            ), "anamnese_clean"),
            ("diag", SafeTfidfVectorizer(
                max_features=DIAG_MAX_FEATURES, ngram_range=(1, 2),
                min_df=3, lowercase=True,
                token_pattern=r"(?u)\b[\wæøåÆØÅ.-]{2,}\b",
            ), "diagnoses_text"),
            ("sks", SafeTfidfVectorizer(
                max_features=SKS_MAX_FEATURES, ngram_range=(1, 1),
                min_df=2, lowercase=False,
                token_pattern=r"(?u)\b[A-Z0-9]{2,}\b",
            ), "sks_text"),
            ("cave", SafeTfidfVectorizer(
                max_features=CAVE_MAX_FEATURES, ngram_range=(1, 2),
                min_df=1, lowercase=True,
                token_pattern=r"(?u)\b[\wæøåÆØÅ.-]{2,}\b",
            ), "cave_text"),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    # ------------------------------------------------------------------
    # 4. Model experiments
    # ------------------------------------------------------------------
    configs = [
        {
            "name": "LightGBM_balanced",
            "class_weight": "balanced",
            "n_estimators": 600, "learning_rate": 0.03,
            "num_leaves": 31, "max_depth": 8,
            "min_child_samples": 70, "subsample": 0.75,
            "colsample_bytree": 0.75, "reg_alpha": 0.5,
            "reg_lambda": 4.0, "min_split_gain": 0.05,
        },
        {
            "name": "LightGBM_no_weight",
            "class_weight": None,
            "n_estimators": 600, "learning_rate": 0.03,
            "num_leaves": 31, "max_depth": 8,
            "min_child_samples": 70, "subsample": 0.75,
            "colsample_bytree": 0.75, "reg_alpha": 0.5,
            "reg_lambda": 4.0, "min_split_gain": 0.05,
        },
        {
            "name": "LightGBM_less_reg",
            "class_weight": "balanced",
            "n_estimators": 800, "learning_rate": 0.025,
            "num_leaves": 45, "max_depth": 9,
            "min_child_samples": 50, "subsample": 0.80,
            "colsample_bytree": 0.80, "reg_alpha": 0.2,
            "reg_lambda": 2.0, "min_split_gain": 0.02,
        },
    ]

    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"}
    cv_strategy = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_val_f1 = -1
    best_pipe = None
    best_name = None
    results = []

    print(f"\n{'=' * 60}")
    print("  MODEL EXPERIMENTS")
    print(f"{'=' * 60}")

    for cfg in configs:
        print(f"\n  → {cfg['name']}")

        clf = LGBMClassifier(
            n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"], max_depth=cfg["max_depth"],
            min_child_samples=cfg["min_child_samples"], subsample=cfg["subsample"],
            subsample_freq=1, colsample_bytree=cfg["colsample_bytree"],
            reg_alpha=cfg["reg_alpha"], reg_lambda=cfg["reg_lambda"],
            min_split_gain=cfg["min_split_gain"], class_weight=cfg["class_weight"],
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )

        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", clf),
        ])

        # Cross-validation
        cv = cross_validate(
            pipe, X_train, y_train, cv=cv_strategy, groups=g_train,
            scoring=scoring, return_train_score=True, n_jobs=1,
        )

        # Fit on full training set and evaluate on validation
        pipe.fit(X_train, y_train)
        y_pred_val = pipe.predict(X_val)
        y_proba_val = pipe.predict_proba(X_val)

        val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, y_pred_val)
        val_recall = recall_score(y_val, y_pred_val, average="macro", zero_division=0)
        val_auc = roc_auc_score(y_val, y_proba_val, multi_class="ovr", average="macro")
        cv_f1 = cv["test_f1_macro"].mean()
        cv_f1_std = cv["test_f1_macro"].std()
        gap = cv["train_f1_macro"].mean() - cv_f1

        print(f"    CV F1: {cv_f1:.4f} ± {cv_f1_std:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Recall: {val_recall:.4f} | Val AUC: {val_auc:.4f} | Gap: {gap:+.3f}")

        results.append({"name": cfg["name"], "cv_f1": cv_f1, "val_f1": val_f1,
                         "val_acc": val_acc, "val_recall": val_recall, "val_auc": val_auc, "gap": gap})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_pipe = pipe
            best_name = cfg["name"]

    # ------------------------------------------------------------------
    # 5. Test set evaluation (best model only)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: {best_name}")
    print(f"{'=' * 60}")

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    test_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    print(f"\n{classification_report(y_test, y_pred, target_names=class_names, digits=3)}")
    print(f"  Accuracy    : {test_acc:.4f}")
    print(f"  F1 macro    : {test_f1:.4f}")
    print(f"  Recall macro: {test_recall:.4f}")
    print(f"  AUC macro   : {test_auc:.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=class_names,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance
    try:
        feat_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
        importances = best_pipe.named_steps["classifier"].feature_importances_
        imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

        top_n = min(30, len(imp))
        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.3)))
        imp.head(top_n).sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Features")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Top 15 features:\n{imp.head(15).to_string()}")
    except Exception as e:
        print(f"  Feature importance skipped: {e}")

    # ------------------------------------------------------------------
    # 6. Save model and metadata
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_file = f"asa_model_{best_name}_{timestamp}.joblib"
    joblib.dump(best_pipe, output_dir / model_file)

    metadata = {
        "model_name": best_name,
        "class_names": class_names,
        "class_map": {str(k): int(v) for k, v in class_map.items()},
        "features": {
            "numerical": NUMERICAL,
            "binary": BINARY,
            "categorical": CATEGORICAL,
            "text": TEXT,
        },
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "test_recall_macro": round(test_recall, 4),
        "test_roc_auc": round(test_auc, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "split_by_patient": True,
        "timestamp": timestamp,
    }

    with open(output_dir / f"metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ Model saved: {model_file}")
    print(f"  ✓ Metadata saved: metadata_{timestamp}.json")
    print(f"  ✓ Plots saved to: {output_dir}/")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()