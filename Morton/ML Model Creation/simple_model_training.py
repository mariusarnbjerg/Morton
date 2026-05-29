"""
================================================================================
  ASA SCORE PREDICTION — SIMPLIFIED ML PIPELINE
  Preoperative Anesthesiology AI — Proof of Concept
================================================================================

  PURPOSE:
    Supervised classification of ASA physical status scores (1, 2, 3) from
    structured preoperative patient data. Designed to align with features
    extractable from a patient chatbot conversation.

  FEATURE PHILOSOPHY:
    Only features that can be reliably obtained from BOTH the clinical training
    data AND a patient chatbot transcript are included. This eliminates the
    domain gap between training and inference.

    Training features come from:
      - Structured hospital data (age, sex, smoking, alcohol, allergies, diagnoses)
      - Regex extraction from Anamnese (height, weight, organ system binary flags)

    Inference features come from:
      - LLM extraction from chatbot transcript → same structured format

  DATA SPLITTING:
    - Patient-level splitting (no patient overlap between train/val/test)
    - One row per patient per ASA class (deduplication)

  USAGE:
    1. Place your CSV export in the same directory as this script
    2. Update DATA_PATH below if the filename differs
    3. Run: python asa_pipeline.py
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
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
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "Dataset" / "Final_raw_data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "simple_ml_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Column names (must match CSV export) ---
COL_PATIENT_ID    = "DurableKey"
COL_DATE          = "Ydelsesdato"
COL_TARGET        = "ASA Score"
COL_AGE           = "Age"
COL_SEX           = "Sex"
# COL_HEIGHT        = "Height_cm"
# COL_WEIGHT        = "Weight_kg"
COL_BMI           = "BMI"
COL_ALCOHOL       = "AlcoholConsumption_X"
COL_SMOKING       = "SmokingStatus_Simplified"
COL_N_ALLERGIES   = "NumberOfAllergies"
COL_N_DIAGNOSES   = "NumberOfDiagnoses"
COL_ALCOHOL_MISSING = "AlcoholConsumption_Missing"

COL_NEURO         = "NeuroPsyk_flag"
COL_RESP          = "Respiratorisk_flag"
COL_KARDIO        = "Kardiovaskulært_flag"
COL_GI            = "GI/Lever/Nyre_flag"
COL_ENDO          = "Endo/Andet_flag"
COL_BEVAEGEAPP    = "Bevægeapparat_flag"

# --- Feature groups ---
NUMERICAL_FEATURES = [
    COL_AGE,
    # COL_HEIGHT,
    # COL_WEIGHT,
    COL_BMI,
    COL_ALCOHOL,
    COL_N_ALLERGIES,
    COL_N_DIAGNOSES,
]

CATEGORICAL_FEATURES = [
    COL_SEX,
    COL_SMOKING,
]

BINARY_FEATURES = [
    COL_NEURO,
    COL_RESP,
    COL_KARDIO,
    COL_GI,
    COL_ENDO,
    COL_BEVAEGEAPP,
    COL_ALCOHOL_MISSING,
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# --- Exclusions ---
EXCLUDE_ASA_ABOVE = 3       # Exclude ASA 4, 5, 6
EXCLUDE_UNDER_AGE = 18

# --- Splitting ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15
CV_FOLDS = 10

# --- Deduplication ---
PATIENT_ROW_MODE = "patient_asa_unique"  # "all_notes" or "patient_asa_unique"


# =============================================================================
# PATIENT-LEVEL SPLITTING
# =============================================================================

def make_group_stratified_splits(X, y, groups, test_size=0.20, val_size=0.15, random_state=42):
    """
    Create train/val/test splits with NO patient overlap between sets.
    Uses StratifiedGroupKFold to approximate the requested proportions
    while keeping all rows from one patient in the same split.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Split off test set
    n_test_splits = max(2, round(1 / test_size))
    sgkf_test = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(sgkf_test.split(X, y, groups))

    X_trainval = X.iloc[trainval_idx].reset_index(drop=True)
    y_trainval = y.iloc[trainval_idx].reset_index(drop=True)
    groups_trainval = groups.iloc[trainval_idx].reset_index(drop=True)

    # Split off validation set from trainval
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

    # Verify no patient overlap
    overlap_tv = set(groups_train) & set(groups_val)
    overlap_tt = set(groups_train) & set(groups_test)
    overlap_vt = set(groups_val) & set(groups_test)
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"Patient overlap detected! "
            f"Train∩Val: {len(overlap_tv)}, Train∩Test: {len(overlap_tt)}, Val∩Test: {len(overlap_vt)}"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    if not LIGHTGBM_AVAILABLE:
        print("Cannot run pipeline without LightGBM. Exiting.")
        return

    print("\n" + "█" * 60)
    print("  ASA SCORE PREDICTION — SIMPLIFIED PIPELINE")
    print("█" * 60)
    print(f"  Data file       : {DATA_PATH.name}")
    print(f"  Output          : {OUTPUT_DIR}")
    print(f"  CV folds        : {CV_FOLDS}")
    print(f"  Patient split   : True")
    print(f"  Row mode        : {PATIENT_ROW_MODE}")
    print(f"  Features        : {len(ALL_FEATURES)} total")
    print(f"    Numerical     : {len(NUMERICAL_FEATURES)}")
    print(f"    Categorical   : {len(CATEGORICAL_FEATURES)}")
    print(f"    Binary        : {len(BINARY_FEATURES)}")
    print()

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("=" * 55)
    print("  STEP 1: LOADING DATA")
    print("=" * 55)

    df = pd.read_excel(DATA_PATH, sheet_name="Final")
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    df[COL_ALCOHOL_MISSING] = df[COL_ALCOHOL].isna().astype(int)

    # Check that all expected columns exist
    missing_cols = [c for c in [COL_PATIENT_ID, COL_TARGET, COL_AGE] + ALL_FEATURES if c not in df.columns]
    if missing_cols:
        print(f"\n  ✗ Missing columns: {missing_cols}")
        print("  Please check your CSV column names match the configuration above.")
        return
    print("  ✓ All expected columns found")

    # =========================================================================
    # STEP 2: EXCLUSIONS
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 2: EXCLUSIONS")
    print("=" * 55)

    n0 = len(df)

    # Drop rows without ASA score
    df = df.dropna(subset=[COL_TARGET])
    print(f"  Missing ASA score removed   : {n0 - len(df):,}")
    n0 = len(df)

    # Exclude high ASA scores
    df = df[df[COL_TARGET] <= EXCLUDE_ASA_ABOVE]
    print(f"  ASA > {EXCLUDE_ASA_ABOVE} removed            : {n0 - len(df):,}")
    n0 = len(df)

    # Exclude children
    df = df[df[COL_AGE].fillna(0) >= EXCLUDE_UNDER_AGE]
    print(f"  Age < {EXCLUDE_UNDER_AGE} removed             : {n0 - len(df):,}")

    df = df.reset_index(drop=True)
    print(f"\n  Remaining: {len(df):,} rows, {df[COL_PATIENT_ID].nunique():,} unique patients")

    # =========================================================================
    # STEP 3: DEDUPLICATION
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 3: PATIENT-LEVEL DEDUPLICATION")
    print("=" * 55)

    if PATIENT_ROW_MODE == "patient_asa_unique":
        n_before = len(df)

        # Show top repeated combinations before reduction
        repeat_summary = (
            df.groupby([COL_PATIENT_ID, COL_TARGET])
            .size()
            .reset_index(name="n_rows")
            .sort_values("n_rows", ascending=False)
        )
        repeat_summary.to_csv(OUTPUT_DIR / "patient_asa_repeats_before_dedup.csv", index=False)

        # Keep first occurrence per patient per ASA class
        if COL_DATE in df.columns:
            df = df.sort_values([COL_PATIENT_ID, COL_TARGET, COL_DATE])
        df = df.drop_duplicates(subset=[COL_PATIENT_ID, COL_TARGET], keep="first").reset_index(drop=True)

        print(f"  Mode: one row per patient per ASA class")
        print(f"  Before : {n_before:,}")
        print(f"  After  : {len(df):,}")
        print(f"  Removed: {n_before - len(df):,}")
    else:
        print("  Mode: keeping all rows")

    # =========================================================================
    # STEP 4: EXPLORATORY DATA ANALYSIS
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    print(f"\n  Dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Unique patients: {df[COL_PATIENT_ID].nunique():,}")

    # Target distribution
    target_counts = df[COL_TARGET].value_counts().sort_index()
    target_pct = (target_counts / len(df) * 100).round(1)
    print(f"\n  ASA distribution:")
    for asa_class, count in target_counts.items():
        print(f"    ASA {int(asa_class)}: {count:,} ({target_pct[asa_class]}%)")

    # Missing values summary
    print(f"\n  Missing values:")
    for col in ALL_FEATURES:
        n_miss = df[col].isna().sum()
        pct = n_miss / len(df) * 100
        if n_miss > 0:
            print(f"    {col}: {n_miss:,} ({pct:.1f}%)")

    # Plot target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    target_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("ASA Class Counts")
    axes[0].set_xlabel("ASA Score")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].pie(
        target_counts,
        labels=[f"ASA {int(c)}" for c in target_counts.index],
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Blues_d", len(target_counts)),
    )
    axes[1].set_title("ASA Class Proportions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✓ Saved: 01_target_distribution.png")

    # =========================================================================
    # STEP 5: TARGET ENCODING
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 5: TARGET ENCODING")
    print("=" * 55)

    unique_classes = sorted(df[COL_TARGET].unique())
    class_map = {c: i for i, c in enumerate(unique_classes)}
    inverse_class_map = {i: c for c, i in class_map.items()}
    class_names = [f"ASA {int(c)}" for c in unique_classes]
    n_classes = len(unique_classes)

    df[COL_TARGET] = df[COL_TARGET].map(class_map).astype(int)
    print(f"  Encoding: {class_map}")
    print(f"  Classes : {class_names}")

    # =========================================================================
    # STEP 6: PREPARE X, y, GROUPS
    # =========================================================================
    X = df[ALL_FEATURES].copy()
    y = df[COL_TARGET].copy()
    groups = df[COL_PATIENT_ID].copy()

    # =========================================================================
    # STEP 7: TRAIN / VAL / TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 7: TRAIN / VAL / TEST SPLIT (patient-level)")
    print("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test, groups_train = make_group_stratified_splits(
        X, y, groups,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"  Train : {len(X_train):,} rows")
    print(f"  Val   : {len(X_val):,} rows")
    print(f"  Test  : {len(X_test):,} rows")

    print("\n  Target distribution per split:")
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = ys.value_counts(normalize=True).sort_index()
        dist_str = ", ".join(f"ASA {int(inverse_class_map[k])}: {v:.1%}" for k, v in dist.items())
        print(f"    {name}: {dist_str}")

    # =========================================================================
    # STEP 8: PREPROCESSING PIPELINE
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 8: PREPROCESSING")
    print("=" * 55)

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20)),
    ])

    # Binary features: impute missing as 0 (assume no finding if section was absent)
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    print(f"  Numerical  : median imputation + scaling")
    print(f"  Categorical: fill 'Unknown' + one-hot encoding")
    print(f"  Binary     : fill 0 (no finding assumed if absent)")

    # =========================================================================
    # STEP 9: MODEL DEFINITION
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 9: MODEL DEFINITION")
    print("=" * 55)

    n_estimators = 600
    learning_rate = 0.03
    num_leaves = 30
    max_depth = 8
    min_child_samples = 70
    subsample = 0.75
    subsample_freq = 1
    colsample_bytree = 0.75
    reg_alpha = 0.5
    reg_lambda = 4.0
    min_split_gain = 0.05
    class_weight = "balanced"
    n_jobs = -1
    verbose = -1

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=verbose,
        )),
    ])

    print("  Model: LightGBM (regularized)")
    print("  Key hyperparameters:")
    print(f"    n_estimators     : {n_estimators}")
    print(f"    learning_rate    : {learning_rate}")
    print(f"    max_depth        : {max_depth}")
    print(f"    min_child_samples: {min_child_samples}")
    print(f"    subsample        : {subsample}")
    print(f"    colsample_bytree : {colsample_bytree}")
    print(f"    class_weight     : {class_weight}")

    # =========================================================================
    # STEP 10: CROSS-VALIDATION
    # =========================================================================
    print("\n" + "=" * 55)
    print(f"  STEP 10: {CV_FOLDS}-FOLD CROSS-VALIDATION")
    print("=" * 55)

    cv_strategy = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    print("  Training with cross-validation...")
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv_strategy,
        groups=groups_train,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
    )

    # Display CV results
    for metric in scoring:
        train_mean = cv_scores[f"train_{metric}"].mean()
        test_mean = cv_scores[f"test_{metric}"].mean()
        test_std = cv_scores[f"test_{metric}"].std()
        gap = train_mean - test_mean
        flag = "⚠ OVERFIT" if gap > 0.10 else ("⚠ moderate" if gap > 0.05 else "✓ ok")
        print(f"  {metric:15s}: train={train_mean:.4f}  val={test_mean:.4f} ± {test_std:.4f}  gap={gap:+.4f} ({flag})")

    # Save CV results
    cv_summary = {
        metric: {
            "train_mean": float(cv_scores[f"train_{metric}"].mean()),
            "val_mean": float(cv_scores[f"test_{metric}"].mean()),
            "val_std": float(cv_scores[f"test_{metric}"].std()),
            "gap": float(cv_scores[f"train_{metric}"].mean() - cv_scores[f"test_{metric}"].mean()),
        }
        for metric in scoring
    }

    # =========================================================================
    # STEP 11: FIT ON FULL TRAINING SET
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 11: FITTING ON FULL TRAINING SET")
    print("=" * 55)

    model.fit(X_train, y_train)
    print("  ✓ Model fitted")

    # =========================================================================
    # STEP 12: VALIDATION SET EVALUATION
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 12: VALIDATION SET EVALUATION")
    print("=" * 55)

    y_val_pred = model.predict(X_val)

    val_metrics = {
        "Accuracy": accuracy_score(y_val, y_val_pred),
        "F1 (macro)": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
        "F1 (weighted)": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
        "Precision (macro)": precision_score(y_val, y_val_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_val, y_val_pred, average="macro", zero_division=0),
    }

    try:
        y_val_proba = model.predict_proba(X_val)
        val_metrics["ROC-AUC (macro)"] = roc_auc_score(y_val, y_val_proba, multi_class="ovr", average="macro")
    except Exception:
        val_metrics["ROC-AUC (macro)"] = None

    for name, value in val_metrics.items():
        if value is not None:
            print(f"  {name:20s}: {value:.4f}")

    # =========================================================================
    # STEP 13: TEST SET EVALUATION
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 13: TEST SET EVALUATION")
    print("=" * 55)

    y_test_pred = model.predict(X_test)

    # Classification report
    print("\n" + classification_report(y_test, y_test_pred, target_names=class_names, digits=3, zero_division=0))

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1m = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_f1w = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1 macro  : {test_f1m:.4f}")
    print(f"  F1 weight : {test_f1w:.4f}")

    auc_score = None
    try:
        y_test_proba = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_test_proba, multi_class="ovr", average="macro")
        print(f"  ROC-AUC   : {auc_score:.4f}")
    except Exception as e:
        print(f"  ROC-AUC   : could not compute ({e})")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_test_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — LightGBM", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✓ Saved: 02_confusion_matrix.png")

    # =========================================================================
    # STEP 14: FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 14: FEATURE IMPORTANCE")
    print("=" * 55)

    try:
        fitted_preprocessor = model.named_steps["preprocessor"]
        feature_names = fitted_preprocessor.get_feature_names_out()
        classifier = model.named_steps["classifier"]

        if hasattr(classifier, "feature_importances_"):
            importances = pd.Series(classifier.feature_importances_, index=feature_names).sort_values(ascending=False)

            # Clean up feature names for display (remove transformer prefixes)
            importances.index = [name.split("__", 1)[-1] if "__" in name else name for name in importances.index]

            # Plot
            top_n = min(20, len(importances))
            fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
            importances.head(top_n).sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"Top {top_n} Features — LightGBM", style="italic", color="grey")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "03_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  ✓ Saved: 03_feature_importance.png")

            print(f"\n  Feature importances:")
            for feat, imp in importances.head(top_n).items():
                print(f"    {feat:35s}: {imp:.4f}")
        else:
            print("  Feature importance not available.")
            importances = None
    except Exception as e:
        print(f"  Feature importance skipped: {e}")
        importances = None

    # =========================================================================
    # STEP 15: EXPORT RESULTS
    # =========================================================================
    print("\n" + "=" * 55)
    print("  STEP 15: EXPORTING RESULTS")
    print("=" * 55)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Test metrics
    test_metrics = {
        "model": "LightGBM",
        "n_classes": int(n_classes),
        "class_names": class_names,
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1m, 4),
        "test_f1_weighted": round(test_f1w, 4),
        "test_roc_auc": round(auc_score, 4) if auc_score else None,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": len(ALL_FEATURES),
        "features": ALL_FEATURES,
        "patient_row_mode": PATIENT_ROW_MODE,
        "cv_folds": CV_FOLDS,
        "cv_results": cv_summary,
        "timestamp": timestamp,
    }

    with open(OUTPUT_DIR / f"test_metrics_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: test_metrics_{timestamp}.json")

    # Classification report
    report_dict = classification_report(
        y_test, y_test_pred,
        target_names=class_names,
        digits=4, zero_division=0, output_dict=True,
    )
    with open(OUTPUT_DIR / f"classification_report_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: classification_report_{timestamp}.json")

    # Feature importance
    if importances is not None:
        importances.to_csv(OUTPUT_DIR / f"feature_importance_{timestamp}.csv", header=["importance"])
        print(f"  ✓ Saved: feature_importance_{timestamp}.csv")

    # Model
    model_filename = f"asa_model_lightgbm_{timestamp}.joblib"
    joblib.dump(model, OUTPUT_DIR / model_filename)
    print(f"  ✓ Saved: {model_filename}")

    # Metadata
    metadata = {
        "model_name": "LightGBM",
        "class_names": class_names,
        "features": ALL_FEATURES,
        "feature_groups": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "binary": BINARY_FEATURES,
        },
        "notes": (
            "Model trained on structured features only (no TF-IDF). "
            "Features are aligned with what can be extracted from a patient "
            "chatbot transcript via LLM, ensuring no domain gap between "
            "training and inference."
        ),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1m, 4),
        "test_roc_auc": round(auc_score, 4) if auc_score else None,
        "patient_row_mode": PATIENT_ROW_MODE,
        "timestamp": timestamp,
    }
    with open(OUTPUT_DIR / f"model_metadata_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: model_metadata_{timestamp}.json")

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1 macro  : {test_f1m:.4f}")
    if auc_score:
        print(f"  ROC-AUC   : {auc_score:.4f}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()