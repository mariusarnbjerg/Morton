# ============================================================
#  ASA SCORE PREDICTION — MASTER'S THESIS ML PIPELINE
#  Preoperative Anesthesiology AI — Proof of Concept
# ============================================================
# CELL 1 — INSTALL DEPENDENCIES (run once in Colab)
# !pip install xgboost shap --quiet

# CELL 2 — IMPORTS
import os
import warnings
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
print("✓ All imports successful.")

# CELL 3 — CONFIGURATION
DATA_PATH            = None          # [REPLACE] e.g. "/content/patient_data.csv"
TARGET_COLUMN        = "ASA_class"   # [REPLACE] exact column name in your data
NUMERICAL_FEATURES   = ["age", "bmi", "weight_kg", "height_cm"]
CATEGORICAL_FEATURES = ["sex", "smoking_status", "diabetes", "hypertension", "surgery_type"]
ORDINAL_FEATURES     = []
TASK_TYPE            = "multiclass"  # "binary" or "multiclass"
BINARY_MAPPING       = {1:0,"I":0,"ASA I":0, 2:0,"II":0,"ASA II":0, 3:1,"III":1,"ASA III":1, 4:1,"IV":1,"ASA IV":1}
EXCLUDE_ASA_V        = True
EXCLUDE_UNDER_18     = True
MIN_AGE_COLUMN       = "age"
RANDOM_STATE         = 42
TEST_SIZE            = 0.20
VAL_SIZE             = 0.15
CV_FOLDS             = 5
TUNE_MODELS          = True
RUN_SHAP             = True
OUTPUT_DIR           = "/content/ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CELL 4 — PLACEHOLDER DATASET (comment out when real data arrives)
def generate_placeholder_data(n_patients=300, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age"            : rng.integers(18, 90, size=n_patients).astype(float),
        "sex"            : rng.choice(["Male", "Female"], size=n_patients),
        "weight_kg"      : rng.normal(loc=78, scale=15, size=n_patients).round(1),
        "height_cm"      : rng.normal(loc=172, scale=10, size=n_patients).round(1),
        "smoking_status" : rng.choice(["Never","Former","Current","Unknown"], size=n_patients, p=[0.5,0.25,0.15,0.10]),
        "diabetes"       : rng.choice(["No","Type 1","Type 2"], size=n_patients, p=[0.7,0.05,0.25]),
        "hypertension"   : rng.choice(["No","Yes"], size=n_patients, p=[0.55,0.45]),
        "surgery_type"   : rng.choice(["Orthopaedic","General","Cardiac","Gynaecology","Urology"], size=n_patients),
        "ASA_class"      : rng.choice([1,2,3,4], size=n_patients, p=[0.20,0.45,0.30,0.05]),
    })
    df["bmi"] = (df["weight_kg"] / (df["height_cm"]/100)**2).round(1)
    for col in ["bmi","smoking_status","weight_kg"]:
        missing_idx = rng.choice(df.index, size=int(0.05*n_patients), replace=False)
        df.loc[missing_idx, col] = np.nan
    return df

df_raw = generate_placeholder_data(n_patients=300, seed=RANDOM_STATE)
print(f"✓ Placeholder dataset: {df_raw.shape}")

# CELL 5 — REAL DATA LOADING (uncomment when data arrives, comment out Cell 4)
# df_raw = pd.read_csv(DATA_PATH, sep=",", encoding="utf-8")
# df_raw = pd.read_excel(DATA_PATH, sheet_name=0)
# df_raw = pd.read_csv(DATA_PATH, sep=";", decimal=",", encoding="latin-1")

# CELL 6 — EDA
print(f"\nDataset shape: {df_raw.shape}")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(1)
missing_df = pd.DataFrame({"Missing count": missing, "Missing %": missing_pct}).query("`Missing count` > 0")
print(f"\nMissing values:\n{missing_df.to_string() if not missing_df.empty else 'None'}")
target_counts = df_raw[TARGET_COLUMN].value_counts().sort_index()
target_pct = (target_counts / len(df_raw) * 100).round(1)
print(f"\nTarget distribution:\n{pd.DataFrame({'Count': target_counts, '%': target_pct}).to_string()}")
imbalance_ratio = (target_pct.max() / target_pct.min()).round(1)
if imbalance_ratio > 3:
    print(f"⚠ Class imbalance detected ({imbalance_ratio}x). Using class_weight='balanced'.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
target_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title(f"Class counts — {TARGET_COLUMN}")
axes[1].pie(target_counts, labels=[f"ASA {c}" for c in target_counts.index], autopct="%1.1f%%", startangle=90)
axes[1].set_title("Class proportions")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# CELL 7 — EXCLUSIONS & TARGET PREP
df = df_raw.copy()
df = df.dropna(subset=[TARGET_COLUMN])
if EXCLUDE_ASA_V:
    df = df[~df[TARGET_COLUMN].astype(str).isin(["5","V","ASA V"])]
if EXCLUDE_UNDER_18 and MIN_AGE_COLUMN in df.columns:
    df = df[df[MIN_AGE_COLUMN] >= 18]
if TASK_TYPE == "binary":
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(BINARY_MAPPING).astype(int)
    class_names = ["Low risk (I-II)", "High risk (III-IV)"]
else:
    unique_classes = sorted(df[TARGET_COLUMN].unique())
    class_map = {c: i for i, c in enumerate(unique_classes)}
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(class_map)
    class_names = [f"ASA {i+1}" for i in range(df[TARGET_COLUMN].nunique())]
n_classes = df[TARGET_COLUMN].nunique()
print(f"✓ Final sample: {len(df)} patients, {n_classes} classes: {class_names}")

# CELL 8 — FEATURE ASSIGNMENT
num_features = [f for f in NUMERICAL_FEATURES  if f in df.columns]
cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
ord_features = [f for f in ORDINAL_FEATURES     if f in df.columns]
all_features = num_features + cat_features + ord_features
LEAKAGE_RISK_COLUMNS = []  # [REPLACE] add any columns derived from the target
X = df[all_features].copy()
y = df[TARGET_COLUMN].copy()
print(f"✓ Features: {len(all_features)} total | X: {X.shape} | y: {y.shape}")

# CELL 9 — TRAIN/VAL/TEST SPLIT
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
val_frac = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_frac, random_state=RANDOM_STATE, stratify=y_trainval)
print(f"✓ Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# CELL 10 — PREPROCESSING PIPELINE
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
transformers_list = [
    ("num", numerical_transformer, num_features),
    ("cat", categorical_transformer, cat_features),
]
preprocessor = ColumnTransformer(transformers=transformers_list, remainder="drop", verbose_feature_names_out=False)
print("✓ Preprocessing pipeline defined.")

# CELL 11 — MODEL DEFINITIONS
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs", multi_class="auto")),
])
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
])
gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE, n_iter_no_change=15, validation_fraction=0.1)),
])
models = {"Logistic Regression": lr_pipeline, "Random Forest": rf_pipeline, "Gradient Boosting": gb_pipeline}
if XGBOOST_AVAILABLE:
    xgb_params = dict(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE, use_label_encoder=False, n_jobs=-1)
    if TASK_TYPE == "multiclass":
        xgb_params.update({"objective": "multi:softmax", "num_class": n_classes, "eval_metric": "mlogloss"})
    models["XGBoost"] = Pipeline([("preprocessor", preprocessor), ("classifier", XGBClassifier(**xgb_params))])
print(f"✓ Models: {list(models.keys())}")

# CELL 12 — CROSS-VALIDATION
cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"}
cv_results_summary = {}
rows = []
for model_name, pipeline in models.items():
    print(f"  CV: {model_name} ...")
    cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv_strategy, scoring=scoring, return_train_score=False, n_jobs=1)
    cv_results_summary[model_name] = {m: {"mean": cv_scores[f"test_{m}"].mean(), "std": cv_scores[f"test_{m}"].std()} for m in scoring}
    row = {"Model": model_name}
    for m, v in cv_results_summary[model_name].items():
        row[m] = f"{v['mean']:.3f} ± {v['std']:.3f}"
    rows.append(row)
    print(f"    F1 macro: {cv_results_summary[model_name]['f1_macro']['mean']:.3f} ± {cv_results_summary[model_name]['f1_macro']['std']:.3f}")
cv_summary_df = pd.DataFrame(rows).set_index("Model")
print(f"\nCV Summary:\n{cv_summary_df.to_string()}")

# CELL 13 — HYPERPARAMETER TUNING
if TUNE_MODELS:
    lr_tuned = RandomizedSearchCV(lr_pipeline, {"classifier__C": [0.001,0.01,0.1,1.0,10.0,100.0]}, n_iter=6, cv=cv_strategy, scoring="f1_macro", refit=True, random_state=RANDOM_STATE, n_jobs=1)
    lr_tuned.fit(X_train, y_train)
    models["Logistic Regression (tuned)"] = lr_tuned.best_estimator_
    print(f"LR tuned — best params: {lr_tuned.best_params_} | F1: {lr_tuned.best_score_:.3f}")

    rf_tuned = RandomizedSearchCV(rf_pipeline, {"classifier__n_estimators":[100,200,300], "classifier__max_depth":[None,5,10,15], "classifier__min_samples_split":[2,5,10], "classifier__max_features":["sqrt","log2"]}, n_iter=12, cv=cv_strategy, scoring="f1_macro", refit=True, random_state=RANDOM_STATE, n_jobs=1)
    rf_tuned.fit(X_train, y_train)
    models["Random Forest (tuned)"] = rf_tuned.best_estimator_
    print(f"RF tuned — best params: {rf_tuned.best_params_} | F1: {rf_tuned.best_score_:.3f}")

# CELL 14 — FIT ALL MODELS
fitted_models = {}
for model_name, pipeline in models.items():
    if "(tuned)" in model_name:
        fitted_models[model_name] = pipeline
        continue
    pipeline.fit(X_train, y_train)
    fitted_models[model_name] = pipeline
    print(f"✓ Fitted: {model_name}")

# CELL 15 — VALIDATION EVALUATION
avg = "binary" if TASK_TYPE == "binary" else "macro"
val_results = {}
for model_name, pipeline in fitted_models.items():
    y_pred_val = pipeline.predict(X_val)
    val_results[model_name] = {
        "Accuracy": round(accuracy_score(y_val, y_pred_val), 3),
        "Precision(macro)": round(precision_score(y_val, y_pred_val, average=avg, zero_division=0), 3),
        "Recall(macro)": round(recall_score(y_val, y_pred_val, average=avg, zero_division=0), 3),
        "F1(macro)": round(f1_score(y_val, y_pred_val, average=avg, zero_division=0), 3),
        "F1(weighted)": round(f1_score(y_val, y_pred_val, average="weighted", zero_division=0), 3),
    }
val_results_df = pd.DataFrame(val_results).T
print(f"\nValidation results:\n{val_results_df.to_string()}")
best_model_name = val_results_df["F1(macro)"].astype(float).idxmax()
best_model = fitted_models[best_model_name]
print(f"\n★ Best model: {best_model_name}")

# CELL 16 — FINAL TEST EVALUATION
y_pred_test = best_model.predict(X_test)
y_proba_test = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
print(f"\nTest Set Results — {best_model_name}:")
print(classification_report(y_test, y_pred_test, target_names=class_names, digits=3, zero_division=0))
test_acc  = accuracy_score(y_test, y_pred_test)
test_f1m  = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
test_f1w  = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
print(f"Accuracy: {test_acc:.3f} | F1 macro: {test_f1m:.3f} | F1 weighted: {test_f1w:.3f}")
if y_proba_test is not None:
    try:
        auc = roc_auc_score(y_test, y_proba_test, multi_class="ovr", average="macro")
        print(f"ROC-AUC (macro OvR): {auc:.3f}")
    except: pass

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test), display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
val_results_df["F1(macro)"].astype(float).sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Macro F1 (Validation)")
ax.set_title("Model Comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# CELL 17 — FEATURE IMPORTANCE & SHAP
def get_feature_names(preprocessor, num_features, cat_features, ord_features):
    names = list(num_features)
    if cat_features:
        ohe = preprocessor.named_transformers_["cat"]["onehot"]
        names.extend(ohe.get_feature_names_out(cat_features).tolist())
    names.extend(ord_features)
    return names

fitted_preprocessor = best_model.named_steps["preprocessor"]
feature_names_out = get_feature_names(fitted_preprocessor, num_features, cat_features, ord_features)
clf = best_model.named_steps["classifier"]

if hasattr(clf, "feature_importances_"):
    imp_series = pd.Series(clf.feature_importances_, index=feature_names_out).sort_values(ascending=False)
    top_n = min(20, len(imp_series))
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    imp_series.head(top_n).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Features (Native Importance)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_feature_importance_native.png", dpi=150, bbox_inches="tight")
    plt.show()

X_val_transformed = pd.DataFrame(fitted_preprocessor.transform(X_val), columns=feature_names_out)
perm_result = permutation_importance(clf, X_val_transformed, y_val, n_repeats=15, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=1)
perm_imp = pd.Series(perm_result.importances_mean, index=feature_names_out).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, max(4, min(20, len(perm_imp)) * 0.35)))
perm_imp.head(20).sort_values().plot(kind="barh", ax=ax, color="coral")
ax.set_title("Permutation Importance (Validation Set)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.show()

if RUN_SHAP and SHAP_AVAILABLE:
    try:
        explainer = shap.TreeExplainer(clf) if hasattr(clf, "feature_importances_") else shap.KernelExplainer(clf.predict_proba, shap.sample(X_val_transformed, 50))
        shap_values = explainer.shap_values(X_val_transformed)
        shap_to_plot = shap_values[-1] if isinstance(shap_values, list) else shap_values
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_to_plot, X_val_transformed, feature_names=feature_names_out, show=False, max_display=15)
        plt.title(f"SHAP Summary — {best_model_name}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/07_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.show()
    except Exception as e:
        print(f"SHAP failed: {e}")

# CELL 18 — EXPORT RESULTS
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
val_results_df.to_csv(f"{OUTPUT_DIR}/validation_results_{timestamp}.csv")
test_metrics = {
    "model": best_model_name, "task_type": TASK_TYPE, "n_classes": n_classes,
    "test_accuracy": round(test_acc, 4), "test_f1_macro": round(test_f1m, 4), "test_f1_weighted": round(test_f1w, 4),
    "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    "features_used": all_features, "cv_folds": CV_FOLDS, "random_state": RANDOM_STATE, "timestamp": timestamp,
    "note": "SYNTHETIC DATA — NOT FOR REPORTING",
}
with open(f"{OUTPUT_DIR}/test_metrics_{timestamp}.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
if hasattr(clf, "feature_importances_"):
    imp_series.to_csv(f"{OUTPUT_DIR}/feature_importance_{timestamp}.csv", header=["importance"])
print(f"✓ Results saved to {OUTPUT_DIR}")

# CELL 19 — MODEL EXPORT
import joblib
model_filename = f"asa_model_{best_model_name.replace(' ', '_').lower()}_{timestamp}.joblib"
model_save_path = f"{OUTPUT_DIR}/{model_filename}"
joblib.dump(best_model, model_save_path)
metadata = {
    "model_name": best_model_name, "task_type": TASK_TYPE, "n_classes": n_classes,
    "class_names": class_names, "numerical_features": num_features,
    "categorical_features": cat_features, "ordinal_features": ord_features,
    "all_features": all_features, "target_column": TARGET_COLUMN,
    "random_state": RANDOM_STATE, "trained_on": timestamp,
    "test_f1_macro": round(test_f1m, 4), "test_accuracy": round(test_acc, 4),
    "note": "Trained in Google Colab — proof of concept only",
}
with open(f"{OUTPUT_DIR}/model_metadata_{timestamp}.json", "w") as f:
    json.dump(metadata, f, indent=2)
reloaded = joblib.load(model_save_path)
test_pred = reloaded.predict(X_test.iloc[:1])
print(f"✓ Model saved and verified: {model_save_path}")
print(f"  Test prediction: {test_pred[0]} → {class_names[test_pred[0]]}")
print(f"\n  Download these two files from {OUTPUT_DIR}:")
print(f"  1. {model_filename}")
print(f"  2. model_metadata_{timestamp}.json")
print(f"  Place both in: ml/models/")
