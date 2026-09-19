import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# EXPERIMENT 4 - STAGE 5
# MODEL SELECTION & SAVING
# ============================================================

print("=" * 70)
print("EXPERIMENT 4 - STAGE 5: MODEL SELECTION & SAVING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

DATA_PATH = "data/medicare_feature_engineered.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_hospital_quality_model.pkl"
)
REPORT_DIR = "reports/experiment_4"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

# Keep only hospitals with a known target
df = df[df["rating_category"].notna()].copy()

target = "rating_category"


# ------------------------------------------------------------
# 2. Remove target leakage / metadata
# ------------------------------------------------------------

drop_columns = [
    "rating_category",
    "overall_rating",
    "overall_rating_raw",
    "provider_id",
    "hospital_name",
    "address",
    "phone_number",
    "profile_url",
    "scraped_at",
    "data_last_updated",
    "patient_survey_rating_raw",
    "all_star_rating_labels",
    "zip_code"
]

X = df.drop(columns=drop_columns)
y = df[target]


# ------------------------------------------------------------
# 3. Identify feature types
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ------------------------------------------------------------
# 4. Train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 5. Preprocessing
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ------------------------------------------------------------
# 6. Selected model
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ------------------------------------------------------------
# 7. Train final pipeline
# ------------------------------------------------------------

print("\nTraining selected model...")
start_time = time.time()

pipeline.fit(X_train, y_train)

training_time = time.time() - start_time


# ------------------------------------------------------------
# 8. Evaluate final model
# ------------------------------------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SELECTED MODEL")
print("-" * 70)

print("Model:              Logistic Regression")
print("Stage:              Baseline")
print(f"Accuracy:            {accuracy:.4f}")
print(f"Macro Precision:     {macro_precision:.4f}")
print(f"Macro Recall:        {macro_recall:.4f}")
print(f"Macro F1:            {macro_f1:.4f}")
print(f"Weighted F1:         {weighted_f1:.4f}")
print(f"Training Time:       {training_time:.4f}s")

print("\nTrain samples:", len(X_train))
print("Test samples: ", len(X_test))
print("Features:      ", X.shape[1])


# ------------------------------------------------------------
# 10. Save complete pipeline
# ------------------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved to:")
print(os.path.abspath(MODEL_PATH))


# ------------------------------------------------------------
# 11. Verify saved model
# ------------------------------------------------------------

print("\nVerifying saved model...")

loaded_pipeline = joblib.load(MODEL_PATH)

loaded_predictions = loaded_pipeline.predict(X_test)

verification_accuracy = accuracy_score(
    y_test,
    loaded_predictions
)

print(
    f"Loaded model accuracy: {verification_accuracy:.4f}"
)

if abs(accuracy - verification_accuracy) < 1e-10:
    print("Model verification: PASS")
else:
    print("Model verification: FAIL")


# ------------------------------------------------------------
# 12. Save model-selection report
# ------------------------------------------------------------

selection_report = pd.DataFrame([
    {
        "selected_model": "Logistic Regression",
        "stage": "Baseline",
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X.shape[1],
        "random_state": 42,
        "test_size": 0.20,
        "model_path": MODEL_PATH
    }
])

REPORT_PATH = os.path.join(
    REPORT_DIR,
    "final_model_selection.csv"
)

selection_report.to_csv(
    REPORT_PATH,
    index=False
)

print("\nModel selection report saved to:")
print(os.path.abspath(REPORT_PATH))


# ------------------------------------------------------------
# 13. Final status
# ------------------------------------------------------------

if (
    os.path.exists(MODEL_PATH)
    and abs(accuracy - verification_accuracy) < 1e-10
):
    print("\n" + "=" * 70)
    print("STAGE 5 MODEL SELECTION & SAVING COMPLETED")
    print("=" * 70)
else:
    print("\nModel saving/verification failed.")