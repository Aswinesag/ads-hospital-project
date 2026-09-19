import os
import time
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_DIR = "data"
REPORT_DIR = "reports/experiment_4"
CONFUSION_DIR = os.path.join(REPORT_DIR, "confusion_matrices")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("EXPERIMENT 4 - STAGE 2: BASELINE MODEL TRAINING")
print("=" * 70)

print("\nLoading prepared dataset...")

df = pd.read_csv(
    os.path.join(DATA_DIR, "medicare_feature_engineered.csv")
)

# Keep only hospitals with a known target
df = df[df["rating_category"].notna()].copy()

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. DEFINE TARGET AND FEATURES
# ============================================================

TARGET = "rating_category"

# Columns that must NOT be used as predictors
DROP_COLUMNS = [
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

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


# ============================================================
# 4. SAME TRAIN/TEST SPLIT AS STAGE 1
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print(f"\nNumerical features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

print("\nCategorical feature cardinality:")
for col in categorical_features:
    print(f"  {col}: {X_train[col].nunique()} unique values")


# ============================================================
# 6. COMMON PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
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
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 7. DEFINE FIVE BASELINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "SVM (RBF)": SVC(
        kernel="rbf",
        random_state=42
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE
# ============================================================

results = []

print("\n" + "=" * 70)
print("TRAINING BASELINE MODELS")
print("=" * 70)

for model_name, model in models.items():

    print(f"\n{'-' * 70}")
    print(f"Training: {model_name}")
    print(f"{'-' * 70}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    start_time = time.time()

    # Train
    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Confusion matrix
    labels = ["Low", "Medium", "High"]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    # Save confusion matrix
    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    cm_path = os.path.join(
        CONFUSION_DIR,
        f"{safe_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # Store result
    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "training_time_seconds": training_time
    })

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1:          {f1_macro:.4f}")
    print(f"Weighted F1:       {f1_weighted:.4f}")
    print(f"Training Time:     {training_time:.4f} seconds")


# ============================================================
# 9. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

# Sort only for easier reading.
# This does NOT mean we are selecting a final model.
results_df = results_df.sort_values(
    by="accuracy",
    ascending=False
)

results_path = os.path.join(
    REPORT_DIR,
    "baseline_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 10. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(f"Comparison table:")
print(f"  {results_path}")

print("\nConfusion matrices:")
print(f"  {CONFUSION_DIR}/")

print("\nStage 2 baseline training completed.")