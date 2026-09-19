import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

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

DATA_PATH = "data/medicare_feature_engineered.csv"
REPORT_DIR = "reports/experiment_4"
CONFUSION_DIR = os.path.join(
    REPORT_DIR,
    "tuned_confusion_matrices"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("EXPERIMENT 4 - STAGE 3: HYPERPARAMETER TUNING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

# Only hospitals with a known target
df = df[df["rating_category"].notna()].copy()

TARGET = "rating_category"

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
# 3. SAME TRAIN/TEST SPLIT AS STAGE 2
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Features:         {X_train.shape[1]}")


# ============================================================
# 4. FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ============================================================
# 5. PREPROCESSING
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
# 6. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TUNING LOGISTIC REGRESSION")
print("=" * 70)

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        )
    ]
)

logistic_params = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"]
}

logistic_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_params,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

start_time = time.time()

logistic_search.fit(X_train, y_train)

logistic_time = time.time() - start_time

print("\nBest Logistic Regression parameters:")
print(logistic_search.best_params_)

print(
    f"Best CV Macro F1: "
    f"{logistic_search.best_score_:.4f}"
)

print(
    f"Tuning time: "
    f"{logistic_time:.2f} seconds"
)


# ============================================================
# 8. GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 70)
print("TUNING GRADIENT BOOSTING")
print("=" * 70)

gradient_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingClassifier(
                random_state=42
            )
        )
    ]
)

gradient_params = {
    "model__n_estimators": [50, 100, 150],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

gradient_search = GridSearchCV(
    estimator=gradient_pipeline,
    param_grid=gradient_params,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

start_time = time.time()

gradient_search.fit(X_train, y_train)

gradient_time = time.time() - start_time

print("\nBest Gradient Boosting parameters:")
print(gradient_search.best_params_)

print(
    f"Best CV Macro F1: "
    f"{gradient_search.best_score_:.4f}"
)

print(
    f"Tuning time: "
    f"{gradient_time:.2f} seconds"
)


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model, tuning_time):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Low", "Actual Medium", "Actual High"],
        columns=["Pred Low", "Pred Medium", "Pred High"]
    )

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    cm_path = os.path.join(
        CONFUSION_DIR,
        f"{filename}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "tuning_time_seconds": tuning_time
    }


# ============================================================
# 10. TEST-SET EVALUATION
# ============================================================

logistic_result = evaluate_model(
    "Tuned Logistic Regression",
    logistic_search.best_estimator_,
    logistic_time
)

gradient_result = evaluate_model(
    "Tuned Gradient Boosting",
    gradient_search.best_estimator_,
    gradient_time
)


# ============================================================
# 11. SAVE TUNED RESULTS
# ============================================================

tuned_results = pd.DataFrame(
    [
        logistic_result,
        gradient_result
    ]
)

tuned_results_path = os.path.join(
    REPORT_DIR,
    "tuned_model_results.csv"
)

tuned_results.to_csv(
    tuned_results_path,
    index=False
)


# ============================================================
# 12. BASELINE VS TUNED COMPARISON
# ============================================================

baseline_path = os.path.join(
    REPORT_DIR,
    "baseline_model_results.csv"
)

baseline_df = pd.read_csv(baseline_path)

baseline_selected = baseline_df[
    baseline_df["model"].isin(
        [
            "Logistic Regression",
            "Gradient Boosting"
        ]
    )
].copy()

baseline_selected["model"] = (
    baseline_selected["model"]
    .map({
        "Logistic Regression":
            "Logistic Regression",
        "Gradient Boosting":
            "Gradient Boosting"
    })
)

comparison_rows = []

for _, row in baseline_selected.iterrows():

    model_name = row["model"]

    tuned_row = tuned_results[
        tuned_results["model"] ==
        f"Tuned {model_name}"
    ].iloc[0]

    comparison_rows.append({
        "model": model_name,
        "baseline_accuracy": row["accuracy"],
        "tuned_accuracy": tuned_row["accuracy"],
        "accuracy_change":
            tuned_row["accuracy"] - row["accuracy"],

        "baseline_macro_f1": row["macro_f1"],
        "tuned_macro_f1": tuned_row["macro_f1"],
        "macro_f1_change":
            tuned_row["macro_f1"] - row["macro_f1"],

        "baseline_weighted_f1": row["weighted_f1"],
        "tuned_weighted_f1": tuned_row["weighted_f1"],
        "weighted_f1_change":
            tuned_row["weighted_f1"] -
            row["weighted_f1"]
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_path = os.path.join(
    REPORT_DIR,
    "baseline_vs_tuned_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TUNED MODEL TEST-SET RESULTS")
print("=" * 70)

print(
    tuned_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("BASELINE VS TUNED")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(f"1. {tuned_results_path}")
print(f"2. {comparison_path}")
print(f"3. {CONFUSION_DIR}/")

print("\nStage 3 hyperparameter tuning completed.")