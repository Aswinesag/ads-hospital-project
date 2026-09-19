import os
import time
import tempfile

import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split
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
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/medicare_feature_engineered.csv"

EXPERIMENT_NAME = "Hospital Quality Classification"

# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_DB = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_DB)

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("EXPERIMENT 4 - STAGE 4: MLFLOW EXPERIMENT TRACKING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df[
    df["rating_category"].notna()
].copy()

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
# 3. SAME TRAIN/TEST SPLIT
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
# 5. PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
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
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

models = {

    # ---------------- BASELINE ----------------

    "Baseline Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Baseline Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Baseline Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "Baseline Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "Baseline SVM RBF":
        SVC(
            kernel="rbf",
            random_state=42
        ),

    # ---------------- TUNED ----------------

    "Tuned Logistic Regression":
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ),

    "Tuned Gradient Boosting":
        GradientBoostingClassifier(
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=150,
            random_state=42
        )
}


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    is_tuned
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print("\n" + "-" * 70)
    print(f"MLflow Run: {model_name}")
    print("-" * 70)

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    elapsed_time = (
        time.time() - start_time
    )

    y_pred = pipeline.predict(X_test)

    # ---------------- METRICS ----------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    # ---------------- CONFUSION MATRIX ----------------

    labels = [
        "Low",
        "Medium",
        "High"
    ]

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

    # ====================================================
    # START MLFLOW RUN
    # ====================================================

    with mlflow.start_run(
        run_name=model_name
    ):

        # ---------------- TAGS ----------------

        mlflow.set_tag(
            "model_type",
            model_name
        )

        mlflow.set_tag(
            "stage",
            "tuned" if is_tuned
            else "baseline"
        )

        mlflow.set_tag(
            "target",
            TARGET
        )

        mlflow.set_tag(
            "dataset",
            "medicare_feature_engineered.csv"
        )

        # ---------------- COMMON PARAMETERS ----------------

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "cv_strategy",
            "Not applicable - held-out test evaluation"
        )

        mlflow.log_param(
            "numeric_features",
            len(numeric_features)
        )

        mlflow.log_param(
            "categorical_features",
            len(categorical_features)
        )

        mlflow.log_param(
            "training_samples",
            len(X_train)
        )

        mlflow.log_param(
            "test_samples",
            len(X_test)
        )

        mlflow.log_param(
            "feature_count",
            X_train.shape[1]
        )

        # ---------------- MODEL PARAMETERS ----------------

        model_params = model.get_params()

        for param_name, value in model_params.items():

            # MLflow parameter values must be simple
            # string/number values.
            try:
                mlflow.log_param(
                    f"model_{param_name}",
                    value
                )
            except Exception:
                mlflow.log_param(
                    f"model_{param_name}",
                    str(value)
                )

        # ---------------- METRICS ----------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "macro_precision",
            macro_precision
        )

        mlflow.log_metric(
            "macro_recall",
            macro_recall
        )

        mlflow.log_metric(
            "macro_f1",
            macro_f1
        )

        mlflow.log_metric(
            "weighted_f1",
            weighted_f1
        )

        mlflow.log_metric(
            "training_time_seconds",
            elapsed_time
        )

        # ---------------- CONFUSION MATRIX ARTIFACT ----------------

        with tempfile.TemporaryDirectory() as temp_dir:

            cm_path = os.path.join(
                temp_dir,
                "confusion_matrix.csv"
            )

            cm_df.to_csv(
                cm_path
            )

            mlflow.log_artifact(
                cm_path,
                artifact_path="evaluation"
            )

        # ---------------- MODEL ARTIFACT ----------------

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        # ---------------- CONSOLE OUTPUT ----------------

        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:    {macro_recall:.4f}")
        print(f"Macro F1:        {macro_f1:.4f}")
        print(f"Weighted F1:     {weighted_f1:.4f}")
        print(f"Training Time:   {elapsed_time:.4f}s")

        run_id = mlflow.active_run().info.run_id

        print(f"Run ID:          {run_id}")

    return {
        "model": model_name,
        "stage": "tuned" if is_tuned
        else "baseline",
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": elapsed_time
    }


# ============================================================
# 8. RUN ALL MODELS
# ============================================================

results = []

for model_name, model in models.items():

    is_tuned = model_name.startswith(
        "Tuned"
    )

    result = evaluate_model(
        model_name,
        model,
        is_tuned
    )

    results.append(result)


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

summary_path = (
    "reports/experiment_4/"
    "mlflow_run_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 70)
print("MLFLOW RUN SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSummary saved to:")
print(summary_path)

print("MLflow tracking database:")
print(os.path.abspath("mlflow.db"))

print("\nStage 4 MLflow tracking completed.")