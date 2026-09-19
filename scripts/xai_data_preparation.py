import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# EXPERIMENT 5 - STAGE 1
# DATASET & MODEL PREPARATION
# ============================================================

print("=" * 70)
print("EXPERIMENT 5 - STAGE 1: DATASET & MODEL PREPARATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATA_PATH = "data/medicare_feature_engineered.csv"
MODEL_PATH = "models/best_hospital_quality_model.pkl"

REPORT_DIR = "reports/experiment_5"

os.makedirs(REPORT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Load dataset and model
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

print("\nLoading final model...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ------------------------------------------------------------
# 3. Keep only labeled hospitals
# ------------------------------------------------------------

target = "rating_category"

df = df[df[target].notna()].copy()

print("\nLabeled hospitals:", len(df))

print("\nTarget distribution:")
print(df[target].value_counts())


# ------------------------------------------------------------
# 4. Remove target leakage and metadata
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
# 5. Recreate the exact Experiment 4 split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "-" * 70)
print("DATA SPLIT")
print("-" * 70)

print("Training samples:", len(X_train))
print("Testing samples: ", len(X_test))
print("Features:        ", X.shape[1])


# ------------------------------------------------------------
# 6. Verify model predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

print("Predictions generated successfully.")


# ------------------------------------------------------------
# 7. Verify prediction classes
# ------------------------------------------------------------

print("\nModel classes:")
print(pipeline.named_steps["model"].classes_)


# ------------------------------------------------------------
# 8. Extract preprocessing information
# ------------------------------------------------------------

preprocessor = pipeline.named_steps["preprocessor"]

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ------------------------------------------------------------
# 9. Transform test data
# ------------------------------------------------------------

print("\nTransforming test data for XAI...")

X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

print(
    "Transformed feature matrix:",
    X_test_transformed.shape
)

print(
    "Transformed feature count:",
    len(feature_names)
)


# ------------------------------------------------------------
# 10. Fairness groups
# ------------------------------------------------------------

# State is used as the primary available grouping attribute.
# The dataset does not contain individual demographic
# attributes such as gender or race.

sensitive_attribute = X_test["state"].copy()

print("\nFairness grouping attribute: state")

print("\nState distribution in test set:")
print(sensitive_attribute.value_counts().sort_index())


# ------------------------------------------------------------
# 11. Create prediction dataframe
# ------------------------------------------------------------

prediction_df = X_test.copy()

prediction_df["actual_rating"] = y_test.values
prediction_df["predicted_rating"] = y_pred

# Add prediction probabilities
for i, class_name in enumerate(
    pipeline.named_steps["model"].classes_
):
    prediction_df[
        f"probability_{class_name.lower()}"
    ] = y_proba[:, i]


# ------------------------------------------------------------
# 12. Save prediction data
# ------------------------------------------------------------

prediction_path = os.path.join(
    REPORT_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print("\nPredictions saved to:")
print(os.path.abspath(prediction_path))


# ------------------------------------------------------------
# 13. Save transformed feature names
# ------------------------------------------------------------

feature_names_path = os.path.join(
    REPORT_DIR,
    "transformed_feature_names.csv"
)

pd.DataFrame({
    "feature_name": feature_names
}).to_csv(
    feature_names_path,
    index=False
)

print("\nFeature names saved to:")
print(os.path.abspath(feature_names_path))


# ------------------------------------------------------------
# 14. Save fairness group data
# ------------------------------------------------------------

fairness_path = os.path.join(
    REPORT_DIR,
    "fairness_groups.csv"
)

pd.DataFrame({
    "state": sensitive_attribute.values,
    "actual_rating": y_test.values,
    "predicted_rating": y_pred
}).to_csv(
    fairness_path,
    index=False
)

print("\nFairness data saved to:")
print(os.path.abspath(fairness_path))


# ------------------------------------------------------------
# 15. Validation
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("VALIDATION")
print("-" * 70)

print(
    "Test rows match predictions:",
    len(X_test) == len(y_pred)
)

print(
    "Test rows match probabilities:",
    len(X_test) == len(y_proba)
)

print(
    "No missing target:",
    y_test.isna().sum() == 0
)

print(
    "State values available:",
    sensitive_attribute.notna().all()
)


# ------------------------------------------------------------
# 16. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 1 DATASET & MODEL PREPARATION COMPLETED")
print("=" * 70)