import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ============================================================
# EXPERIMENT 5 - STAGE 2
# SHAP EXPLAINABILITY
# ============================================================

print("=" * 70)
print("EXPERIMENT 5 - STAGE 2: SHAP EXPLAINABILITY")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATA_PATH = "data/medicare_feature_engineered.csv"
MODEL_PATH = "models/best_hospital_quality_model.pkl"

REPORT_DIR = "reports/experiment_5"
PLOT_DIR = os.path.join(REPORT_DIR, "shap_plots")

os.makedirs(PLOT_DIR, exist_ok=True)


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
# 3. Keep labeled hospitals
# ------------------------------------------------------------

target = "rating_category"

df = df[df[target].notna()].copy()

print("\nLabeled hospitals:", len(df))


# ------------------------------------------------------------
# 4. Prepare predictors
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
# 5. Recreate exact Experiment 4 split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples: ", len(X_test))


# ------------------------------------------------------------
# 6. Extract preprocessing and classifier
# ------------------------------------------------------------

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["model"]

print("\nModel:", type(classifier).__name__)

print("Model classes:")
print(classifier.classes_)


# ------------------------------------------------------------
# 7. Transform training and test data
# ------------------------------------------------------------

print("\nTransforming data...")

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

print(
    "Training transformed shape:",
    X_train_transformed.shape
)

print(
    "Testing transformed shape:",
    X_test_transformed.shape
)

print(
    "Transformed features:",
    len(feature_names)
)


# ------------------------------------------------------------
# 8. Convert to dense NumPy arrays
# ------------------------------------------------------------

X_train_transformed = np.asarray(
    X_train_transformed
)

X_test_transformed = np.asarray(
    X_test_transformed
)


# ------------------------------------------------------------
# 9. Create SHAP explainer
# ------------------------------------------------------------

print("\nCreating SHAP LinearExplainer...")

# Use a representative subset of training data
# as the SHAP background dataset.
background_size = min(100, len(X_train_transformed))

background = X_train_transformed[
    :background_size
]

explainer = shap.LinearExplainer(
    classifier,
    background
)

print("SHAP explainer created successfully.")


# ------------------------------------------------------------
# 10. Calculate SHAP values
# ------------------------------------------------------------

print("\nCalculating SHAP values...")

shap_values = explainer(
    X_test_transformed
)

print("SHAP values calculated successfully.")

print("SHAP value shape:", shap_values.values.shape)


# ------------------------------------------------------------
# 11. Handle multiclass SHAP values
# ------------------------------------------------------------

# SHAP 0.52 returns:
#
# samples × features × classes
#
# for this multiclass classification problem.

if shap_values.values.ndim != 3:
    raise ValueError(
        "Unexpected SHAP output shape: "
        + str(shap_values.values.shape)
    )

n_samples, n_features, n_classes = (
    shap_values.values.shape
)

print("\nSHAP dimensions:")
print("Samples:", n_samples)
print("Features:", n_features)
print("Classes:", n_classes)


# ------------------------------------------------------------
# 12. Global feature importance
# ------------------------------------------------------------

# Mean absolute SHAP value across:
# samples AND classes.

mean_abs_shap = np.mean(
    np.abs(shap_values.values),
    axis=(0, 2)
)

importance_df = pd.DataFrame({
    "feature_name": feature_names,
    "mean_absolute_shap": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "mean_absolute_shap",
    ascending=False
).reset_index(drop=True)

importance_df["rank"] = (
    importance_df.index + 1
)

# Reorder columns
importance_df = importance_df[
    [
        "rank",
        "feature_name",
        "mean_absolute_shap"
    ]
]


# ------------------------------------------------------------
# 13. Display top features
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("TOP 20 SHAP FEATURES")
print("-" * 70)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 14. Save feature importance
# ------------------------------------------------------------

importance_path = os.path.join(
    REPORT_DIR,
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance saved to:")
print(os.path.abspath(importance_path))


# ------------------------------------------------------------
# 15. Global SHAP summary plot
# ------------------------------------------------------------

print("\nGenerating SHAP summary plot...")


# Aggregate multiclass SHAP values across classes
# for a global explanation.

global_shap_values = np.mean(
    shap_values.values,
    axis=2
)

global_explanation = shap.Explanation(
    values=global_shap_values,
    base_values=np.mean(
        shap_values.base_values,
        axis=1
    ),
    data=X_test_transformed,
    feature_names=feature_names
)

plt.figure(figsize=(12, 9))

shap.plots.beeswarm(
    global_explanation,
    max_display=20,
    show=False
)

plt.title(
    "SHAP Global Feature Importance - Hospital Quality Model"
)

plt.tight_layout()

summary_path = os.path.join(
    PLOT_DIR,
    "shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Summary plot saved to:")
print(os.path.abspath(summary_path))


# ------------------------------------------------------------
# 16. Bar-style global importance plot
# ------------------------------------------------------------

print("\nGenerating SHAP bar plot...")

top_n = 20

top_features = importance_df.head(
    top_n
).sort_values(
    "mean_absolute_shap"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature_name"],
    top_features["mean_absolute_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")

plt.ylabel("Feature")

plt.title(
    "Top 20 SHAP Feature Importances"
)

plt.tight_layout()

bar_path = os.path.join(
    PLOT_DIR,
    "shap_feature_importance_bar.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Bar plot saved to:")
print(os.path.abspath(bar_path))


# ------------------------------------------------------------
# 17. Class-specific SHAP importance
# ------------------------------------------------------------

print("\nCalculating class-specific SHAP importance...")

class_importance_records = []

for class_index, class_name in enumerate(
    classifier.classes_
):

    class_values = shap_values.values[
        :, :, class_index
    ]

    class_mean_abs = np.mean(
        np.abs(class_values),
        axis=0
    )

    class_df = pd.DataFrame({
        "feature_name": feature_names,
        "mean_absolute_shap": class_mean_abs
    })

    class_df = class_df.sort_values(
        "mean_absolute_shap",
        ascending=False
    ).head(20)

    class_df["class"] = class_name
    class_df["rank"] = range(
        1,
        len(class_df) + 1
    )

    class_importance_records.append(
        class_df[
            [
                "class",
                "rank",
                "feature_name",
                "mean_absolute_shap"
            ]
        ]
    )


class_importance_df = pd.concat(
    class_importance_records,
    ignore_index=True
)

class_importance_path = os.path.join(
    REPORT_DIR,
    "shap_class_feature_importance.csv"
)

class_importance_df.to_csv(
    class_importance_path,
    index=False
)

print(
    "Class-specific importance saved to:"
)

print(
    os.path.abspath(
        class_importance_path
    )
)


# ------------------------------------------------------------
# 18. Validation
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("VALIDATION")
print("-" * 70)

print(
    "SHAP samples match test rows:",
    shap_values.values.shape[0]
    == len(X_test)
)

print(
    "SHAP features match transformed features:",
    shap_values.values.shape[1]
    == len(feature_names)
)

print(
    "SHAP classes match model classes:",
    shap_values.values.shape[2]
    == len(classifier.classes_)
)

print(
    "Feature importance rows:",
    len(importance_df)
)


# ------------------------------------------------------------
# 19. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 2 SHAP EXPLAINABILITY COMPLETED")
print("=" * 70)