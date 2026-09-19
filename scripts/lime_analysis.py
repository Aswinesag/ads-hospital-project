import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# EXPERIMENT 5 - STAGE 3
# LIME LOCAL EXPLAINABILITY
# ============================================================

print("=" * 70)
print("EXPERIMENT 5 - STAGE 3: LIME LOCAL EXPLAINABILITY")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATA_PATH = "data/medicare_feature_engineered.csv"
MODEL_PATH = "models/best_hospital_quality_model.pkl"

REPORT_DIR = "reports/experiment_5"
LIME_DIR = os.path.join(REPORT_DIR, "lime_explanations")

os.makedirs(LIME_DIR, exist_ok=True)


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

print("Classes:", classifier.classes_)


# ------------------------------------------------------------
# 7. Transform data
# ------------------------------------------------------------

print("\nTransforming data...")

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

X_train_transformed = np.asarray(
    X_train_transformed
)

X_test_transformed = np.asarray(
    X_test_transformed
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

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
# 8. Generate predictions
# ------------------------------------------------------------

y_pred = pipeline.predict(X_test)

y_proba = pipeline.predict_proba(X_test)


# ------------------------------------------------------------
# 9. Create LIME explainer
# ------------------------------------------------------------

print("\nCreating LIME explainer...")

explainer = LimeTabularExplainer(
    X_train_transformed,
    feature_names=feature_names,
    class_names=classifier.classes_.tolist(),
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

print("LIME explainer created successfully.")


# ------------------------------------------------------------
# 10. Select representative examples
# ------------------------------------------------------------

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred
})

results["correct"] = (
    results["actual"] == results["predicted"]
)

selected_indices = {}

for class_name in classifier.classes_:

    # Prefer correctly classified examples
    correct_indices = results.index[
        (results["predicted"] == class_name)
        & (results["correct"])
    ].tolist()

    if correct_indices:
        selected_indices[class_name] = (
            correct_indices[0]
        )
        continue

    # Otherwise use any prediction of that class
    predicted_indices = results.index[
        results["predicted"] == class_name
    ].tolist()

    if predicted_indices:
        selected_indices[class_name] = (
            predicted_indices[0]
        )
        continue

    # Final fallback: any test example
    selected_indices[class_name] = 0


print("\nSelected representative examples:")

for class_name, index in selected_indices.items():

    print(
        f"{class_name}: "
        f"test index {index}, "
        f"actual={results.loc[index, 'actual']}, "
        f"predicted={results.loc[index, 'predicted']}"
    )


# ------------------------------------------------------------
# 11. Generate local explanations
# ------------------------------------------------------------

all_explanations = []

for class_name, index in selected_indices.items():

    print("\n" + "-" * 70)
    print(
        f"Generating LIME explanation: "
        f"{class_name}"
    )
    print("-" * 70)

    instance = X_test_transformed[index]

    explanation = explainer.explain_instance(
        instance,
        classifier.predict_proba,
        num_features=10,
        top_labels=3
    )

    predicted_class = y_pred[index]
    actual_class = y_test.iloc[index]

    print("Actual class:    ", actual_class)
    print("Predicted class: ", predicted_class)

    print("\nTop local features:")

    # Get explanation for predicted class
    predicted_class_index = list(
        classifier.classes_
    ).index(predicted_class)

    explanation_list = explanation.as_list(
        label=predicted_class_index
    )

    for feature, contribution in explanation_list:

        print(
            f"{feature:<60} "
            f"{contribution:+.6f}"
        )

        all_explanations.append({
            "example_class": class_name,
            "test_index": index,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "feature": feature,
            "contribution": contribution
        })


    # --------------------------------------------------------
    # Save HTML explanation
    # --------------------------------------------------------

    html_path = os.path.join(
        LIME_DIR,
        f"lime_{class_name.lower()}.html"
    )

    explanation.save_to_file(
        html_path
    )

    print(
        "\nHTML explanation saved to:"
    )

    print(
        os.path.abspath(html_path)
    )


    # --------------------------------------------------------
    # Create static PNG visualization
    # --------------------------------------------------------

    features = [
        item[0]
        for item in explanation_list
    ]

    contributions = [
        item[1]
        for item in explanation_list
    ]

    # Reverse for horizontal bar chart
    features = features[::-1]
    contributions = contributions[::-1]

    plt.figure(figsize=(10, 7))

    plt.barh(
        features,
        contributions
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "LIME Contribution"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"LIME Explanation - "
        f"Predicted {predicted_class}"
    )

    plt.tight_layout()

    png_path = os.path.join(
        LIME_DIR,
        f"lime_{class_name.lower()}.png"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "PNG explanation saved to:"
    )

    print(
        os.path.abspath(png_path)
    )


# ------------------------------------------------------------
# 12. Save combined explanation data
# ------------------------------------------------------------

explanation_df = pd.DataFrame(
    all_explanations
)

csv_path = os.path.join(
    REPORT_DIR,
    "lime_explanations.csv"
)

explanation_df.to_csv(
    csv_path,
    index=False
)

print(
    "\nCombined LIME results saved to:"
)

print(
    os.path.abspath(csv_path)
)


# ------------------------------------------------------------
# 13. Validation
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("VALIDATION")
print("-" * 70)

print(
    "Three class explanations generated:",
    len(selected_indices) == 3
)

print(
    "Explanation rows generated:",
    len(explanation_df)
)

print(
    "Prediction count matches test rows:",
    len(y_pred) == len(X_test)
)


# ------------------------------------------------------------
# 14. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 3 LIME LOCAL EXPLAINABILITY COMPLETED")
print("=" * 70)