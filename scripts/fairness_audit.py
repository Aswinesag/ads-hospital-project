import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)


# ============================================================
# EXPERIMENT 5 - STAGE 4
# FAIRNESS AUDIT WITH FAIRLEARN
# ============================================================

print("=" * 70)
print("EXPERIMENT 5 - STAGE 4: FAIRNESS AUDIT")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATA_PATH = "data/medicare_feature_engineered.csv"
MODEL_PATH = "models/best_hospital_quality_model.pkl"

REPORT_DIR = "reports/experiment_5"
FAIRNESS_DIR = os.path.join(
    REPORT_DIR,
    "fairness"
)

os.makedirs(
    FAIRNESS_DIR,
    exist_ok=True
)


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

df = df[
    df[target].notna()
].copy()

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

X = df.drop(
    columns=drop_columns
)

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
# 6. Generate predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)

print("Predictions generated successfully.")


# ------------------------------------------------------------
# 7. Define sensitive/group attribute
# ------------------------------------------------------------

sensitive_attribute = X_test[
    "state"
].reset_index(drop=True)

y_test = y_test.reset_index(drop=True)

y_pred = pd.Series(
    y_pred
).reset_index(drop=True)


print("\nFairness grouping attribute: state")

print("\nState distribution:")

state_counts = (
    sensitive_attribute
    .value_counts()
    .sort_index()
)

print(state_counts)


# ------------------------------------------------------------
# 8. Define classes
# ------------------------------------------------------------

classes = [
    "High",
    "Low",
    "Medium"
]


# ------------------------------------------------------------
# 9. Minimum group size
# ------------------------------------------------------------

MIN_GROUP_SIZE = 10

small_groups = state_counts[
    state_counts < MIN_GROUP_SIZE
].index.tolist()

print("\nGroups with fewer than 10 test hospitals:")

if small_groups:
    print(small_groups)
else:
    print("None")


# ------------------------------------------------------------
# 10. Create fairness records
# ------------------------------------------------------------

fairness_records = []


for class_name in classes:

    print("\n" + "-" * 70)

    print(
        f"FAIRNESS ANALYSIS: "
        f"{class_name} VS NOT {class_name.upper()}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # Convert multiclass target into binary one-vs-rest
    # --------------------------------------------------------

    y_true_binary = (
        y_test == class_name
    ).astype(int)

    y_pred_binary = (
        y_pred == class_name
    ).astype(int)


    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall_accuracy = accuracy_score(
        y_true_binary,
        y_pred_binary
    )

    overall_selection_rate = (
        y_pred_binary.mean()
    )

    print(
        "Overall prediction/selection rate:",
        round(
            overall_selection_rate,
            4
        )
    )


    # --------------------------------------------------------
    # Fairlearn demographic parity difference
    # --------------------------------------------------------

    dp_difference = (
        demographic_parity_difference(
            y_true_binary,
            y_pred_binary,
            sensitive_features=sensitive_attribute
        )
    )


    # --------------------------------------------------------
    # Fairlearn equalized odds difference
    # --------------------------------------------------------

    eo_difference = (
        equalized_odds_difference(
            y_true_binary,
            y_pred_binary,
            sensitive_features=sensitive_attribute
        )
    )


    print(
        "Demographic parity difference:",
        round(
            dp_difference,
            4
        )
    )

    print(
        "Equalized odds difference:",
        round(
            eo_difference,
            4
        )
    )


    # --------------------------------------------------------
    # Group-level metrics
    # --------------------------------------------------------

    for group in sorted(
        sensitive_attribute.unique()
    ):

        mask = (
            sensitive_attribute == group
        )

        group_true = (
            y_true_binary[mask]
        )

        group_pred = (
            y_pred_binary[mask]
        )

        group_n = len(
            group_true
        )


        # Prediction rate
        selection_rate = (
            group_pred.mean()
        )


        # Accuracy
        group_accuracy = accuracy_score(
            group_true,
            group_pred
        )


        # Confusion matrix components
        tp = np.sum(
            (group_true == 1)
            & (group_pred == 1)
        )

        tn = np.sum(
            (group_true == 0)
            & (group_pred == 0)
        )

        fp = np.sum(
            (group_true == 0)
            & (group_pred == 1)
        )

        fn = np.sum(
            (group_true == 1)
            & (group_pred == 0)
        )


        # True positive rate
        if (tp + fn) > 0:
            tpr = tp / (
                tp + fn
            )
        else:
            tpr = np.nan


        # False positive rate
        if (fp + tn) > 0:
            fpr = fp / (
                fp + tn
            )
        else:
            fpr = np.nan


        fairness_records.append({

            "class": class_name,

            "state": group,

            "group_size": group_n,

            "small_group_flag": (
                group_n < MIN_GROUP_SIZE
            ),

            "accuracy": group_accuracy,

            "selection_rate": selection_rate,

            "true_positive_rate": tpr,

            "false_positive_rate": fpr,

            "overall_demographic_parity_difference":
                dp_difference,

            "overall_equalized_odds_difference":
                eo_difference

        })


# ------------------------------------------------------------
# 11. Convert results to dataframe
# ------------------------------------------------------------

fairness_df = pd.DataFrame(
    fairness_records
)


# ------------------------------------------------------------
# 12. Save detailed fairness report
# ------------------------------------------------------------

detailed_path = os.path.join(
    FAIRNESS_DIR,
    "fairness_group_metrics.csv"
)

fairness_df.to_csv(
    detailed_path,
    index=False
)

print(
    "\nDetailed fairness report saved to:"
)

print(
    os.path.abspath(
        detailed_path
    )
)


# ------------------------------------------------------------
# 13. Create class-level fairness summary
# ------------------------------------------------------------

summary_df = (
    fairness_df
    .groupby("class")
    .agg(
        demographic_parity_difference=(
            "overall_demographic_parity_difference",
            "first"
        ),
        equalized_odds_difference=(
            "overall_equalized_odds_difference",
            "first"
        ),
        minimum_group_accuracy=(
            "accuracy",
            "min"
        ),
        maximum_group_accuracy=(
            "accuracy",
            "max"
        ),
        minimum_selection_rate=(
            "selection_rate",
            "min"
        ),
        maximum_selection_rate=(
            "selection_rate",
            "max"
        )
    )
    .reset_index()
)


# Add accuracy disparity
summary_df[
    "accuracy_difference"
] = (
    summary_df[
        "maximum_group_accuracy"
    ]
    -
    summary_df[
        "minimum_group_accuracy"
    ]
)


# Add selection-rate disparity
summary_df[
    "selection_rate_difference"
] = (
    summary_df[
        "maximum_selection_rate"
    ]
    -
    summary_df[
        "minimum_selection_rate"
    ]
)


summary_path = os.path.join(
    FAIRNESS_DIR,
    "fairness_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print(
    "\nFairness summary saved to:"
)

print(
    os.path.abspath(
        summary_path
    )
)


# ------------------------------------------------------------
# 14. Display fairness summary
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("FAIRNESS SUMMARY")
print("-" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 15. Display group metrics
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("GROUP-LEVEL FAIRNESS METRICS")
print("-" * 70)

display_columns = [
    "class",
    "state",
    "group_size",
    "small_group_flag",
    "accuracy",
    "selection_rate",
    "true_positive_rate",
    "false_positive_rate"
]

print(
    fairness_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 16. Visualization - accuracy by state
# ------------------------------------------------------------

print("\nGenerating accuracy plot...")

for class_name in classes:

    plot_df = fairness_df[
        fairness_df["class"] == class_name
    ].copy()

    plot_df = plot_df.sort_values(
        "accuracy"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        plot_df["state"],
        plot_df["accuracy"]
    )

    plt.axhline(
        0.5,
        linewidth=1,
        linestyle="--"
    )

    plt.xlabel("State")

    plt.ylabel("Binary Classification Accuracy")

    plt.title(
        f"Fairness Audit - "
        f"{class_name} Prediction Accuracy by State"
    )

    plt.tight_layout()

    path = os.path.join(
        FAIRNESS_DIR,
        f"accuracy_by_state_{class_name.lower()}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 17. Visualization - selection rate
# ------------------------------------------------------------

print(
    "Generating selection-rate plots..."
)

for class_name in classes:

    plot_df = fairness_df[
        fairness_df["class"] == class_name
    ].copy()

    plot_df = plot_df.sort_values(
        "selection_rate"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        plot_df["state"],
        plot_df["selection_rate"]
    )

    plt.xlabel("State")

    plt.ylabel(
        "Prediction / Selection Rate"
    )

    plt.title(
        f"Prediction Rate for {class_name} "
        f"by State"
    )

    plt.tight_layout()

    path = os.path.join(
        FAIRNESS_DIR,
        f"selection_rate_{class_name.lower()}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 18. Overall validation
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("VALIDATION")
print("-" * 70)

print(
    "Fairness rows generated:",
    len(fairness_df)
)

print(
    "Expected rows:",
    len(classes) * len(state_counts)
)

print(
    "All demographic parity values finite:",
    np.isfinite(
        fairness_df[
            "overall_demographic_parity_difference"
        ]
    ).all()
)

print(
    "All equalized odds values finite:",
    np.isfinite(
        fairness_df[
            "overall_equalized_odds_difference"
        ]
    ).all()
)

print(
    "Small groups identified:",
    len(small_groups)
)


# ------------------------------------------------------------
# 19. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 4 FAIRNESS AUDIT COMPLETED")
print("=" * 70)