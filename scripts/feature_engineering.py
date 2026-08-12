from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/medicare_cleaned.csv")
OUTPUT_FILE = Path("data/medicare_feature_engineered.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("FEATURE ENGINEERING - HOSPITAL QUALITY DATASET")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    dtype={
        "provider_id": str,
        "zip_code": str,
    }
)

original_rows = len(df)
original_columns = len(df.columns)

print(
    f"\nInput dataset shape: "
    f"{original_rows} rows x {original_columns} columns"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "mort_better",
    "safety_better",
    "readm_better",

    "mort_worse",
    "safety_worse",
    "readm_worse",

    "facility_mort_measures",
    "facility_safety_measures",
    "facility_readm_measures",
    "facility_patient_exp_measures",
    "facility_te_measures",

    "affiliated_clinicians",
    "overall_rating",
]

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise KeyError(
        "Required columns missing from cleaned dataset: "
        + ", ".join(missing_required)
    )


# ============================================================
# ENSURE NUMERIC TYPES
# ============================================================

numeric_columns = [
    "mort_better",
    "safety_better",
    "readm_better",

    "mort_worse",
    "safety_worse",
    "readm_worse",

    "facility_mort_measures",
    "facility_safety_measures",
    "facility_readm_measures",
    "facility_patient_exp_measures",
    "facility_te_measures",

    "affiliated_clinicians",
    "overall_rating",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 1. TOTAL BETTER MEASURES
# ============================================================

better_columns = [
    "mort_better",
    "safety_better",
    "readm_better",
]

df["total_better_measures"] = (
    df[better_columns]
    .sum(
        axis=1,
        min_count=1
    )
)


# ============================================================
# 2. TOTAL WORSE MEASURES
# ============================================================

worse_columns = [
    "mort_worse",
    "safety_worse",
    "readm_worse",
]

df["total_worse_measures"] = (
    df[worse_columns]
    .sum(
        axis=1,
        min_count=1
    )
)


# ============================================================
# 3. QUALITY BALANCE SCORE
# ============================================================

df["quality_balance_score"] = (
    df["total_better_measures"]
    - df["total_worse_measures"]
)


# ============================================================
# INTERNAL DENOMINATOR FOR NORMALIZED RATIOS
# ============================================================

# We only use mortality, safety and readmission measures here
# because those are the domains for which Better/Worse counts exist.

evaluated_outcome_measures = (
    df[
        [
            "facility_mort_measures",
            "facility_safety_measures",
            "facility_readm_measures",
        ]
    ]
    .sum(
        axis=1,
        min_count=1
    )
)

# Avoid division by zero.
valid_denominator = evaluated_outcome_measures.replace(
    0,
    np.nan
)


# ============================================================
# 4. BETTER MEASURE RATIO
# ============================================================

df["better_measure_ratio"] = (
    df["total_better_measures"]
    / valid_denominator
)


# ============================================================
# 5. WORSE MEASURE RATIO
# ============================================================

df["worse_measure_ratio"] = (
    df["total_worse_measures"]
    / valid_denominator
)


# ============================================================
# 6. QUALITY COVERAGE
# ============================================================

# Total number of CMS quality measures reported across:
# mortality, safety, readmission, patient experience,
# and timely/effective care.

coverage_columns = [
    "facility_mort_measures",
    "facility_safety_measures",
    "facility_readm_measures",
    "facility_patient_exp_measures",
    "facility_te_measures",
]

df["quality_coverage"] = (
    df[coverage_columns]
    .sum(
        axis=1,
        min_count=1
    )
)


# ============================================================
# 7. HOSPITAL SIZE CATEGORY
# ============================================================

# Use the distribution of affiliated clinicians rather than
# arbitrary fixed thresholds.

clinician_values = (
    df["affiliated_clinicians"]
    .dropna()
)

if clinician_values.empty:
    raise ValueError(
        "No affiliated clinician values are available."
    )

small_threshold = clinician_values.quantile(0.33)
large_threshold = clinician_values.quantile(0.67)

df["hospital_size_category"] = pd.cut(
    df["affiliated_clinicians"],
    bins=[
        -np.inf,
        small_threshold,
        large_threshold,
        np.inf,
    ],
    labels=[
        "Small",
        "Medium",
        "Large",
    ],
    include_lowest=True,
)

# Missing clinician count remains explicitly identifiable.
df["hospital_size_category"] = (
    df["hospital_size_category"]
    .astype("object")
    .fillna("Unknown")
)


# ============================================================
# 8. RATING CATEGORY
# ============================================================

# This is a derived TARGET variable.
#
# 1-2 stars -> Low
# 3 stars   -> Medium
# 4-5 stars -> High
#
# Missing overall ratings remain missing.

def create_rating_category(
    rating: float,
) -> str | float:

    if pd.isna(rating):
        return np.nan

    if 1 <= rating <= 2:
        return "Low"

    if rating == 3:
        return "Medium"

    if 4 <= rating <= 5:
        return "High"

    return np.nan


df["rating_category"] = (
    df["overall_rating"]
    .apply(create_rating_category)
)


# ============================================================
# VALIDATION
# ============================================================

engineered_features = [
    "total_better_measures",
    "total_worse_measures",
    "quality_balance_score",
    "better_measure_ratio",
    "worse_measure_ratio",
    "quality_coverage",
    "hospital_size_category",
    "rating_category",
]

final_rows = len(df)
final_columns = len(df.columns)

assert final_rows == original_rows, (
    "ERROR: Row count changed during feature engineering."
)

assert len(engineered_features) == 8, (
    "ERROR: Unexpected engineered feature count."
)

assert all(
    column in df.columns
    for column in engineered_features
), "ERROR: One or more engineered features are missing."


# Validate ratios where values exist.

better_ratio_valid = (
    df["better_measure_ratio"]
    .dropna()
    .between(0, 1)
    .all()
)

worse_ratio_valid = (
    df["worse_measure_ratio"]
    .dropna()
    .between(0, 1)
    .all()
)

if not better_ratio_valid:
    print(
        "\nWARNING: Some better_measure_ratio "
        "values fall outside 0-1."
    )

if not worse_ratio_valid:
    print(
        "\nWARNING: Some worse_measure_ratio "
        "values fall outside 0-1."
    )


# ============================================================
# SAVE FEATURE-ENGINEERED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

feature_summary = pd.DataFrame(
    [
        {
            "feature": "total_better_measures",
            "description":
                "Sum of mortality, safety and readmission "
                "measures rated better than comparison.",
        },
        {
            "feature": "total_worse_measures",
            "description":
                "Sum of mortality, safety and readmission "
                "measures rated worse than comparison.",
        },
        {
            "feature": "quality_balance_score",
            "description":
                "Total better measures minus total worse measures.",
        },
        {
            "feature": "better_measure_ratio",
            "description":
                "Better measures divided by available evaluated "
                "mortality, safety and readmission measures.",
        },
        {
            "feature": "worse_measure_ratio",
            "description":
                "Worse measures divided by available evaluated "
                "mortality, safety and readmission measures.",
        },
        {
            "feature": "quality_coverage",
            "description":
                "Total reported measures across mortality, safety, "
                "readmission, patient experience and timely care.",
        },
        {
            "feature": "hospital_size_category",
            "description":
                "Small, Medium or Large based on 33rd and 67th "
                "percentiles of affiliated clinician count; Unknown "
                "when clinician count is missing.",
        },
        {
            "feature": "rating_category",
            "description":
                "Derived target: 1-2 Low, 3 Medium, 4-5 High.",
        },
    ]
)

feature_summary.to_csv(
    REPORT_DIR / "feature_engineering_summary.csv",
    index=False
)


# ============================================================
# DISTRIBUTION REPORT
# ============================================================

rating_distribution = (
    df["rating_category"]
    .value_counts(
        dropna=False
    )
    .rename_axis("rating_category")
    .reset_index(name="count")
)

rating_distribution["percent"] = (
    rating_distribution["count"]
    / len(df)
    * 100
).round(2)

rating_distribution.to_csv(
    REPORT_DIR / "rating_category_distribution.csv",
    index=False
)


size_distribution = (
    df["hospital_size_category"]
    .value_counts(
        dropna=False
    )
    .rename_axis("hospital_size_category")
    .reset_index(name="count")
)

size_distribution["percent"] = (
    size_distribution["count"]
    / len(df)
    * 100
).round(2)

size_distribution.to_csv(
    REPORT_DIR / "hospital_size_distribution.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print(
    f"Input shape       : "
    f"{original_rows} x {original_columns}"
)

print(
    f"Output shape      : "
    f"{final_rows} x {final_columns}"
)

print(
    f"Features created  : "
    f"{len(engineered_features)}"
)

print("\nEngineered features:")

for feature in engineered_features:
    print(f" - {feature}")

print(
    "\nHospital size thresholds:"
)

print(
    f" Small  <= {small_threshold:.2f} clinicians"
)

print(
    f" Medium <= {large_threshold:.2f} clinicians"
)

print(
    f" Large  >  {large_threshold:.2f} clinicians"
)

print(
    f"\nFeature-engineered dataset saved to: "
    f"{OUTPUT_FILE}"
)

print(
    "Feature summary saved to: "
    "reports/feature_engineering_summary.csv"
)

print(
    "Rating distribution saved to: "
    "reports/rating_category_distribution.csv"
)

print(
    "Hospital size distribution saved to: "
    "reports/hospital_size_distribution.csv"
)