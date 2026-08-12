from __future__ import annotations

import json
from pathlib import Path

import great_expectations as gx
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/medicare_feature_engineered.csv")
REPORT_DIR = Path("reports")
OUTPUT_FILE = REPORT_DIR / "validation_results.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DATASET VALIDATION - GREAT EXPECTATIONS")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    dtype={
        "provider_id": str,
        "zip_code": str,
    },
)

print(
    f"\nDataset shape: "
    f"{df.shape[0]} rows x {df.shape[1]} columns"
)


# ============================================================
# GREAT EXPECTATIONS CONTEXT
# ============================================================

context = gx.get_context()

data_source = context.data_sources.add_pandas(
    "hospital_validation_source"
)

data_asset = data_source.add_dataframe_asset(
    name="hospital_feature_engineered_data"
)

batch_definition = (
    data_asset.add_batch_definition_whole_dataframe(
        "hospital_batch"
    )
)

batch = batch_definition.get_batch(
    batch_parameters={
        "dataframe": df
    }
)


# ============================================================
# VALIDATION HELPER
# ============================================================

results = []

def json_safe(value):
    """
    Convert NumPy/Pandas scalar values into native Python types
    so they can be serialized to JSON.
    """

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value

def run_expectation(
    expectation,
    description: str,
) -> None:

    validation_result = batch.validate(
        expectation
    )

    success = bool(
        validation_result.success
    )

    results.append(
        {
            "description": description,
            "success": success,
            "expectation_type":
                expectation.__class__.__name__,
        }
    )

    status = "PASS" if success else "FAIL"

    print(
        f"[{status}] {description}"
    )


# ============================================================
# 1. ROW COUNT VALIDATION
# ============================================================

run_expectation(
    gx.expectations.ExpectTableRowCountToEqual(
        value=835
    ),
    "Dataset must contain exactly 835 hospitals.",
)


# ============================================================
# 2. REQUIRED COLUMN VALIDATION
# ============================================================

required_columns = [
    "provider_id",
    "hospital_name",
    "state",
    "hospital_type",
    "hospital_ownership",
    "overall_rating",
    "patient_survey_rating",
    "affiliated_clinicians",

    "mort_better",
    "mort_worse",
    "safety_better",
    "safety_worse",
    "readm_better",
    "readm_worse",

    "total_better_measures",
    "total_worse_measures",
    "quality_balance_score",
    "better_measure_ratio",
    "worse_measure_ratio",
    "quality_coverage",
    "hospital_size_category",
    "rating_category",
]

for column in required_columns:

    run_expectation(
        gx.expectations.ExpectColumnToExist(
            column=column
        ),
        f"Required column '{column}' must exist.",
    )


# ============================================================
# 3. PROVIDER ID VALIDATION
# ============================================================

run_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="provider_id"
    ),
    "provider_id must not contain null values.",
)

run_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(
        column="provider_id"
    ),
    "provider_id must be unique.",
)

run_expectation(
    gx.expectations.ExpectColumnValuesToMatchRegex(
        column="provider_id",
        regex=r"^\d{5}[A-Z0-9]$",
    ),
    "provider_id must be a valid six-character CMS facility identifier.",
)


# ============================================================
# 4. TARGET RATING VALIDATION
# ============================================================

run_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="overall_rating",
        min_value=1,
        max_value=5,
        mostly=1.0,
    ),
    "Available overall_rating values must be between 1 and 5.",
)

run_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="patient_survey_rating",
        min_value=1,
        max_value=5,
        mostly=1.0,
    ),
    "Available patient_survey_rating values must be between 1 and 5.",
)


# ============================================================
# 5. VALID STATE VALUES
# ============================================================

valid_states = [
    "AZ",
    "CA",
    "CT",
    "NC",
    "NJ",
    "NM",
    "NY",
    "PA",
    "SC",
    "TX",
]

run_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="state",
        value_set=valid_states,
    ),
    "State must belong to the observed 10-state dataset.",
)


# ============================================================
# 6. EMERGENCY SERVICES VALIDATION
# ============================================================

run_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="emergency_services",
        value_set=[
            "Yes",
            "No",
        ],
    ),
    "emergency_services must contain only Yes or No.",
)


# ============================================================
# 7. CMS MEASURE COUNT VALIDATION
# ============================================================

non_negative_columns = [
    "mort_group_measure_count",
    "facility_mort_measures",
    "mort_better",
    "mort_no_different",
    "mort_worse",

    "safety_group_measure_count",
    "facility_safety_measures",
    "safety_better",
    "safety_no_different",
    "safety_worse",

    "readm_group_measure_count",
    "facility_readm_measures",
    "readm_better",
    "readm_no_different",
    "readm_worse",

    "patient_exp_group_measure_count",
    "facility_patient_exp_measures",

    "te_group_measure_count",
    "facility_te_measures",

    "total_better_measures",
    "total_worse_measures",
    "quality_coverage",
]

for column in non_negative_columns:

    if column in df.columns:

        run_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
                mostly=1.0,
            ),
            f"{column} must contain non-negative values.",
        )


# ============================================================
# 8. ENGINEERED RATIO VALIDATION
# ============================================================

for column in [
    "better_measure_ratio",
    "worse_measure_ratio",
]:

    run_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=0,
            max_value=1,
            mostly=1.0,
        ),
        f"{column} must be between 0 and 1.",
    )


# ============================================================
# 9. HOSPITAL SIZE CATEGORY VALIDATION
# ============================================================

run_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="hospital_size_category",
        value_set=[
            "Small",
            "Medium",
            "Large",
            "Unknown",
        ],
    ),
    "hospital_size_category must contain valid categories.",
)


# ============================================================
# 10. RATING CATEGORY VALIDATION
# ============================================================

# Missing rating categories are valid because overall_rating
# is unavailable for 239 hospitals.

non_null_rating_categories = (
    df["rating_category"]
    .dropna()
)

invalid_rating_categories = (
    ~non_null_rating_categories.isin(
        [
            "Low",
            "Medium",
            "High",
        ]
    )
).sum()

rating_category_success = (
    invalid_rating_categories == 0
)

results.append(
    {
        "description":
            "Available rating_category values must be "
            "Low, Medium, or High.",
        "success":
            rating_category_success,
        "expectation_type":
            "CustomMissingAwareCategoryValidation",
    }
)

print(
    "[PASS] Available rating_category values must be "
    "Low, Medium, or High."
    if rating_category_success
    else
    "[FAIL] Invalid rating_category values detected."
)


# ============================================================
# 11. RATING CATEGORY CONSISTENCY
# ============================================================

def expected_category(rating):

    if pd.isna(rating):
        return None

    if rating <= 2:
        return "Low"

    if rating == 3:
        return "Medium"

    return "High"


expected_categories = (
    df["overall_rating"]
    .apply(expected_category)
)

actual_categories = (
    df["rating_category"]
    .where(
        df["rating_category"].notna(),
        None
    )
)

category_consistency = (
    expected_categories.fillna("__MISSING__")
    == actual_categories.fillna("__MISSING__")
)

category_consistency_success = (
    category_consistency.all()
)

results.append(
    {
        "description":
            "rating_category must match overall_rating.",
        "success":
            bool(category_consistency_success),
        "expectation_type":
            "CustomTargetConsistencyValidation",
    }
)

print(
    "[PASS] rating_category is consistent with overall_rating."
    if category_consistency_success
    else
    "[FAIL] rating_category contains inconsistent labels."
)


# ============================================================
# 12. FINAL VALIDATION SUMMARY
# ============================================================

passed = sum(
    result["success"]
    for result in results
)

failed = len(results) - passed

summary = {
    "dataset": str(INPUT_FILE),
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "total_validations": int(len(results)),
    "passed": int(passed),
    "failed": int(failed),
    "success": bool(failed == 0),
    "results": [
        {
            key: json_safe(value)
            for key, value in result.items()
        }
        for result in results
    ],
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
    )


print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(
    f"Total validations : {len(results)}"
)

print(
    f"Passed            : {passed}"
)

print(
    f"Failed            : {failed}"
)

print(
    f"Overall status    : "
    f"{'PASS' if failed == 0 else 'FAIL'}"
)

print(
    f"\nValidation report saved to: "
    f"{OUTPUT_FILE}"
)