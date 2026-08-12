from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/medicare_enriched.csv")
OUTPUT_FILE = Path("data/medicare_cleaned.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("DATA CLEANING - HOSPITAL QUALITY DATASET")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    dtype={
        "provider_id": str,
    }
)

original_rows = len(df)
original_columns = len(df.columns)

print(
    f"\nOriginal dataset shape: "
    f"{original_rows} rows x {original_columns} columns"
)


# ============================================================
# 1. REMOVE DUPLICATE ROWS
# ============================================================

duplicate_rows_before = int(df.duplicated().sum())

if duplicate_rows_before > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print(
    f"\nDuplicate rows found: "
    f"{duplicate_rows_before}"
)


# ============================================================
# 2. REMOVE DUPLICATE PROVIDER IDs
# ============================================================

duplicate_provider_ids_before = int(
    df["provider_id"].duplicated().sum()
)

if duplicate_provider_ids_before > 0:
    df = (
        df
        .drop_duplicates(
            subset=["provider_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

print(
    f"Duplicate provider IDs found: "
    f"{duplicate_provider_ids_before}"
)


# ============================================================
# 3. DROP COMPLETELY EMPTY / NON-USEFUL COLUMNS
# ============================================================

columns_to_drop = [
    "scraping_error",

    "timely_and_effective_care_text",
    "complications_and_deaths_text",
    "unplanned_hospital_visits_text",
    "maternal_health_text",
    "patient_reported_outcomes_text",
    "psychiatric_unit_services_text",
    "payment_text",

    "birthing_friendly",

    "quality_categories",
    "quality_section_text",
    "scraping_status",
]

existing_drop_columns = [
    column
    for column in columns_to_drop
    if column in df.columns
]

missing_drop_columns = [
    column
    for column in columns_to_drop
    if column not in df.columns
]

df = df.drop(
    columns=existing_drop_columns
)

print(
    f"\nDropped columns: "
    f"{len(existing_drop_columns)}"
)

for column in existing_drop_columns:
    print(f" - {column}")

if missing_drop_columns:
    print(
        "\nWarning: Expected columns not found:"
    )

    for column in missing_drop_columns:
        print(f" - {column}")


# ============================================================
# 4. NORMALIZE PROVIDER ID
# ============================================================

df["provider_id"] = (
    df["provider_id"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(6)
)


# ============================================================
# 5. CONVERT ZIP CODE TO STRING
# ============================================================

if "zip_code" in df.columns:
    df["zip_code"] = (
        df["zip_code"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


# ============================================================
# 6. CONVERT CMS "NOT AVAILABLE" VALUES TO NAN
# ============================================================

cms_measure_columns = [
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
]

available_cms_measure_columns = [
    column
    for column in cms_measure_columns
    if column in df.columns
]

for column in available_cms_measure_columns:

    df[column] = (
        df[column]
        .replace(
            {
                "Not Available": np.nan,
                "Not available": np.nan,
                "NOT AVAILABLE": np.nan,
                "": np.nan,
                " ": np.nan,
            }
        )
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


print(
    f"\nCMS measure columns converted to numeric: "
    f"{len(available_cms_measure_columns)}"
)


# ============================================================
# 7. PRESERVE TARGET MISSING VALUES
# ============================================================

if "overall_rating" in df.columns:
    df["overall_rating"] = pd.to_numeric(
        df["overall_rating"],
        errors="coerce"
    )

overall_rating_missing = (
    int(df["overall_rating"].isna().sum())
    if "overall_rating" in df.columns
    else None
)

print(
    f"\nMissing overall ratings preserved: "
    f"{overall_rating_missing}"
)


# ============================================================
# 8. PRESERVE PATIENT SURVEY MISSING VALUES
# ============================================================

if "patient_survey_rating" in df.columns:
    df["patient_survey_rating"] = pd.to_numeric(
        df["patient_survey_rating"],
        errors="coerce"
    )

patient_survey_missing = (
    int(
        df["patient_survey_rating"]
        .isna()
        .sum()
    )
    if "patient_survey_rating" in df.columns
    else None
)

print(
    f"Missing patient survey ratings preserved: "
    f"{patient_survey_missing}"
)


# ============================================================
# 9. PRESERVE AFFILIATED CLINICIAN MISSING VALUES
# ============================================================

if "affiliated_clinicians" in df.columns:
    df["affiliated_clinicians"] = pd.to_numeric(
        df["affiliated_clinicians"],
        errors="coerce"
    )

clinician_missing = (
    int(
        df["affiliated_clinicians"]
        .isna()
        .sum()
    )
    if "affiliated_clinicians" in df.columns
    else None
)

print(
    f"Missing affiliated clinician values preserved: "
    f"{clinician_missing}"
)


# ============================================================
# 10. KEEP OUTLIERS UNCHANGED
# ============================================================

print(
    "\nOutliers in patient survey ratings and "
    "affiliated clinician counts were retained."
)


# ============================================================
# 11. VALIDATE STATE VALUES
# ============================================================

if "state" in df.columns:

    state_values = sorted(
        df["state"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print(
        f"\nUnique states retained: "
        f"{len(state_values)}"
    )

    print(
        "States:",
        ", ".join(state_values)
    )


# ============================================================
# 12. CLEANING VALIDATION
# ============================================================

final_rows = len(df)
final_columns = len(df.columns)

duplicate_rows_after = int(
    df.duplicated().sum()
)

duplicate_provider_ids_after = int(
    df["provider_id"]
    .duplicated()
    .sum()
)

assert final_rows == original_rows, (
    "ERROR: Row count changed during cleaning."
)

assert duplicate_rows_after == 0, (
    "ERROR: Duplicate rows remain after cleaning."
)

assert duplicate_provider_ids_after == 0, (
    "ERROR: Duplicate provider IDs remain."
)

assert df["provider_id"].notna().all(), (
    "ERROR: Missing provider IDs detected."
)


# ============================================================
# 13. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. SAVE CLEANING SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "metric": "original_rows",
            "value": original_rows,
        },
        {
            "metric": "original_columns",
            "value": original_columns,
        },
        {
            "metric": "final_rows",
            "value": final_rows,
        },
        {
            "metric": "final_columns",
            "value": final_columns,
        },
        {
            "metric": "columns_dropped",
            "value": len(existing_drop_columns),
        },
        {
            "metric": "duplicate_rows_removed",
            "value": duplicate_rows_before,
        },
        {
            "metric": "duplicate_provider_ids_removed",
            "value": duplicate_provider_ids_before,
        },
        {
            "metric": "missing_overall_rating",
            "value": overall_rating_missing,
        },
        {
            "metric": "missing_patient_survey_rating",
            "value": patient_survey_missing,
        },
        {
            "metric": "missing_affiliated_clinicians",
            "value": clinician_missing,
        },
    ]
)

summary.to_csv(
    REPORT_DIR / "cleaning_summary.csv",
    index=False
)


# ============================================================
# 15. SAVE POST-CLEANING MISSING VALUE REPORT
# ============================================================

post_cleaning_missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (
        df.isna().mean().values * 100
    ).round(2),
})

post_cleaning_missing = (
    post_cleaning_missing
    .sort_values(
        "missing_percent",
        ascending=False
    )
)

post_cleaning_missing.to_csv(
    REPORT_DIR / "post_cleaning_missing_values.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED")
print("=" * 70)

print(
    f"Original shape : "
    f"{original_rows} x {original_columns}"
)

print(
    f"Final shape    : "
    f"{final_rows} x {final_columns}"
)

print(
    f"Columns dropped: "
    f"{len(existing_drop_columns)}"
)

print(
    f"Duplicate rows after cleaning: "
    f"{duplicate_rows_after}"
)

print(
    f"Duplicate provider IDs after cleaning: "
    f"{duplicate_provider_ids_after}"
)

print(
    f"\nCleaned dataset saved to: "
    f"{OUTPUT_FILE}"
)

print(
    "Cleaning summary saved to: "
    "reports/cleaning_summary.csv"
)

print(
    "Post-cleaning missing-value report saved to: "
    "reports/post_cleaning_missing_values.csv"
)