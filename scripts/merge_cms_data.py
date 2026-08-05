import pandas as pd


SCRAPED_FILE = "data/medicare_raw.csv"
CMS_FILE = "data/Hospital_General_Information.csv"
OUTPUT_FILE = "data/medicare_enriched.csv"


# =====================================================
# Load datasets
# =====================================================

scraped = pd.read_csv(
    SCRAPED_FILE,
    dtype={"provider_id": str},
)

cms = pd.read_csv(
    CMS_FILE,
    dtype=str,
)

# Remove accidental spaces or hidden BOM characters from headers.
scraped.columns = (
    scraped.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

cms.columns = (
    cms.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)


# =====================================================
# Validate merge-key columns
# =====================================================

if "provider_id" not in scraped.columns:
    raise KeyError(
        "provider_id was not found in the scraped dataset."
    )

if "Facility ID" not in cms.columns:
    raise KeyError(
        "Facility ID was not found in the CMS dataset.\n"
        f"Available CMS columns:\n{cms.columns.tolist()}"
    )


# =====================================================
# Normalize IDs
# =====================================================

scraped["provider_id"] = (
    scraped["provider_id"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(6)
)

cms["Facility ID"] = (
    cms["Facility ID"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(6)
)


# =====================================================
# Select useful CMS columns
# =====================================================

desired_columns = [
    "Facility ID",
    "Hospital Ownership",
    "County/Parish",
    "Meets criteria for birthing friendly designation",

    "MORT Group Measure Count",
    "Count of Facility MORT Measures",
    "Count of MORT Measures Better",
    "Count of MORT Measures No Different",
    "Count of MORT Measures Worse",

    "Safety Group Measure Count",
    "Count of Facility Safety Measures",
    "Count of Safety Measures Better",
    "Count of Safety Measures No Different",
    "Count of Safety Measures Worse",

    "READM Group Measure Count",
    "Count of Facility READM Measures",
    "Count of READM Measures Better",
    "Count of READM Measures No Different",
    "Count of READM Measures Worse",

    "Pt Exp Group Measure Count",
    "Count of Facility Pt Exp Measures",

    "TE Group Measure Count",
    "Count of Facility TE Measures",
]

available_columns = [
    column
    for column in desired_columns
    if column in cms.columns
]

missing_columns = [
    column
    for column in desired_columns
    if column not in cms.columns
]

if missing_columns:
    print("\nWarning: These CMS columns were not found:")

    for column in missing_columns:
        print(f" - {column}")

if "Facility ID" not in available_columns:
    raise KeyError(
        "Facility ID is required for merging but was not found."
    )

cms_selected = cms[available_columns].copy()


# =====================================================
# Rename CMS columns
# =====================================================

rename_map = {
    "Facility ID": "facility_id",
    "Hospital Ownership": "hospital_ownership",
    "County/Parish": "county",
    "Meets criteria for birthing friendly designation":
        "birthing_friendly",

    "MORT Group Measure Count":
        "mort_group_measure_count",
    "Count of Facility MORT Measures":
        "facility_mort_measures",
    "Count of MORT Measures Better":
        "mort_better",
    "Count of MORT Measures No Different":
        "mort_no_different",
    "Count of MORT Measures Worse":
        "mort_worse",

    "Safety Group Measure Count":
        "safety_group_measure_count",
    "Count of Facility Safety Measures":
        "facility_safety_measures",
    "Count of Safety Measures Better":
        "safety_better",
    "Count of Safety Measures No Different":
        "safety_no_different",
    "Count of Safety Measures Worse":
        "safety_worse",

    "READM Group Measure Count":
        "readm_group_measure_count",
    "Count of Facility READM Measures":
        "facility_readm_measures",
    "Count of READM Measures Better":
        "readm_better",
    "Count of READM Measures No Different":
        "readm_no_different",
    "Count of READM Measures Worse":
        "readm_worse",

    "Pt Exp Group Measure Count":
        "patient_exp_group_measure_count",
    "Count of Facility Pt Exp Measures":
        "facility_patient_exp_measures",

    "TE Group Measure Count":
        "te_group_measure_count",
    "Count of Facility TE Measures":
        "facility_te_measures",
}

cms_selected = cms_selected.rename(
    columns=rename_map
)


# =====================================================
# Check CMS key uniqueness
# =====================================================

duplicate_cms_ids = cms_selected[
    cms_selected["facility_id"].duplicated(
        keep=False
    )
]

if not duplicate_cms_ids.empty:
    raise ValueError(
        "Duplicate Facility IDs were found in the CMS dataset. "
        "A one-to-one merge cannot be performed safely."
    )


# =====================================================
# Merge datasets
# =====================================================

merged = scraped.merge(
    cms_selected,
    left_on="provider_id",
    right_on="facility_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)


# =====================================================
# Validate matches
# =====================================================

match_counts = merged["_merge"].value_counts()

matched_rows = int(
    match_counts.get("both", 0)
)

unmatched_rows = int(
    match_counts.get("left_only", 0)
)

match_rate = (
    matched_rows / len(scraped) * 100
    if len(scraped) > 0
    else 0
)

merged = merged.drop(
    columns=[
        "facility_id",
        "_merge",
    ]
)


# =====================================================
# Validate final dataset
# =====================================================

assert len(merged) == len(scraped), (
    "ERROR: Row count changed after merge."
)

assert not merged["provider_id"].duplicated().any(), (
    "ERROR: Duplicate provider IDs exist after merging."
)


# =====================================================
# Save output
# =====================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =====================================================
# Print summary
# =====================================================

print("\n" + "=" * 60)
print("MERGE SUMMARY")
print("=" * 60)

print(f"Scraped rows        : {len(scraped)}")
print(f"CMS rows            : {len(cms)}")
print(f"Matched rows        : {matched_rows}")
print(f"Unmatched rows      : {unmatched_rows}")
print(f"Match rate          : {match_rate:.2f}%")
print(f"Merged rows         : {len(merged)}")
print(f"Merged columns      : {len(merged.columns)}")
print(f"CMS columns imported: {len(cms_selected.columns) - 1}")

print("\n✓ Row count validation passed.")
print("✓ Provider IDs remain unique.")
print(f"✓ Dataset saved to: {OUTPUT_FILE}")