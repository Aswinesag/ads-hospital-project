import pandas as pd

# -----------------------------
# Load datasets
# -----------------------------

scraped = pd.read_csv(
    "data/medicare_raw.csv",
    dtype={"provider_id": str}
)

cms = pd.read_csv(
    "data/Hospital_General_Information.csv",
    dtype={"Facility ID": str}
)

# -----------------------------
# Normalize IDs
# -----------------------------

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

# -----------------------------
# Matching statistics
# -----------------------------

scraped_ids = set(scraped["provider_id"])
cms_ids = set(cms["Facility ID"])

matched = scraped_ids.intersection(cms_ids)
unmatched = scraped_ids.difference(cms_ids)

print("=" * 60)
print("MERGE VALIDATION")
print("=" * 60)

print(f"Scraped hospitals : {len(scraped_ids)}")
print(f"CMS hospitals      : {len(cms_ids)}")
print(f"Matched hospitals  : {len(matched)}")
print(f"Unmatched hospitals: {len(unmatched)}")

print()

print(
    f"Match Rate : "
    f"{len(matched)/len(scraped_ids)*100:.2f}%"
)

# -----------------------------
# Show unmatched hospitals
# -----------------------------

if unmatched:

    print("\nFirst unmatched hospitals:\n")

    unmatched_df = scraped[
        scraped["provider_id"].isin(unmatched)
    ][
        [
            "provider_id",
            "hospital_name",
            "city",
            "state",
        ]
    ]

    print(unmatched_df.head(20))

else:

    print("\nEvery scraped hospital exists in CMS.")