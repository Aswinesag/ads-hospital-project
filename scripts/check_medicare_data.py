from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/medicare_raw.csv")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "data/medicare_raw.csv was not found. "
            "Run medicare_scraper.py first."
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={"provider_id": str}
    )

    important_columns = [
        "provider_id",
        "hospital_name",
        "city",
        "state",
        "hospital_type",
        "emergency_services",
        "overall_rating",
        "patient_survey_rating",
        "affiliated_clinicians",
        "scraping_status",
    ]

    existing_columns = [
        column
        for column in important_columns
        if column in df.columns
    ]

    print("\nDATASET SHAPE")
    print(df.shape)

    print("\nCOLUMN COMPLETENESS")
    completeness = (
        df[existing_columns]
        .notna()
        .sum()
        .to_frame("non_missing_rows")
    )

    completeness["missing_rows"] = (
        len(df) - completeness["non_missing_rows"]
    )

    completeness["completion_percent"] = (
        completeness["non_missing_rows"]
        / len(df)
        * 100
    ).round(2)

    print(completeness)

    print("\nHOSPITAL PREVIEW")
    print(
        df[existing_columns]
        .to_string(index=False)
    )

    if "scraping_status" in df.columns:
        print("\nSCRAPING STATUS")
        print(
            df["scraping_status"]
            .value_counts(dropna=False)
        )

    if "overall_rating" in df.columns:
        print("\nOVERALL RATING DISTRIBUTION")
        print(
            df["overall_rating"]
            .value_counts(dropna=False)
            .sort_index()
        )


if __name__ == "__main__":
    main()