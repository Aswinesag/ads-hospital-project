from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/medicare_enriched.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("DATA PROFILING - HOSPITAL QUALITY DATASET")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    dtype={"provider_id": str}
)

print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# BASIC DATA PROFILE
# ============================================================

profile_rows = []

for column in df.columns:

    series = df[column]

    missing_count = int(series.isna().sum())
    missing_percent = round(
        (missing_count / len(df)) * 100,
        2
    )

    unique_count = int(series.nunique(dropna=True))

    profile_rows.append(
        {
            "column": column,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "is_constant": unique_count <= 1,
        }
    )

profile_df = pd.DataFrame(profile_rows)

profile_df.to_csv(
    REPORT_DIR / "data_profile.csv",
    index=False
)

print("\nSaved: reports/data_profile.csv")


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

missing_df = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (
        df.isna().mean().values * 100
    ).round(2),
})

missing_df = missing_df.sort_values(
    "missing_percent",
    ascending=False
)

missing_df.to_csv(
    REPORT_DIR / "missing_values.csv",
    index=False
)

print("Saved: reports/missing_values.csv")


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

duplicate_rows = int(df.duplicated().sum())

duplicate_provider_ids = (
    int(df["provider_id"].duplicated().sum())
    if "provider_id" in df.columns
    else None
)

duplicate_summary = pd.DataFrame(
    [
        {
            "metric": "duplicate_rows",
            "count": duplicate_rows,
        },
        {
            "metric": "duplicate_provider_ids",
            "count": duplicate_provider_ids,
        },
    ]
)

duplicate_summary.to_csv(
    REPORT_DIR / "duplicate_summary.csv",
    index=False
)

print("Saved: reports/duplicate_summary.csv")


# ============================================================
# UNIQUE VALUE ANALYSIS
# ============================================================

unique_df = pd.DataFrame({
    "column": df.columns,
    "unique_values": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ],
})

unique_df = unique_df.sort_values(
    "unique_values",
    ascending=True
)

unique_df.to_csv(
    REPORT_DIR / "unique_values.csv",
    index=False
)

print("Saved: reports/unique_values.csv")


# ============================================================
# NUMERIC STATISTICS
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

if numeric_columns:

    numeric_statistics = (
        df[numeric_columns]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "column"})
    )

    numeric_statistics["median"] = [
        df[column].median()
        for column in numeric_columns
    ]

    numeric_statistics["missing_count"] = [
        df[column].isna().sum()
        for column in numeric_columns
    ]

    numeric_statistics.to_csv(
        REPORT_DIR / "numeric_statistics.csv",
        index=False
    )

    print("Saved: reports/numeric_statistics.csv")


# ============================================================
# CATEGORICAL STATISTICS
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

categorical_records = []

for column in categorical_columns:

    series = df[column]

    mode_values = series.mode(dropna=True)

    mode_value = (
        mode_values.iloc[0]
        if not mode_values.empty
        else None
    )

    mode_count = (
        int((series == mode_value).sum())
        if mode_value is not None
        else 0
    )

    categorical_records.append(
        {
            "column": column,
            "unique_values": int(
                series.nunique(dropna=True)
            ),
            "missing_count": int(
                series.isna().sum()
            ),
            "most_frequent_value": mode_value,
            "most_frequent_count": mode_count,
        }
    )

categorical_df = pd.DataFrame(
    categorical_records
)

categorical_df.to_csv(
    REPORT_DIR / "categorical_statistics.csv",
    index=False
)

print("Saved: reports/categorical_statistics.csv")


# ============================================================
# IQR OUTLIER DETECTION
# ============================================================

outlier_records = []

for column in numeric_columns:

    series = df[column].dropna()

    if series.empty:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound)
        | (series > upper_bound)
    ]

    outlier_records.append(
        {
            "column": column,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outliers),
            "outlier_percent": round(
                len(outliers)
                / len(series)
                * 100,
                2
            ),
        }
    )

outlier_df = pd.DataFrame(
    outlier_records
)

outlier_df.to_csv(
    REPORT_DIR / "outlier_report.csv",
    index=False
)

print("Saved: reports/outlier_report.csv")


# ============================================================
# HIGH-MISSING COLUMNS
# ============================================================

high_missing_df = missing_df[
    missing_df["missing_percent"] >= 50
].copy()

high_missing_df.to_csv(
    REPORT_DIR / "high_missing_columns.csv",
    index=False
)

print("Saved: reports/high_missing_columns.csv")


# ============================================================
# CONSTANT / NEAR-CONSTANT COLUMNS
# ============================================================

constant_df = profile_df[
    profile_df["is_constant"] == True
].copy()

constant_df.to_csv(
    REPORT_DIR / "constant_columns.csv",
    index=False
)

print("Saved: reports/constant_columns.csv")


# ============================================================
# MISSING VALUE MATRIX
# ============================================================

plt.figure(figsize=(16, 8))

msno.matrix(df)

plt.savefig(
    REPORT_DIR / "missing_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: reports/missing_matrix.png")


# ============================================================
# MISSING VALUE BAR CHART
# ============================================================

plt.figure(figsize=(14, 8))

missing_plot_df = missing_df[
    missing_df["missing_count"] > 0
]

plt.barh(
    missing_plot_df["column"],
    missing_plot_df["missing_percent"]
)

plt.xlabel("Missing Values (%)")
plt.ylabel("Columns")
plt.title("Missing Value Percentage by Column")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "missing_percentage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: reports/missing_percentage.png")


# ============================================================
# CORRELATION MATRIX
# ============================================================

if len(numeric_columns) > 1:

    correlation_matrix = (
        df[numeric_columns]
        .corr(numeric_only=True)
    )

    correlation_matrix.to_csv(
        REPORT_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(14, 12))

    plt.imshow(
        correlation_matrix,
        aspect="auto"
    )

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.title(
        "Numeric Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: reports/correlation_matrix.csv")
    print("Saved: reports/correlation_matrix.png")


# ============================================================
# TOP VALUE DISTRIBUTIONS
# ============================================================

distribution_records = []

for column in categorical_columns:

    value_counts = (
        df[column]
        .value_counts(dropna=False)
        .head(10)
    )

    for value, count in value_counts.items():

        distribution_records.append(
            {
                "column": column,
                "value": value,
                "count": int(count),
                "percent": round(
                    count / len(df) * 100,
                    2
                ),
            }
        )

distribution_df = pd.DataFrame(
    distribution_records
)

distribution_df.to_csv(
    REPORT_DIR / "categorical_distributions.csv",
    index=False
)

print("Saved: reports/categorical_distributions.csv")


# ============================================================
# PROFILING SUMMARY TEXT FILE
# ============================================================

with open(
    REPORT_DIR / "profiling_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "HOSPITAL QUALITY DATASET PROFILING SUMMARY\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Rows: {df.shape[0]}\n"
    )

    file.write(
        f"Columns: {df.shape[1]}\n"
    )

    file.write(
        f"Duplicate Rows: {duplicate_rows}\n"
    )

    file.write(
        f"Duplicate Provider IDs: "
        f"{duplicate_provider_ids}\n\n"
    )

    file.write(
        "Columns with >= 50% missing values:\n"
    )

    if high_missing_df.empty:
        file.write("None\n")
    else:
        for _, row in high_missing_df.iterrows():

            file.write(
                f"- {row['column']}: "
                f"{row['missing_percent']}%\n"
            )

    file.write("\nConstant Columns:\n")

    if constant_df.empty:
        file.write("None\n")
    else:
        for column in constant_df["column"]:

            file.write(
                f"- {column}\n"
            )


print("Saved: reports/profiling_summary.txt")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROFILING SUMMARY")
print("=" * 70)

print(
    f"Dataset Shape              : "
    f"{df.shape[0]} rows x {df.shape[1]} columns"
)

print(
    f"Duplicate Rows             : "
    f"{duplicate_rows}"
)

print(
    f"Duplicate Provider IDs     : "
    f"{duplicate_provider_ids}"
)

print(
    f"Numeric Columns            : "
    f"{len(numeric_columns)}"
)

print(
    f"Categorical Columns        : "
    f"{len(categorical_columns)}"
)

print(
    f"Columns >= 50% Missing     : "
    f"{len(high_missing_df)}"
)

print(
    f"Constant Columns           : "
    f"{len(constant_df)}"
)

print("\nData profiling completed successfully.")