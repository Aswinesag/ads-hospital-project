import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATASET
# ============================================================

INPUT_PATH = "data/medicare_feature_engineered.csv"

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("EXPERIMENT 4 - STAGE 1: DATASET PREPARATION")
print("=" * 70)

print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# 2. KEEP ONLY LABELED HOSPITALS
# ============================================================

model_df = df[df["rating_category"].notna()].copy()

print(f"Labeled hospitals: {len(model_df)}")
print(f"Excluded unlabeled hospitals: {len(df) - len(model_df)}")

print("\nTarget distribution:")
print(model_df["rating_category"].value_counts())

print("\nTarget distribution (%):")
print(
    model_df["rating_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 3. DEFINE TARGET
# ============================================================

TARGET = "rating_category"

y = model_df[TARGET]


# ============================================================
# 4. REMOVE LEAKAGE / IDENTIFIER / METADATA COLUMNS
# ============================================================

exclude_columns = [
    # Target
    "rating_category",

    # Direct target leakage
    "overall_rating",

    # Raw representation of target source
    "overall_rating_raw",

    # Identifiers
    "provider_id",
    "hospital_name",
    "address",
    "phone_number",
    "profile_url",

    # Metadata
    "scraped_at",
    "data_last_updated",

    # Raw/duplicate representation
    "patient_survey_rating_raw",
    "all_star_rating_labels",

    # Location identifier rather than meaningful numeric quantity
    "zip_code",
]

existing_exclusions = [
    col for col in exclude_columns
    if col in model_df.columns
]

X = model_df.drop(columns=existing_exclusions)


# ============================================================
# 5. DISPLAY FINAL FEATURE SET
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print(f"\nNumber of predictor features: {X.shape[1]}")

print("\nPredictor columns:")
for i, column in enumerate(X.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print(f"\nNumerical features: {len(numerical_features)}")
for column in numerical_features:
    print(f"  - {column}")

print(f"\nCategorical features: {len(categorical_features)}")
for column in categorical_features:
    print(f"  - {column}")


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

missing = X.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\n" + "=" * 70)
print("MISSING VALUES IN MODELING FEATURES")
print("=" * 70)

if len(missing) == 0:
    print("\nNo missing values.")
else:
    print(missing)


# ============================================================
# 8. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")


# ============================================================
# 9. VERIFY STRATIFICATION
# ============================================================

print("\nTraining target distribution:")
print(
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting target distribution:")
print(
    y_test.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

assert "overall_rating" not in X.columns, \
    "Target leakage detected: overall_rating is present."

assert "rating_category" not in X.columns, \
    "Target column found inside predictors."

assert len(X) == len(y), \
    "Feature and target row counts do not match."

assert y.isnull().sum() == 0, \
    "Missing target values detected."

assert set(X_train.index).isdisjoint(set(X_test.index)), \
    "Train/test overlap detected."


print("\n" + "=" * 70)
print("STAGE 1 VALIDATION")
print("=" * 70)

print("""
[PASS] Target contains no missing values
[PASS] overall_rating removed to prevent target leakage
[PASS] rating_category removed from predictor matrix
[PASS] Train/test sets do not overlap
[PASS] Stratified train/test split completed
""")

print("Experiment 4 Stage 1 dataset preparation completed.")