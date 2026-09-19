import os
import joblib
import pandas as pd


# ============================================================
# EXPERIMENT 6 - STAGE 1
# Inspect Final Experiment 4 Model
# ============================================================

print("=" * 70)
print("EXPERIMENT 6 - STAGE 1: MODEL INSPECTION & DEPLOYMENT SETUP")
print("=" * 70)

MODEL_PATH = "models/best_hospital_quality_model.pkl"


# ------------------------------------------------------------
# 1. Check model file
# ------------------------------------------------------------

print("\n[1] Checking model file...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print(f"Model found: {MODEL_PATH}")
print(f"Model size: {file_size:.2f} MB")


# ------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------

print("\n[2] Loading saved model...")

model = joblib.load(MODEL_PATH)

print(f"Model loaded successfully.")
print(f"Model type: {type(model).__name__}")


# ------------------------------------------------------------
# 3. Inspect pipeline
# ------------------------------------------------------------

print("\n[3] Inspecting pipeline...")

if not hasattr(model, "named_steps"):
    raise TypeError(
        "The saved model is not a scikit-learn Pipeline."
    )

print("Pipeline steps:")

for name, step in model.named_steps.items():
    print(f"  - {name}: {type(step).__name__}")


# ------------------------------------------------------------
# 4. Extract preprocessing information
# ------------------------------------------------------------

print("\n[4] Inspecting preprocessing...")

preprocessor = model.named_steps.get("preprocessor")

if preprocessor is None:
    raise ValueError(
        "Could not find 'preprocessor' step in the pipeline."
    )

numeric_features = list(
    preprocessor.transformers_[0][2]
)

categorical_features = list(
    preprocessor.transformers_[1][2]
)

all_features = numeric_features + categorical_features

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total input features: {len(all_features)}")


print("\nNumeric features:")
for feature in numeric_features:
    print(f"  - {feature}")

print("\nCategorical features:")
for feature in categorical_features:
    print(f"  - {feature}")


# ------------------------------------------------------------
# 5. Inspect transformed feature space
# ------------------------------------------------------------

print("\n[5] Inspecting transformed features...")

try:
    transformed_names = preprocessor.get_feature_names_out()

    print(
        f"Transformed feature count: "
        f"{len(transformed_names)}"
    )

except Exception as e:
    print(f"Could not retrieve transformed names: {e}")


# ------------------------------------------------------------
# 6. Inspect prediction classes
# ------------------------------------------------------------

print("\n[6] Inspecting prediction classes...")

classifier = model.named_steps.get("model")

if classifier is None:
    raise ValueError(
        "Could not find 'model' step in the pipeline."
    )

if hasattr(classifier, "classes_"):
    print("Classes:")
    for cls in classifier.classes_:
        print(f"  - {cls}")
else:
    print("Classifier classes unavailable before fitting.")


# ------------------------------------------------------------
# 7. Check prediction interface
# ------------------------------------------------------------

print("\n[7] Checking prediction interface...")

print(f"Has predict(): {hasattr(model, 'predict')}")
print(f"Has predict_proba(): {hasattr(model, 'predict_proba')}")


# ------------------------------------------------------------
# 8. Deployment folder check
# ------------------------------------------------------------

print("\n[8] Checking deployment folder...")

deployment_folder = "deployment"

if not os.path.exists(deployment_folder):
    os.makedirs(deployment_folder)

print(f"Deployment folder ready: {deployment_folder}")


# ------------------------------------------------------------
# 9. Save feature schema
# ------------------------------------------------------------

schema = pd.DataFrame({
    "feature_name": all_features,
    "feature_type": (
        ["numeric"] * len(numeric_features)
        + ["categorical"] * len(categorical_features)
    )
})

schema_path = "reports/experiment_6/input_feature_schema.csv"

schema.to_csv(schema_path, index=False)

print(f"Feature schema saved to: {schema_path}")


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STAGE 1 VALIDATION")
print("-" * 70)

checks = {
    "Model file exists": os.path.exists(MODEL_PATH),
    "Model loaded": model is not None,
    "Pipeline detected": hasattr(model, "named_steps"),
    "Preprocessor detected": preprocessor is not None,
    "34 input features": len(all_features) == 34,
    "3 prediction classes": len(classifier.classes_) == 3,
    "predict() available": hasattr(model, "predict"),
    "predict_proba() available": hasattr(model, "predict_proba"),
    "Deployment folder exists": os.path.exists(deployment_folder),
    "Feature schema created": os.path.exists(schema_path),
}

for check, result in checks.items():
    print(f"{check}: {'PASS' if result else 'FAIL'}")


if not all(checks.values()):
    raise RuntimeError("One or more Stage 1 validations failed.")

print("\n" + "=" * 70)
print("STAGE 1 MODEL INSPECTION & DEPLOYMENT SETUP COMPLETED")
print("=" * 70)