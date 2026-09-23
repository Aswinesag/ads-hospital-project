from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# EXPERIMENT 6 - FASTAPI APPLICATION
# Hospital Quality Prediction API
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "best_hospital_quality_model.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_hospital_quality_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Create FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="Hospital Quality Prediction API",
    description="API for predicting hospital quality categories.",
    version="1.0.0"
)


# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class HospitalInput(BaseModel):

    # Numerical features
    patient_survey_rating: float | None = None
    affiliated_clinicians: float | None = None

    mort_group_measure_count: float | None = None
    facility_mort_measures: float | None = None
    mort_better: float | None = None
    mort_no_different: float | None = None
    mort_worse: float | None = None

    safety_group_measure_count: float | None = None
    facility_safety_measures: float | None = None
    safety_better: float | None = None
    safety_no_different: float | None = None
    safety_worse: float | None = None

    readm_group_measure_count: float | None = None
    facility_readm_measures: float | None = None
    readm_better: float | None = None
    readm_no_different: float | None = None
    readm_worse: float | None = None

    patient_exp_group_measure_count: float | None = None
    facility_patient_exp_measures: float | None = None

    te_group_measure_count: float | None = None
    facility_te_measures: float | None = None

    total_better_measures: float | None = None
    total_worse_measures: float | None = None
    quality_balance_score: float | None = None
    better_measure_ratio: float | None = None
    worse_measure_ratio: float | None = None
    quality_coverage: float | None = None

    # Categorical features
    city: str
    state: str
    hospital_type: str
    emergency_services: str
    hospital_ownership: str
    county: str
    hospital_size_category: str


# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Hospital Quality Prediction API is running"
    }


# ------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------

@app.post("/predict")
def predict(hospital: HospitalInput):

    try:

        # Convert request to dictionary
        data = hospital.model_dump()

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Generate probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Get class names
        classes = model.classes_

        probability_dict = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }

        return {
            "predicted_rating_category": str(prediction),
            "probabilities": probability_dict
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )