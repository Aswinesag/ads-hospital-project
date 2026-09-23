from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/best_hospital_quality_model.pkl")


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Saved model file does not exist."


def test_model_loads():
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_pipeline_structure():
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "named_steps")
    assert "preprocessor" in model.named_steps
    assert "model" in model.named_steps


def test_model_has_expected_features():
    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]

    numeric_features = list(
        preprocessor.transformers_[0][2]
    )

    categorical_features = list(
        preprocessor.transformers_[1][2]
    )

    total_features = len(numeric_features) + len(categorical_features)

    assert total_features == 34


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "city": "Austin",
        "state": "TX",
        "hospital_type": "Acute Care Hospitals",
        "emergency_services": "Yes",
        "patient_survey_rating": 3.0,
        "affiliated_clinicians": 400,
        "hospital_ownership": "Voluntary non-profit - Private",
        "county": "TRAVIS",
        "mort_group_measure_count": 10,
        "facility_mort_measures": 10,
        "mort_better": 3,
        "mort_no_different": 6,
        "mort_worse": 1,
        "safety_group_measure_count": 10,
        "facility_safety_measures": 10,
        "safety_better": 3,
        "safety_no_different": 6,
        "safety_worse": 1,
        "readm_group_measure_count": 10,
        "facility_readm_measures": 10,
        "readm_better": 3,
        "readm_no_different": 6,
        "readm_worse": 1,
        "patient_exp_group_measure_count": 8,
        "facility_patient_exp_measures": 8,
        "te_group_measure_count": 8,
        "facility_te_measures": 8,
        "total_better_measures": 9,
        "total_worse_measures": 3,
        "quality_balance_score": 6,
        "better_measure_ratio": 0.30,
        "worse_measure_ratio": 0.10,
        "quality_coverage": 46,
        "hospital_size_category": "Medium"
    }])

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in ["High", "Low", "Medium"]


def test_prediction_probabilities():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "city": "Austin",
        "state": "TX",
        "hospital_type": "Acute Care Hospitals",
        "emergency_services": "Yes",
        "patient_survey_rating": 3.0,
        "affiliated_clinicians": 400,
        "hospital_ownership": "Voluntary non-profit - Private",
        "county": "TRAVIS",
        "mort_group_measure_count": 10,
        "facility_mort_measures": 10,
        "mort_better": 3,
        "mort_no_different": 6,
        "mort_worse": 1,
        "safety_group_measure_count": 10,
        "facility_safety_measures": 10,
        "safety_better": 3,
        "safety_no_different": 6,
        "safety_worse": 1,
        "readm_group_measure_count": 10,
        "facility_readm_measures": 10,
        "readm_better": 3,
        "readm_no_different": 6,
        "readm_worse": 1,
        "patient_exp_group_measure_count": 8,
        "facility_patient_exp_measures": 8,
        "te_group_measure_count": 8,
        "facility_te_measures": 8,
        "total_better_measures": 9,
        "total_worse_measures": 3,
        "quality_balance_score": 6,
        "better_measure_ratio": 0.30,
        "worse_measure_ratio": 0.10,
        "quality_coverage": 46,
        "hospital_size_category": "Medium"
    }])

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 3)
    assert abs(probabilities.sum() - 1.0) < 1e-6
    assert all(0 <= value <= 1 for value in probabilities[0])