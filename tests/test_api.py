from fastapi.testclient import TestClient

from deployment.app import app

client = TestClient(app)


SAMPLE_HOSPITAL = {
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
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json=SAMPLE_HOSPITAL
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_rating_category" in data
    assert "probabilities" in data

    assert data["predicted_rating_category"] in [
        "High",
        "Low",
        "Medium"
    ]


def test_predict_probabilities():
    response = client.post(
        "/predict",
        json=SAMPLE_HOSPITAL
    )

    assert response.status_code == 200

    probabilities = response.json()["probabilities"]

    assert set(probabilities.keys()) == {
        "High",
        "Low",
        "Medium"
    }

    assert abs(sum(probabilities.values()) - 1.0) < 1e-6

    for probability in probabilities.values():
        assert 0 <= probability <= 1