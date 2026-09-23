from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy import stats

# ============================================================
# EXPERIMENT 8 - STREAMLIT DASHBOARD
# Hospital Quality Prediction & Responsible AI Dashboard
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "medicare_feature_engineered.csv"
MODEL_PATH = BASE_DIR / "models" / "best_hospital_quality_model.pkl"


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Hospital Quality Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .dashboard-subtitle {
            color: #9ca3af;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            background: #161b22;
            border: 1px solid #30363d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Load dataset and model
# ------------------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_dataset()
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load project resources: {exc}")
    st.stop()


# ------------------------------------------------------------
# Model feature schema
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "patient_survey_rating",
    "affiliated_clinicians",
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
    "total_better_measures",
    "total_worse_measures",
    "quality_balance_score",
    "better_measure_ratio",
    "worse_measure_ratio",
    "quality_coverage",
]

CATEGORICAL_FEATURES = [
    "city",
    "state",
    "hospital_type",
    "emergency_services",
    "hospital_ownership",
    "county",
    "hospital_size_category",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ------------------------------------------------------------
# Feature engineering helper
# ------------------------------------------------------------

def calculate_engineered_features(data):
    """Calculate the eight engineered features used by the model."""

    data = data.copy()

    better_columns = [
        "mort_better",
        "safety_better",
        "readm_better",
    ]

    worse_columns = [
        "mort_worse",
        "safety_worse",
        "readm_worse",
    ]

    facility_outcome_columns = [
        "facility_mort_measures",
        "facility_safety_measures",
        "facility_readm_measures",
    ]

    coverage_columns = [
        "facility_mort_measures",
        "facility_safety_measures",
        "facility_readm_measures",
        "facility_patient_exp_measures",
        "facility_te_measures",
    ]

    # Better/worse totals
    data["total_better_measures"] = data[better_columns].sum(
        axis=1,
        min_count=1,
    )

    data["total_worse_measures"] = data[worse_columns].sum(
        axis=1,
        min_count=1,
    )

    # Balance
    data["quality_balance_score"] = (
        data["total_better_measures"]
        - data["total_worse_measures"]
    )

    # Outcome denominator
    outcome_total = data[facility_outcome_columns].sum(
        axis=1,
        min_count=1,
    )

    data["better_measure_ratio"] = (
        data["total_better_measures"] / outcome_total
    )

    data["worse_measure_ratio"] = (
        data["total_worse_measures"] / outcome_total
    )

    # Overall quality-information coverage
    data["quality_coverage"] = data[coverage_columns].sum(
        axis=1,
        min_count=1,
    )

    # Hospital size category
    clinicians = pd.to_numeric(
        data["affiliated_clinicians"],
        errors="coerce",
    )

    q33 = df["affiliated_clinicians"].quantile(0.33)
    q67 = df["affiliated_clinicians"].quantile(0.67)

    data["hospital_size_category"] = "Unknown"

    data.loc[clinicians <= q33, "hospital_size_category"] = "Small"
    data.loc[
        (clinicians > q33) & (clinicians <= q67),
        "hospital_size_category",
    ] = "Medium"
    data.loc[
        clinicians > q67,
        "hospital_size_category",
    ] = "Large"

    return data


# ------------------------------------------------------------
# Prepare model input
# ------------------------------------------------------------

def prepare_model_input(data):
    """Keep exactly the 34 features expected by the trained model."""

    data = data.copy()

    for column in NUMERIC_FEATURES:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data[MODEL_FEATURES]


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("🏥 Hospital Quality")

st.sidebar.caption(
    "Hospital Quality Prediction & Responsible AI"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Predictions",
        "Quality Insights",
        "Model Performance",
        "SHAP Explainability",
        "Drift Checks",
        "Responsible AI",
        "Portfolio",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Project Status")
st.sidebar.success("Model available")
st.sidebar.success("Dataset available")

st.sidebar.caption(
    f"Dataset: {len(df):,} hospitals"
)

st.sidebar.caption(
    f"Model inputs: {len(MODEL_FEATURES)}"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="dashboard-title">'
        "Hospital Quality Intelligence"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Interactive dashboard for hospital quality prediction, "
        "model analysis, explainability, and responsible AI."
        "</div>",
        unsafe_allow_html=True,
    )

    labeled_hospitals = (
        df["rating_category"].notna().sum()
    )

    state_count = df["state"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Hospitals", f"{len(df):,}")

    with col2:
        st.metric("Labeled Hospitals", f"{labeled_hospitals:,}")

    with col3:
        st.metric("Model Features", len(MODEL_FEATURES))

    with col4:
        st.metric("States", state_count)

    st.divider()

    st.subheader("About the Project")

    st.write(
        """
        This project develops a machine-learning system for analyzing
        hospital quality using Medicare/CMS healthcare performance
        indicators.

        The project combines hospital data collection, CMS data
        integration, data profiling, feature engineering, statistical
        analysis, machine learning, explainable AI, fairness analysis,
        API deployment, containerization, and CI/CD automation.
        """
    )

    st.subheader("Project Pipeline")

    pipeline = [
        ("01", "Data Collection", "Medicare/CMS"),
        ("02", "Data Preparation", "Cleaning & Features"),
        ("03", "EDA", "Statistical Analysis"),
        ("04", "ML Modeling", "Classification"),
        ("05", "XAI & Fairness", "SHAP, LIME, Fairlearn"),
        ("06", "API", "FastAPI"),
        ("07", "CI/CD", "GitHub Actions"),
        ("08", "Dashboard", "Streamlit"),
    ]

    cols = st.columns(4)

    for index, (number, title, description) in enumerate(pipeline):
        with cols[index % 4]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <strong>{number} · {title}</strong>
                    <br>
                    <span style="color:#9ca3af;">
                        {description}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("Final Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "Logistic Regression")

    with col2:
        st.metric("Task", "3-Class Classification")

    with col3:
        st.metric("Classes", "High / Medium / Low")


# ============================================================
# PREDICTIONS
# ============================================================

elif page == "Predictions":

    st.title("🤖 Hospital Quality Prediction")

    st.write(
        """
        Use an existing hospital from the project dataset or enter
        hospital quality indicators manually. The dashboard uses the
        same trained model deployed by the FastAPI application.
        """
    )

    prediction_mode = st.radio(
        "Prediction mode",
        [
            "Existing Hospital",
            "Manual Entry",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # Existing hospital
    # --------------------------------------------------------

    if prediction_mode == "Existing Hospital":

        st.subheader("Select a Hospital")

        hospital_options = (
            df["hospital_name"]
            .fillna("Unnamed Hospital")
            .astype(str)
            .tolist()
        )

        selected_hospital = st.selectbox(
            "Hospital",
            hospital_options,
        )

        selected_row = df[
            df["hospital_name"].fillna("Unnamed Hospital").astype(str)
            == selected_hospital
        ].iloc[0]

        st.write(
            f"**Location:** "
            f"{selected_row.get('city', 'N/A')}, "
            f"{selected_row.get('state', 'N/A')}"
        )

        if "overall_rating" in selected_row:
            st.write(
                f"**Recorded overall rating:** "
                f"{selected_row['overall_rating']}"
            )

        if st.button(
            "Predict Hospital Quality",
            type="primary",
        ):

            input_data = pd.DataFrame(
                [selected_row.to_dict()]
            )

            input_data = calculate_engineered_features(
                input_data
            )

            model_input = prepare_model_input(
                input_data
            )

            try:
                prediction = model.predict(model_input)[0]
                probabilities = model.predict_proba(
                    model_input
                )[0]

                classes = model.classes_

                probability_df = pd.DataFrame(
                    {
                        "Category": classes,
                        "Probability": probabilities,
                    }
                )

                st.divider()

                st.subheader("Prediction Result")

                result_col, chart_col = st.columns(
                    [1, 2]
                )

                with result_col:
                    st.success(
                        f"Predicted Quality: **{prediction}**"
                    )

                    st.caption(
                        "This is a machine-learning prediction "
                        "and should not be interpreted as a clinical "
                        "or administrative decision."
                    )

                with chart_col:
                    st.bar_chart(
                        probability_df.set_index("Category")
                    )

                st.dataframe(
                    probability_df.assign(
                        Probability=lambda x:
                        (x["Probability"] * 100).round(2).astype(str)
                        + "%"
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:
                st.error(
                    f"Prediction failed: {exc}"
                )

    # --------------------------------------------------------
    # Manual entry
    # --------------------------------------------------------

    else:

        st.subheader("Manual Hospital Information")

        st.info(
            "Enter the available hospital indicators below. "
            "The engineered quality features are calculated "
            "automatically."
        )

        with st.form("manual_prediction_form"):

            st.markdown("### Hospital Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                city = st.text_input(
                    "City",
                    value="Austin",
                )

            with col2:
                state_options = sorted(
                    df["state"].dropna().astype(str).unique()
                )

                state = st.selectbox(
                    "State",
                    state_options,
                )

            with col3:
                hospital_type_options = sorted(
                    df["hospital_type"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                hospital_type = st.selectbox(
                    "Hospital Type",
                    hospital_type_options,
                )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                emergency_services = st.selectbox(
                    "Emergency Services",
                    sorted(
                        df["emergency_services"]
                        .dropna()
                        .astype(str)
                        .unique()
                    ),
                )

            with col2:
                hospital_ownership = st.selectbox(
                    "Hospital Ownership",
                    sorted(
                        df["hospital_ownership"]
                        .dropna()
                        .astype(str)
                        .unique()
                    ),
                )

            with col3:
                county = st.text_input(
                    "County",
                    value="Travis",
                )

            with col4:
                affiliated_clinicians = st.number_input(
                    "Affiliated Clinicians",
                    min_value=0.0,
                    value=370.0,
                )

            st.markdown("### Patient Experience")

            patient_survey_rating = st.number_input(
                "Patient Survey Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=1.0,
            )

            st.markdown("### Mortality Measures")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                mort_group_measure_count = st.number_input(
                    "Mortality Group Count",
                    min_value=0.0,
                    value=0.0,
                )

            with col2:
                facility_mort_measures = st.number_input(
                    "Facility Mortality Measures",
                    min_value=0.0,
                    value=0.0,
                )

            with col3:
                mort_better = st.number_input(
                    "Mortality Better",
                    min_value=0.0,
                    value=0.0,
                )

            with col4:
                mort_no_different = st.number_input(
                    "Mortality No Different",
                    min_value=0.0,
                    value=0.0,
                )

            with col5:
                mort_worse = st.number_input(
                    "Mortality Worse",
                    min_value=0.0,
                    value=0.0,
                )

            st.markdown("### Safety Measures")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                safety_group_measure_count = st.number_input(
                    "Safety Group Count",
                    min_value=0.0,
                    value=0.0,
                )

            with col2:
                facility_safety_measures = st.number_input(
                    "Facility Safety Measures",
                    min_value=0.0,
                    value=0.0,
                )

            with col3:
                safety_better = st.number_input(
                    "Safety Better",
                    min_value=0.0,
                    value=0.0,
                )

            with col4:
                safety_no_different = st.number_input(
                    "Safety No Different",
                    min_value=0.0,
                    value=0.0,
                )

            with col5:
                safety_worse = st.number_input(
                    "Safety Worse",
                    min_value=0.0,
                    value=0.0,
                )

            st.markdown("### Readmission Measures")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                readm_group_measure_count = st.number_input(
                    "Readmission Group Count",
                    min_value=0.0,
                    value=0.0,
                )

            with col2:
                facility_readm_measures = st.number_input(
                    "Facility Readmission Measures",
                    min_value=0.0,
                    value=0.0,
                )

            with col3:
                readm_better = st.number_input(
                    "Readmission Better",
                    min_value=0.0,
                    value=0.0,
                )

            with col4:
                readm_no_different = st.number_input(
                    "Readmission No Different",
                    min_value=0.0,
                    value=0.0,
                )

            with col5:
                readm_worse = st.number_input(
                    "Readmission Worse",
                    min_value=0.0,
                    value=0.0,
                )

            st.markdown("### Patient Experience Measures")

            col1, col2 = st.columns(2)

            with col1:
                patient_exp_group_measure_count = st.number_input(
                    "Patient Experience Group Count",
                    min_value=0.0,
                    value=0.0,
                )

            with col2:
                facility_patient_exp_measures = st.number_input(
                    "Facility Patient Experience Measures",
                    min_value=0.0,
                    value=0.0,
                )

            st.markdown("### Timely & Effective Care Measures")

            col1, col2 = st.columns(2)

            with col1:
                te_group_measure_count = st.number_input(
                    "Timely & Effective Group Count",
                    min_value=0.0,
                    value=0.0,
                )

            with col2:
                facility_te_measures = st.number_input(
                    "Facility Timely & Effective Measures",
                    min_value=0.0,
                    value=0.0,
                )

            submitted = st.form_submit_button(
                "Predict Hospital Quality",
                type="primary",
            )

        if submitted:

            manual_data = pd.DataFrame(
                [
                    {
                        "patient_survey_rating":
                            patient_survey_rating,
                        "affiliated_clinicians":
                            affiliated_clinicians,

                        "mort_group_measure_count":
                            mort_group_measure_count,
                        "facility_mort_measures":
                            facility_mort_measures,
                        "mort_better":
                            mort_better,
                        "mort_no_different":
                            mort_no_different,
                        "mort_worse":
                            mort_worse,

                        "safety_group_measure_count":
                            safety_group_measure_count,
                        "facility_safety_measures":
                            facility_safety_measures,
                        "safety_better":
                            safety_better,
                        "safety_no_different":
                            safety_no_different,
                        "safety_worse":
                            safety_worse,

                        "readm_group_measure_count":
                            readm_group_measure_count,
                        "facility_readm_measures":
                            facility_readm_measures,
                        "readm_better":
                            readm_better,
                        "readm_no_different":
                            readm_no_different,
                        "readm_worse":
                            readm_worse,

                        "patient_exp_group_measure_count":
                            patient_exp_group_measure_count,
                        "facility_patient_exp_measures":
                            facility_patient_exp_measures,

                        "te_group_measure_count":
                            te_group_measure_count,
                        "facility_te_measures":
                            facility_te_measures,

                        "city": city,
                        "state": state,
                        "hospital_type": hospital_type,
                        "emergency_services":
                            emergency_services,
                        "hospital_ownership":
                            hospital_ownership,
                        "county": county,
                    }
                ]
            )

            manual_data = calculate_engineered_features(
                manual_data
            )

            model_input = prepare_model_input(
                manual_data
            )

            try:

                prediction = model.predict(
                    model_input
                )[0]

                probabilities = model.predict_proba(
                    model_input
                )[0]

                classes = model.classes_

                probability_df = pd.DataFrame(
                    {
                        "Category": classes,
                        "Probability": probabilities,
                    }
                )

                st.divider()

                st.subheader("Prediction Result")

                result_col, chart_col = st.columns(
                    [1, 2]
                )

                with result_col:
                    st.success(
                        f"Predicted Quality: **{prediction}**"
                    )

                    st.caption(
                        "This is a machine-learning prediction "
                        "and should not be interpreted as a clinical "
                        "or administrative decision."
                    )

                with chart_col:
                    st.bar_chart(
                        probability_df.set_index("Category")
                    )

                st.dataframe(
                    probability_df.assign(
                        Probability=lambda x:
                        (x["Probability"] * 100).round(2).astype(str)
                        + "%"
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander(
                    "View engineered model features"
                ):
                    st.dataframe(
                        model_input,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:
                st.error(
                    f"Prediction failed: {exc}"
                )



elif page == "Quality Insights":

    st.title("📊 Hospital Quality Insights")

    st.write(
        """
        This section presents descriptive and statistical analysis
        performed on the hospital quality dataset. The visualizations
        are generated directly from the feature-engineered dataset
        used throughout the project.
        """
    )

    # --------------------------------------------------------
    # Analysis dataset
    # --------------------------------------------------------

    rated_df = df[
        df["overall_rating"].notna()
    ].copy()

    rated_df["overall_rating"] = pd.to_numeric(
        rated_df["overall_rating"],
        errors="coerce",
    )

    rated_df["patient_survey_rating"] = pd.to_numeric(
        rated_df["patient_survey_rating"],
        errors="coerce",
    )

    rated_df["affiliated_clinicians"] = pd.to_numeric(
        rated_df["affiliated_clinicians"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Overview metrics
    # --------------------------------------------------------

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Hospitals",
            f"{len(df):,}",
        )

    with col2:
        st.metric(
            "Rated Hospitals",
            f"{len(rated_df):,}",
        )

    with col3:
        st.metric(
            "Average Overall Rating",
            f"{rated_df['overall_rating'].mean():.2f}",
        )

    with col4:
        st.metric(
            "Average Patient Survey",
            f"{rated_df['patient_survey_rating'].mean():.2f}",
        )

    st.divider()

    # ========================================================
    # Rating distribution
    # ========================================================

    st.subheader("Overall Rating Distribution")

    rating_counts = (
        rated_df["rating_category"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0,
        )
    )

    rating_distribution = pd.DataFrame(
        {
            "Category": rating_counts.index,
            "Hospitals": rating_counts.values,
        }
    )

    rating_distribution["Percentage"] = (
        rating_distribution["Hospitals"]
        / rating_distribution["Hospitals"].sum()
        * 100
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.bar_chart(
            rating_distribution.set_index("Category")[
                ["Hospitals"]
            ]
        )

    with col2:
        st.dataframe(
            rating_distribution.assign(
                Percentage=lambda x:
                x["Percentage"].round(2).astype(str) + "%"
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "The analysis includes the 596 hospitals with an available "
        "overall rating."
    )

    st.divider()

    # ========================================================
    # Key numeric statistics
    # ========================================================

    st.subheader("Descriptive Statistics")

    descriptive_columns = [
        "overall_rating",
        "patient_survey_rating",
        "affiliated_clinicians",
        "total_better_measures",
        "total_worse_measures",
        "quality_balance_score",
        "better_measure_ratio",
        "worse_measure_ratio",
        "quality_coverage",
    ]

    available_descriptive = [
        column
        for column in descriptive_columns
        if column in df.columns
    ]

    descriptive_table = df[
        available_descriptive
    ].describe().T

    descriptive_table = descriptive_table[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ].round(3)

    st.dataframe(
        descriptive_table,
        use_container_width=True,
    )

    st.divider()

    # ========================================================
    # Correlation analysis
    # ========================================================

    st.subheader("Correlation with Overall Rating")

    correlation_columns = [
        "patient_survey_rating",
        "affiliated_clinicians",
        "total_better_measures",
        "total_worse_measures",
        "quality_balance_score",
        "better_measure_ratio",
        "worse_measure_ratio",
        "quality_coverage",
        "mort_better",
        "mort_worse",
        "safety_better",
        "safety_worse",
        "readm_better",
        "readm_worse",
    ]

    available_correlations = [
        column
        for column in correlation_columns
        if column in rated_df.columns
    ]

    correlation_results = []

    for column in available_correlations:

        pair = rated_df[
            ["overall_rating", column]
        ].dropna()

        if len(pair) >= 3:
            correlation = pair[
                "overall_rating"
            ].corr(pair[column])

            correlation_results.append(
                {
                    "Feature": column,
                    "Pearson Correlation": correlation,
                    "Absolute Correlation": abs(
                        correlation
                    ),
                    "Observations": len(pair),
                }
            )

    correlation_df = pd.DataFrame(
        correlation_results
    ).sort_values(
        "Absolute Correlation",
        ascending=False,
    )

    correlation_plot = correlation_df[
        ["Feature", "Pearson Correlation"]
    ].set_index("Feature")

    st.bar_chart(correlation_plot)

    st.dataframe(
        correlation_df[
            [
                "Feature",
                "Pearson Correlation",
                "Observations",
            ]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        """
        Correlation describes the strength and direction of a
        linear relationship between two variables. It does not
        establish causation.
        """
    )

    st.divider()

    # ========================================================
    # Hospital size vs rating
    # ========================================================

    st.subheader(
        "Hospital Size Category vs Rating Category"
    )

    size_rating = pd.crosstab(
        df["hospital_size_category"],
        df["rating_category"],
    )

    size_rating = size_rating.reindex(
        index=["Small", "Medium", "Large"],
        columns=["Low", "Medium", "High"],
        fill_value=0,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            size_rating,
            use_container_width=True,
        )

    with col2:

        st.bar_chart(
            size_rating,
        )

    chi_data = df[
        [
            "hospital_size_category",
            "rating_category",
        ]
    ].dropna()

    if len(chi_data) > 0:

        contingency = pd.crosstab(
            chi_data["hospital_size_category"],
            chi_data["rating_category"],
        )

        chi2, chi_p, chi_df, _ = (
            stats.chi2_contingency(
                contingency
            )
        )

        st.markdown("#### Chi-square Test")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Chi-square",
                f"{chi2:.3f}",
            )

        with col2:
            st.metric(
                "p-value",
                f"{chi_p:.3e}",
            )

        with col3:
            st.metric(
                "Degrees of Freedom",
                chi_df,
            )

        if chi_p < 0.05:
            st.success(
                "The test indicates a statistically significant "
                "association between hospital size category and "
                "rating category at α = 0.05."
            )
        else:
            st.info(
                "The test does not indicate a statistically "
                "significant association at α = 0.05."
            )

    st.caption(
        "A statistically significant association does not imply "
        "that hospital size causes the observed rating differences."
    )

    st.divider()

    # ========================================================
    # Patient survey rating across rating categories
    # ========================================================

    st.subheader(
        "Patient Survey Rating Across Quality Categories"
    )

    anova_data = rated_df[
        [
            "rating_category",
            "patient_survey_rating",
        ]
    ].dropna()

    anova_groups = [
        group["patient_survey_rating"].values
        for _, group in anova_data.groupby(
            "rating_category"
        )
        if len(group) > 1
    ]

    group_names = [
        name
        for name, group in anova_data.groupby(
            "rating_category"
        )
        if len(group) > 1
    ]

    if len(anova_groups) >= 2:

        f_stat, anova_p = stats.f_oneway(
            *anova_groups
        )

        category_means = (
            anova_data
            .groupby("rating_category")[
                "patient_survey_rating"
            ]
            .agg(["count", "mean"])
            .reindex(
                ["Low", "Medium", "High"]
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.dataframe(
                category_means.round(3),
                use_container_width=True,
            )

        with col2:

            st.bar_chart(
                category_means[
                    ["mean"]
                ].rename(
                    columns={
                        "mean":
                        "Mean Patient Survey Rating"
                    }
                )
            )

        st.markdown("#### One-way ANOVA")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "F-statistic",
                f"{f_stat:.3f}",
            )

        with col2:
            st.metric(
                "p-value",
                f"{anova_p:.3e}",
            )

        if anova_p < 0.05:
            st.success(
                "The ANOVA indicates that mean patient survey "
                "ratings differ significantly across the "
                "quality-rating categories at α = 0.05."
            )
        else:
            st.info(
                "The ANOVA does not indicate a statistically "
                "significant difference at α = 0.05."
            )

    st.divider()

    # ========================================================
    # Emergency services test
    # ========================================================

    st.subheader(
        "Emergency Services vs Overall Rating"
    )

    emergency_data = rated_df[
        [
            "emergency_services",
            "overall_rating",
        ]
    ].dropna()

    emergency_groups = []

    emergency_labels = sorted(
        emergency_data[
            "emergency_services"
        ].astype(str).unique()
    )

    for label in emergency_labels:

        values = emergency_data.loc[
            emergency_data[
                "emergency_services"
            ].astype(str)
            == label,
            "overall_rating",
        ]

        if len(values) > 1:
            emergency_groups.append(
                values.values
            )

    if len(emergency_groups) == 2:

        t_stat, t_p = stats.ttest_ind(
            emergency_groups[0],
            emergency_groups[1],
            equal_var=False,
        )

        emergency_summary = (
            emergency_data
            .groupby("emergency_services")[
                "overall_rating"
            ]
            .agg(["count", "mean"])
        )

        st.dataframe(
            emergency_summary.round(3),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Welch t-statistic",
                f"{t_stat:.4f}",
            )

        with col2:
            st.metric(
                "p-value",
                f"{t_p:.4f}",
            )

        if t_p < 0.05:
            st.success(
                "The Welch t-test indicates a statistically "
                "significant difference in mean overall rating "
                "between the two emergency-service groups."
            )
        else:
            st.info(
                "The Welch t-test does not indicate a statistically "
                "significant difference in mean overall rating "
                "between the two emergency-service groups."
            )

    st.divider()

    # ========================================================
    # Interpretation
    # ========================================================

    st.subheader("Interpretation Notes")

    st.markdown(
        """
        - The rating distribution describes the labeled subset of
          hospitals and does not include hospitals without an
          overall rating.
        - Correlations describe pairwise linear associations and
          should not be interpreted as causal effects.
        - Statistical hypothesis tests use a significance level of
          α = 0.05.
        - Statistical significance does not by itself establish
          practical importance or causation.
        - The analysis reflects the available Medicare/CMS-derived
          variables and their associated missingness.
        """
    )


elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        """
        This section presents the performance of the machine-learning
        models developed in Experiment 4. The results are loaded from
        the saved experiment reports, so the dashboard uses the same
        evaluation results used during model selection.
        """
    )

    # --------------------------------------------------------
    # Report paths
    # --------------------------------------------------------

    EXP4_DIR = BASE_DIR / "reports" / "experiment_4"

    BASELINE_PATH = (
        EXP4_DIR / "baseline_model_results.csv"
    )

    TUNED_PATH = (
        EXP4_DIR / "tuned_model_results.csv"
    )

    COMPARISON_PATH = (
        EXP4_DIR / "baseline_vs_tuned_comparison.csv"
    )

    FINAL_PATH = (
        EXP4_DIR / "final_model_selection.csv"
    )

    CONFUSION_PATH = (
        EXP4_DIR
        / "confusion_matrices"
        / "logistic_regression_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Load experiment reports
    # --------------------------------------------------------

    try:

        baseline_results = pd.read_csv(
            BASELINE_PATH
        )

        tuned_results = pd.read_csv(
            TUNED_PATH
        )

        comparison_results = pd.read_csv(
            COMPARISON_PATH
        )

        final_results = pd.read_csv(
            FINAL_PATH
        )

        confusion_matrix = pd.read_csv(
            CONFUSION_PATH,
            index_col=0,
        )

    except Exception as exc:

        st.error(
            f"Unable to load Experiment 4 reports: {exc}"
        )

        st.stop()

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    st.subheader("Final Selected Model")

    final_row = final_results.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Logistic Regression",
        )

    with col2:
        st.metric(
            "Accuracy",
            f"{float(final_row['accuracy']) * 100:.2f}%",
        )

    with col3:
        st.metric(
            "Macro F1",
            f"{float(final_row['macro_f1']) * 100:.2f}%",
        )

    with col4:
        st.metric(
            "Weighted F1",
            f"{float(final_row['weighted_f1']) * 100:.2f}%",
        )

    st.caption(
        "Final model selected during Experiment 4 model selection."
    )

    st.divider()

    # --------------------------------------------------------
    # Final model detailed metrics
    # --------------------------------------------------------

    st.subheader("Final Model Metrics")

    final_metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Weighted F1",
            ],
            "Score": [
                float(final_row["accuracy"]),
                float(final_row["macro_precision"]),
                float(final_row["macro_recall"]),
                float(final_row["macro_f1"]),
                float(final_row["weighted_f1"]),
            ],
        }
    )

    final_metrics["Score"] = (
        final_metrics["Score"] * 100
    ).round(2)

    st.dataframe(
        final_metrics.rename(
            columns={"Score": "Score (%)"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Baseline model comparison
    # --------------------------------------------------------

    st.subheader("Baseline Model Comparison")

    display_baseline = baseline_results.copy()

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    for column in metric_columns:

        if column in display_baseline.columns:

            display_baseline[column] = (
                display_baseline[column] * 100
            ).round(2)

    st.dataframe(
        display_baseline,
        use_container_width=True,
        hide_index=True,
    )

    # Accuracy chart

    if "model" in baseline_results.columns:

        accuracy_chart = baseline_results[
            ["model", "accuracy"]
        ].copy()

        accuracy_chart["accuracy"] *= 100

        accuracy_chart = accuracy_chart.set_index(
            "model"
        )

        st.markdown("#### Accuracy Comparison")

        st.bar_chart(
            accuracy_chart
        )

    st.divider()

    # --------------------------------------------------------
    # Tuned model comparison
    # --------------------------------------------------------

    st.subheader("Baseline vs Tuned Models")

    display_comparison = comparison_results.copy()

    for column in metric_columns:

        if column in display_comparison.columns:

            display_comparison[column] = (
                display_comparison[column] * 100
            ).round(2)

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.subheader(
        "Final Model Confusion Matrix"
    )

    st.write(
        """
        The confusion matrix shows how the final Logistic Regression
        model classified the 120 test hospitals across the three
        quality categories.
        """
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True,
    )

    st.caption(
        "Rows represent actual classes and columns represent "
        "predicted classes."
    )

    st.divider()

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    st.subheader("Experiment Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Labeled Hospitals",
            "596",
        )

    with col2:
        st.metric(
            "Training Samples",
            "476",
        )

    with col3:
        st.metric(
            "Test Samples",
            "120",
        )

    with col4:
        st.metric(
            "Model Features",
            "34",
        )

    st.markdown(
        """
        **Classification task:** 3-class hospital quality prediction

        **Classes:** Low, Medium, High

        **Evaluation metrics:** Accuracy, Macro Precision,
        Macro Recall, Macro F1, and Weighted F1.

        **Experiment tracking:** MLflow was used to record baseline
        and tuned model experiments.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader("Performance Interpretation")

    st.markdown(
        """
        - The final selected model is a Logistic Regression pipeline.
        - The model achieved 68.33% accuracy on the held-out test set.
        - Macro F1 was 64.75%, providing a class-balanced view of
          performance.
        - Weighted F1 was 67.10%, accounting for the number of
          samples in each class.
        - Five baseline models were evaluated before selecting the
          final model.
        - Logistic Regression and Gradient Boosting were subsequently
          tuned using cross-validated hyperparameter search.
        - The final model was selected using the Experiment 4
          evaluation results and saved as the project model artifact.
        """
    )

elif page == "SHAP Explainability":

    st.title("🔍 SHAP Explainability")

    st.write(
        """
        This section explains how the trained Logistic Regression model
        makes its predictions using SHAP (SHapley Additive exPlanations).
        The results are loaded from the SHAP analysis performed in
        Experiment 5.
        """
    )

    # --------------------------------------------------------
    # Experiment 5 paths
    # --------------------------------------------------------

    EXP5_DIR = BASE_DIR / "reports" / "experiment_5"

    SHAP_IMPORTANCE_PATH = (
        EXP5_DIR / "shap_feature_importance.csv"
    )

    SHAP_CLASS_IMPORTANCE_PATH = (
        EXP5_DIR / "shap_class_feature_importance.csv"
    )

    SHAP_SUMMARY_PATH = (
        EXP5_DIR / "shap_plots" / "shap_summary.png"
    )

    SHAP_BAR_PATH = (
        EXP5_DIR / "shap_plots" / "shap_feature_importance_bar.png"
    )

    # --------------------------------------------------------
    # Load SHAP reports
    # --------------------------------------------------------

    try:

        shap_importance = pd.read_csv(
            SHAP_IMPORTANCE_PATH
        )

        shap_class_importance = pd.read_csv(
            SHAP_CLASS_IMPORTANCE_PATH
        )

    except Exception as exc:

        st.error(
            f"Unable to load SHAP reports: {exc}"
        )

        st.stop()

    # --------------------------------------------------------
    # SHAP overview
    # --------------------------------------------------------

    st.subheader("What is SHAP?")

    st.info(
        """
        SHAP explains individual model predictions by measuring how
        each input feature contributes to the prediction.

        In this project, SHAP was applied to the final Logistic Regression
        model using the transformed test dataset. The analysis provides
        both global feature importance and class-specific explanations.
        """
    )

    # --------------------------------------------------------
    # SHAP experiment information
    # --------------------------------------------------------

    st.subheader("SHAP Analysis Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Logistic Regression",
        )

    with col2:
        st.metric(
            "Test Samples",
            "120",
        )

    with col3:
        st.metric(
            "Original Features",
            "34",
        )

    with col4:
        st.metric(
            "Transformed Features",
            "477",
        )

    st.caption(
        "SHAP was calculated on the held-out test set after the "
        "model preprocessing pipeline transformed the input features."
    )

    st.divider()

    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    st.subheader("Global Feature Importance")

    st.write(
        """
        The table below ranks features according to their mean absolute
        SHAP value. A larger value indicates that the feature had a
        greater average contribution to the model's predictions across
        the test set.
        """
    )

    display_importance = shap_importance.copy()

    # Detect the feature and importance columns
    feature_column = next(
        (
            column
            for column in display_importance.columns
            if str(column).lower()
            in [
                "feature",
                "feature_name",
                "features",
            ]
        ),
        None,
    )

    importance_column = next(
        (
            column
            for column in display_importance.columns
            if "mean" in str(column).lower()
            and "shap" in str(column).lower()
        ),
        None,
    )

    if feature_column is None:
        feature_column = display_importance.columns[0]

    if importance_column is None:

        numeric_candidates = (
            display_importance
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        if numeric_candidates:
            importance_column = numeric_candidates[-1]

    if importance_column is not None:

        display_importance = (
            display_importance
            .sort_values(
                importance_column,
                ascending=False,
            )
            .reset_index(drop=True)
        )

        top_features = display_importance.head(10).copy()

        top_features[importance_column] = (
            top_features[importance_column]
            .round(4)
        )

        st.dataframe(
            top_features,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Top 10 Feature Contributions")

        chart_data = (
            top_features[
                [
                    feature_column,
                    importance_column,
                ]
            ]
            .set_index(feature_column)
            .sort_values(
                importance_column,
                ascending=True,
            )
        )

        st.bar_chart(
            chart_data
        )

    else:

        st.dataframe(
            display_importance,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --------------------------------------------------------
    # SHAP summary plot
    # --------------------------------------------------------

    st.subheader("SHAP Summary Plot")

    if SHAP_SUMMARY_PATH.exists():

        st.image(
            str(SHAP_SUMMARY_PATH),
            use_container_width=True,
        )

        st.caption(
            "SHAP summary plot generated during Experiment 5."
        )

    else:

        st.warning(
            "SHAP summary plot was not found."
        )

    st.divider()

    # --------------------------------------------------------
    # SHAP feature importance plot
    # --------------------------------------------------------

    st.subheader("SHAP Feature Importance")

    if SHAP_BAR_PATH.exists():

        st.image(
            str(SHAP_BAR_PATH),
            use_container_width=True,
        )

        st.caption(
            "Mean absolute SHAP feature importance generated "
            "during Experiment 5."
        )

    else:

        st.warning(
            "SHAP feature importance plot was not found."
        )

    st.divider()

    # --------------------------------------------------------
    # Class-wise SHAP importance
    # --------------------------------------------------------

    st.subheader("Class-wise SHAP Importance")

    st.write(
        """
        SHAP importance can differ between the three prediction
        classes. This section allows the feature contributions for
        each class to be inspected separately.
        """
    )

    class_column = next(
        (
            column
            for column in shap_class_importance.columns
            if str(column).lower()
            in [
                "class",
                "category",
                "target",
                "prediction_class",
            ]
        ),
        None,
    )

    if class_column is not None:

        available_classes = sorted(
            shap_class_importance[
                class_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_class = st.selectbox(
            "Select prediction class",
            available_classes,
        )

        class_data = shap_class_importance[
            shap_class_importance[
                class_column
            ].astype(str)
            == selected_class
        ].copy()

        class_feature_column = next(
            (
                column
                for column in class_data.columns
                if str(column).lower()
                in [
                    "feature",
                    "feature_name",
                    "features",
                ]
            ),
            None,
        )

        class_importance_column = next(
            (
                column
                for column in class_data.columns
                if "mean" in str(column).lower()
                and "shap" in str(column).lower()
            ),
            None,
        )

        if class_feature_column is None:

            non_class_columns = [
                column
                for column in class_data.columns
                if column != class_column
            ]

            if non_class_columns:
                class_feature_column = non_class_columns[0]

        if class_importance_column is None:

            numeric_candidates = (
                class_data
                .select_dtypes(include="number")
                .columns
                .tolist()
            )

            if numeric_candidates:
                class_importance_column = (
                    numeric_candidates[-1]
                )

        if (
            class_feature_column is not None
            and class_importance_column is not None
        ):

            class_data = (
                class_data
                .sort_values(
                    class_importance_column,
                    ascending=False,
                )
                .head(10)
            )

            st.dataframe(
                class_data,
                use_container_width=True,
                hide_index=True,
            )

            class_chart = (
                class_data[
                    [
                        class_feature_column,
                        class_importance_column,
                    ]
                ]
                .set_index(
                    class_feature_column
                )
                .sort_values(
                    class_importance_column,
                    ascending=True,
                )
            )

            st.bar_chart(
                class_chart
            )

        else:

            st.dataframe(
                class_data,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.dataframe(
            shap_class_importance,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --------------------------------------------------------
    # Key findings from Experiment 5
    # --------------------------------------------------------

    st.subheader("Key SHAP Findings")

    st.markdown(
        """
        Based on the Experiment 5 SHAP analysis:

        - **Patient survey rating** had the largest average absolute
          SHAP contribution among the examined features.
        - **Mortality better measures** also made a substantial
          contribution to model predictions.
        - **Total worse measures** had a notable contribution to the
          model's predictions.
        - **Mortality no-different measures** and
          **quality coverage** were also among the higher-ranked
          features.
        - Engineered quality indicators such as
          **quality balance score** and **worse measure ratio**
          contributed to the model predictions.
        """
    )

    st.info(
        """
        SHAP describes how features contribute to the model's
        predictions. These contributions should not be interpreted
        as causal effects. A high SHAP importance means that the
        feature was influential for the trained model, not that
        changing the feature would necessarily cause hospital quality
        to change.
        """
    )


elif page == "Drift Checks":

    st.title("📉 Data Drift Checks")

    st.write(
        """
        This section provides a dataset-level drift monitoring baseline
        for the hospital quality data. It compares distributions within
        the available dataset to identify variables that may require
        monitoring.
        """
    )

    st.info(
        """
        No separate production or time-separated reference dataset is
        currently available in this project. Therefore, these results
        should be interpreted as a distribution-shift baseline rather
        than evidence of real-world production drift.
        """
    )

    # --------------------------------------------------------
    # Drift configuration
    # --------------------------------------------------------

    st.subheader("Drift Analysis Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Size",
            f"{len(df):,}",
        )

    with col2:
        st.metric(
            "Numeric Features",
            len(
                [
                    column
                    for column in NUMERIC_FEATURES
                    if column in df.columns
                ]
            ),
        )

    with col3:
        st.metric(
            "Categorical Features",
            len(
                [
                    column
                    for column in CATEGORICAL_FEATURES
                    if column in df.columns
                ]
            ),
        )

    with col4:
        st.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}",
        )

    st.divider()

    # --------------------------------------------------------
    # Missing-value monitoring
    # --------------------------------------------------------

    st.subheader("Missing-Value Monitoring")

    missing_summary = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    missing_summary.columns = [
        "Feature",
        "Missing Percentage",
    ]

    missing_summary["Missing Percentage"] = (
        missing_summary["Missing Percentage"]
        .round(2)
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        missing_plot = (
            missing_summary
            .head(15)
            .set_index("Feature")
        )

        st.bar_chart(
            missing_plot[
                ["Missing Percentage"]
            ]
        )

    with col2:

        st.dataframe(
            missing_summary.head(15),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Missing-value percentage is calculated across the complete "
        "feature-engineered dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # Numeric distribution comparison
    # --------------------------------------------------------

    st.subheader("Numeric Distribution Shift")

    st.write(
        """
        The dataset is divided into two comparable portions and a
        two-sample Kolmogorov-Smirnov test is applied to numeric
        features. The test measures whether the two samples show
        evidence of different distributions.
        """
    )

    numeric_columns = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns
    ]

    # Use only rows with valid observations for each feature.
    # Splitting the dataset first avoids comparing the same observations
    # against themselves.
    midpoint = len(df) // 2

    reference_df = df.iloc[:midpoint].copy()
    comparison_df = df.iloc[midpoint:].copy()

    numeric_drift_results = []

    for column in numeric_columns:

        reference_values = pd.to_numeric(
            reference_df[column],
            errors="coerce",
        ).dropna()

        comparison_values = pd.to_numeric(
            comparison_df[column],
            errors="coerce",
        ).dropna()

        if (
            len(reference_values) >= 3
            and len(comparison_values) >= 3
        ):

            ks_stat, p_value = stats.ks_2samp(
                reference_values,
                comparison_values,
            )

            numeric_drift_results.append(
                {
                    "Feature": column,
                    "KS Statistic": ks_stat,
                    "p-value": p_value,
                    "Reference Count": len(
                        reference_values
                    ),
                    "Comparison Count": len(
                        comparison_values
                    ),
                }
            )

    numeric_drift_df = pd.DataFrame(
        numeric_drift_results
    )

    if not numeric_drift_df.empty:

        numeric_drift_df = (
            numeric_drift_df
            .sort_values(
                "KS Statistic",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        numeric_display = numeric_drift_df.copy()

        numeric_display[
            "KS Statistic"
        ] = numeric_display[
            "KS Statistic"
        ].round(4)

        numeric_display[
            "p-value"
        ] = numeric_display[
            "p-value"
        ].round(6)

        st.dataframe(
            numeric_display,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### Largest Distribution Differences"
        )

        drift_chart = (
            numeric_drift_df
            .head(10)
            .set_index("Feature")[
                ["KS Statistic"]
            ]
            .sort_values(
                "KS Statistic",
                ascending=True,
            )
        )

        st.bar_chart(
            drift_chart
        )

        significant_count = (
            numeric_drift_df["p-value"] < 0.05
        ).sum()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Numeric Features Tested",
                len(numeric_drift_df),
            )

        with col2:
            st.metric(
                "p < 0.05",
                int(significant_count),
            )

        with col3:
            st.metric(
                "Largest KS Statistic",
                f"{numeric_drift_df['KS Statistic'].max():.3f}",
            )

    else:

        st.warning(
            "Not enough numeric data was available for the "
            "distribution comparison."
        )

    st.caption(
        "The KS test compares the two dataset portions. A small "
        "p-value indicates evidence that the distributions differ; "
        "it does not identify the cause of the difference."
    )

    st.divider()

    # --------------------------------------------------------
    # Categorical distribution monitoring
    # --------------------------------------------------------

    st.subheader("Categorical Distribution Monitoring")

    categorical_columns = [
        column
        for column in CATEGORICAL_FEATURES
        if column in df.columns
    ]

    categorical_results = []

    for column in categorical_columns:

        reference_counts = (
            reference_df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts(
                normalize=True
            )
        )

        comparison_counts = (
            comparison_df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts(
                normalize=True
            )
        )

        categories = set(
            reference_counts.index
        ).union(
            comparison_counts.index
        )

        max_difference = 0.0

        for category in categories:

            reference_share = (
                reference_counts.get(
                    category,
                    0.0,
                )
            )

            comparison_share = (
                comparison_counts.get(
                    category,
                    0.0,
                )
            )

            difference = abs(
                reference_share
                - comparison_share
            )

            max_difference = max(
                max_difference,
                difference,
            )

        categorical_results.append(
            {
                "Feature": column,
                "Maximum Category Difference (%)":
                    max_difference * 100,
                "Reference Unique Values":
                    len(reference_counts),
                "Comparison Unique Values":
                    len(comparison_counts),
            }
        )

    categorical_drift_df = pd.DataFrame(
        categorical_results
    )

    if not categorical_drift_df.empty:

        categorical_drift_df = (
            categorical_drift_df
            .sort_values(
                "Maximum Category Difference (%)",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        categorical_display = (
            categorical_drift_df.copy()
        )

        categorical_display[
            "Maximum Category Difference (%)"
        ] = (
            categorical_display[
                "Maximum Category Difference (%)"
            ].round(2)
        )

        st.dataframe(
            categorical_display,
            use_container_width=True,
            hide_index=True,
        )

        categorical_chart = (
            categorical_drift_df
            .head(10)
            .set_index("Feature")[
                [
                    "Maximum Category Difference (%)"
                ]
            ]
            .sort_values(
                "Maximum Category Difference (%)",
                ascending=True,
            )
        )

        st.bar_chart(
            categorical_chart
        )

    else:

        st.warning(
            "No categorical features were available."
        )

    st.divider()

    # --------------------------------------------------------
    # Feature monitoring selector
    # --------------------------------------------------------

    st.subheader("Feature Distribution Viewer")

    available_numeric = [
        column
        for column in numeric_columns
        if column in reference_df.columns
    ]

    if available_numeric:

        selected_feature = st.selectbox(
            "Select numeric feature",
            available_numeric,
        )

        reference_values = pd.to_numeric(
            reference_df[selected_feature],
            errors="coerce",
        ).dropna()

        comparison_values = pd.to_numeric(
            comparison_df[selected_feature],
            errors="coerce",
        ).dropna()

        distribution_data = pd.DataFrame(
            {
                "Reference": pd.Series(
                    reference_values
                    .reset_index(drop=True)
                ),
                "Comparison": pd.Series(
                    comparison_values
                    .reset_index(drop=True)
                ),
            }
        )

        st.line_chart(
            distribution_data
        )

        selected_result = numeric_drift_df[
            numeric_drift_df["Feature"]
            == selected_feature
        ]

        if not selected_result.empty:

            selected_ks = float(
                selected_result.iloc[0][
                    "KS Statistic"
                ]
            )

            selected_p = float(
                selected_result.iloc[0][
                    "p-value"
                ]
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "KS Statistic",
                    f"{selected_ks:.4f}",
                )

            with col2:
                st.metric(
                    "p-value",
                    f"{selected_p:.6f}",
                )

    st.divider()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader("Drift Interpretation")

    st.markdown(
        """
        - Drift analysis is intended to identify changes in the
          distribution of model inputs over time or between datasets.
        - A larger KS statistic indicates a larger difference between
          the two numeric distributions.
        - A small p-value provides statistical evidence that the two
          distributions differ.
        - For categorical variables, the maximum category difference
          indicates the largest change in category proportion.
        - Statistical distribution differences do not by themselves
          indicate model degradation, data-quality problems, or a
          specific cause.
        """
    )

    st.warning(
        """
        **Monitoring limitation:** The current project contains one
        feature-engineered hospital dataset rather than separate
        reference and production datasets. The comparison above
        therefore provides a baseline monitoring mechanism. For actual
        deployment monitoring, a future version should compare a fixed
        reference dataset against newly collected hospital data.
        """
    )


elif page == "Responsible AI":

    st.title("🛡️ Responsible AI")

    st.write(
        """
        This section presents the fairness analysis performed in
        Experiment 5 using Fairlearn. The audit examines whether
        prediction outcomes differ across geographic groups represented
        by hospital state.
        """
    )

    st.info(
        """
        **Important:** State is used here as a geographic grouping
        variable. It is not being treated as a protected demographic
        attribute. The current dataset does not contain individual-level
        race, gender, or age attributes, so this analysis cannot establish
        fairness with respect to those protected characteristics.
        """
    )

    # --------------------------------------------------------
    # Experiment 5 fairness reports
    # --------------------------------------------------------

    EXP5_FAIRNESS_DIR = (
        BASE_DIR
        / "reports"
        / "experiment_5"
        / "fairness"
    )

    FAIRNESS_GROUP_PATH = (
        BASE_DIR
        / "reports"
        / "experiment_5"
        / "fairness_groups.csv"
    )

    FAIRNESS_METRICS_PATH = (
        EXP5_FAIRNESS_DIR
        / "fairness_group_metrics.csv"
    )

    FAIRNESS_SUMMARY_PATH = (
        EXP5_FAIRNESS_DIR
        / "fairness_summary.csv"
    )

    # --------------------------------------------------------
    # Load fairness reports
    # --------------------------------------------------------

    if not FAIRNESS_METRICS_PATH.exists():

        st.error(
            "Fairlearn group metrics report was not found at: "
            f"{FAIRNESS_METRICS_PATH}"
        )

        st.stop()

    fairness_metrics = pd.read_csv(
        FAIRNESS_METRICS_PATH
    )

    fairness_summary = None

    if FAIRNESS_SUMMARY_PATH.exists():

        fairness_summary = pd.read_csv(
            FAIRNESS_SUMMARY_PATH
        )

    fairness_groups = None

    if FAIRNESS_GROUP_PATH.exists():

        fairness_groups = pd.read_csv(
            FAIRNESS_GROUP_PATH
        )

    # --------------------------------------------------------
    # Fairness audit configuration
    # --------------------------------------------------------

    st.subheader("Fairness Audit Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Logistic Regression",
        )

    with col2:
        st.metric(
            "Test Samples",
            "120",
        )

    with col3:
        st.metric(
            "Grouping Variable",
            "State",
        )

    with col4:
        st.metric(
            "Prediction Classes",
            "3",
        )

    st.caption(
        "Fairness metrics were calculated on the held-out test set "
        "during Experiment 5."
    )

    st.divider()

    # --------------------------------------------------------
    # Geographic group distribution
    # --------------------------------------------------------

    st.subheader("Geographic Group Distribution")

    if fairness_groups is not None:

        st.dataframe(
            fairness_groups,
            use_container_width=True,
            hide_index=True,
        )

        group_column = next(
            (
                column
                for column in fairness_groups.columns
                if str(column).lower()
                in [
                    "state",
                    "group",
                    "sensitive_feature",
                    "group_value",
                ]
            ),
            None,
        )

        count_column = next(
            (
                column
                for column in fairness_groups.columns
                if (
                    "count"
                    in str(column).lower()
                    or "samples"
                    in str(column).lower()
                )
            ),
            None,
        )

        if (
            group_column is not None
            and count_column is not None
        ):

            group_chart = (
                fairness_groups[
                    [
                        group_column,
                        count_column,
                    ]
                ]
                .set_index(group_column)
                .sort_values(
                    count_column,
                    ascending=True,
                )
            )

            st.bar_chart(
                group_chart
            )

        st.caption(
            "Smaller geographic groups can produce less stable "
            "fairness estimates."
        )

    else:

        st.warning(
            "The geographic group distribution report was not found."
        )

    st.divider()

    # --------------------------------------------------------
    # Fairness summary
    # --------------------------------------------------------

    st.subheader("Fairness Summary")

    if fairness_summary is not None:

        st.dataframe(
            fairness_summary,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Fairness metrics
    # --------------------------------------------------------

    st.subheader("Fairness Metrics")

    st.write(
        """
        The Fairlearn audit evaluates differences in model outcomes
        across state-based geographic groups.

        **Demographic Parity Difference** measures the largest
        difference in prediction/selection rates between groups.

        **Equalized Odds Difference** measures the largest difference
        in group-level error-related outcomes.
        """
    )

    st.dataframe(
        fairness_metrics,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Detect fairness metric columns
    # --------------------------------------------------------

    class_column = next(
        (
            column
            for column in fairness_metrics.columns
            if str(column).lower()
            in [
                "class",
                "category",
                "prediction_class",
                "target_class",
            ]
        ),
        None,
    )

    dp_column = next(
        (
            column
            for column in fairness_metrics.columns
            if (
                "demographic"
                in str(column).lower()
                and "difference"
                in str(column).lower()
            )
            or str(column).lower()
            in [
                "dp_difference",
                "demographic_parity_difference",
            ]
        ),
        None,
    )

    eo_column = next(
        (
            column
            for column in fairness_metrics.columns
            if (
                "equalized"
                in str(column).lower()
                and "difference"
                in str(column).lower()
            )
            or str(column).lower()
            in [
                "eo_difference",
                "equalized_odds_difference",
            ]
        ),
        None,
    )

    # --------------------------------------------------------
    # Fairness metric cards
    # --------------------------------------------------------

    if (
        class_column is not None
        and dp_column is not None
        and eo_column is not None
    ):

        st.subheader("Observed Fairness Differences")

        preferred_order = [
            "High",
            "Medium",
            "Low",
        ]

        available_classes = (
            fairness_metrics[
                class_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        classes = [
            category
            for category in preferred_order
            if category in available_classes
        ]

        classes += [
            category
            for category in available_classes
            if category not in classes
        ]

        cards = st.columns(
            len(classes)
        )

        for index, category in enumerate(classes):

            row = fairness_metrics[
                fairness_metrics[
                    class_column
                ].astype(str)
                == category
            ]

            if row.empty:
                continue

            dp_value = float(
                row.iloc[0][dp_column]
            )

            eo_value = float(
                row.iloc[0][eo_column]
            )

            with cards[index]:

                st.markdown(
                    f"### {category}"
                )

                st.metric(
                    "DP Difference",
                    f"{dp_value:.3f}",
                )

                st.metric(
                    "EO Difference",
                    f"{eo_value:.3f}",
                )

        st.divider()

        # ----------------------------------------------------
        # Fairness comparison chart
        # ----------------------------------------------------

        st.subheader(
            "Fairness Difference Comparison"
        )

        fairness_chart = fairness_metrics[
            [
                class_column,
                dp_column,
                eo_column,
            ]
        ].copy()

        fairness_chart = fairness_chart.set_index(
            class_column
        )

        fairness_chart.columns = [
            "Demographic Parity Difference",
            "Equalized Odds Difference",
        ]

        st.bar_chart(
            fairness_chart
        )

    # --------------------------------------------------------
    # Experiment 5 observed findings
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Experiment 5 Fairness Findings"
    )

    st.markdown(
        """
        The Experiment 5 audit produced the following observed
        disparities across state-based geographic groups:

        - **High:** DP Difference = 0.475;
          EO Difference = 0.500.
        - **Low:** DP Difference = 0.533;
          EO Difference = 1.000.
        - **Medium:** DP Difference = 0.475;
          EO Difference = 1.000.

        These values indicate measurable differences in prediction
        behavior across the geographic groups included in the test set.
        They do not, by themselves, establish why those differences
        occurred.
        """
    )

    st.warning(
        """
        Several state groups in the test set are small. In particular,
        Arizona, Connecticut, North Carolina, and Pennsylvania have
        relatively few test observations. Fairness estimates for small
        groups can therefore be unstable and should be interpreted
        cautiously.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Fairness scope
    # --------------------------------------------------------

    st.subheader(
        "Fairness Scope & Limitations"
    )

    st.markdown(
        """
        **Grouping variable:** State

        State was selected as a geographic grouping variable for the
        Experiment 5 audit. It should not be interpreted as a direct
        protected demographic attribute.

        The available project dataset does not contain individual-level
        protected attributes such as:

        - Race or ethnicity
        - Gender
        - Age
        - Disability status

        Therefore, this dashboard cannot determine whether model
        predictions are fair across those individual demographic
        characteristics.

        The current audit evaluates geographic differences in model
        behavior rather than making a claim about demographic fairness.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Mitigation strategies
    # --------------------------------------------------------

    st.subheader(
        "Potential Mitigation Strategies"
    )

    mitigation_data = pd.DataFrame(
        {
            "Approach": [
                "Pre-processing",
                "In-processing",
                "Post-processing",
            ],
            "Description": [
                "Modify training data or representation to reduce "
                "group-related disparities.",
                "Add fairness constraints or fairness-aware objectives "
                "during model training.",
                "Adjust prediction thresholds or outputs using "
                "validated group-level fairness criteria.",
            ],
        }
    )

    st.dataframe(
        mitigation_data,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        """
        No mitigation method was applied to the final model in
        Experiment 5 because the available geographic groups included
        small sample sizes. Applying a mitigation method without
        sufficient group-level evidence could produce unreliable
        conclusions.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Responsible AI interpretation
    # --------------------------------------------------------

    st.subheader(
        "Responsible AI Interpretation"
    )

    st.markdown(
        """
        - Fairness metrics describe differences in model behavior
          between groups.
        - A fairness disparity does not automatically identify its
          underlying cause.
        - Small group sizes can make fairness estimates unstable.
        - State is being used as a geographic grouping variable, not
          as a substitute for unavailable protected demographic data.
        - SHAP explanations describe model behavior and should not be
          interpreted as causal explanations.
        - Model predictions should support analysis and investigation,
          not replace clinical, administrative, or policy judgment.
        """
    )

    st.caption(
        "Responsible AI analysis is based on the Fairlearn audit "
        "performed in Experiment 5."
    )

elif page == "Portfolio":

    st.title("💼 Hospital Quality Prediction — Project Portfolio")

    st.write(
        """
        An end-to-end machine learning system for analyzing hospital
        quality indicators, predicting hospital quality categories, and
        providing explainable and responsible AI insights.
        """
    )

    # --------------------------------------------------------
    # Project overview
    # --------------------------------------------------------

    st.subheader("Project Overview")

    overview_col1, overview_col2 = st.columns(
        [2, 1]
    )

    with overview_col1:

        st.markdown(
            """
            ### Predictive Analysis of Hospital Quality Ratings

            This project combines hospital quality information collected
            from Medicare Care Compare with CMS Hospital General
            Information to build a complete machine-learning workflow.

            The system covers the complete lifecycle from data collection
            and validation to model training, explainability, fairness
            analysis, API deployment, CI/CD, and interactive monitoring.
            """
        )

    with overview_col2:

        st.metric(
            "Hospitals",
            f"{len(df):,}",
        )

        st.metric(
            "Dataset Features",
            "47",
        )

        st.metric(
            "Labeled Hospitals",
            "596",
        )

    st.divider()

    # --------------------------------------------------------
    # Project statistics
    # --------------------------------------------------------

    st.subheader("Project at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Hospitals",
            "835",
        )

    with col2:
        st.metric(
            "Quality Classes",
            "3",
        )

    with col3:
        st.metric(
            "Model Features",
            "34",
        )

    with col4:
        st.metric(
            "Test Samples",
            "120",
        )

    st.divider()

    # --------------------------------------------------------
    # End-to-end pipeline
    # --------------------------------------------------------

    st.subheader("End-to-End ML Pipeline")

    pipeline = pd.DataFrame(
        {
            "Stage": [
                "1. Data Collection",
                "2. Data Integration",
                "3. Profiling & Cleaning",
                "4. Feature Engineering",
                "5. Validation",
                "6. EDA & Statistics",
                "7. Model Training",
                "8. Explainability",
                "9. Fairness Audit",
                "10. API Deployment",
                "11. CI/CD",
                "12. Interactive Dashboard",
            ],
            "Implementation": [
                "Medicare Care Compare web scraping",
                "CMS Hospital General Information merge",
                "Pandas-based profiling and cleaning",
                "Domain-driven quality features",
                "Great Expectations",
                "Statistical analysis and hypothesis testing",
                "Scikit-learn + MLflow",
                "SHAP + LIME",
                "Fairlearn",
                "FastAPI + Docker",
                "GitHub Actions + Ruff + Pytest",
                "Streamlit",
            ],
        }
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    st.subheader("Dataset & Feature Engineering")

    dataset_col1, dataset_col2 = st.columns(
        2
    )

    with dataset_col1:

        st.markdown(
            """
            **Data sources**

            - Medicare Care Compare
            - CMS Hospital General Information
            - Provider/F​​acility ID used for record matching

            **Dataset preparation**

            - 835 hospital records
            - 51 attributes after CMS integration
            - 39 attributes after cleaning
            - 47 attributes after feature engineering
            - 0 duplicate hospital records
            - 100% CMS merge match rate
            """
        )

    with dataset_col2:

        st.markdown(
            """
            **Engineered features**

            - Total better measures
            - Total worse measures
            - Quality balance score
            - Better-measure ratio
            - Worse-measure ratio
            - Quality coverage
            - Hospital size category
            - Rating category

            The engineered features were created using domain-driven
            aggregation, ratios, and rule-based categorization.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Machine learning
    # --------------------------------------------------------

    st.subheader("Machine Learning")

    ml_col1, ml_col2 = st.columns(
        2
    )

    with ml_col1:

        st.markdown(
            """
            ### Prediction Task

            **Target:** `rating_category`

            Three classes were used:

            - Low
            - Medium
            - High

            The model was trained only on the 596 hospitals with an
            available overall rating. The remaining 239 hospitals were
            excluded from supervised model training because their target
            value was unavailable.
            """
        )

    with ml_col2:

        st.markdown(
            """
            ### Final Model

            **Logistic Regression**

            The final model was selected from five baseline models:

            - Logistic Regression
            - Decision Tree
            - Random Forest
            - Gradient Boosting
            - SVM with RBF kernel

            The selected model is the baseline Logistic Regression
            pipeline used consistently across the deployment stack.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    st.subheader("Final Model Performance")

    performance_col1, performance_col2, performance_col3 = (
        st.columns(3)
    )

    with performance_col1:

        st.metric(
            "Accuracy",
            "68.33%",
        )

    with performance_col2:

        st.metric(
            "Macro F1",
            "64.75%",
        )

    with performance_col3:

        st.metric(
            "Weighted F1",
            "67.10%",
        )

    st.caption(
        "Performance values are from the held-out test set used in "
        "Experiment 4."
    )

    st.divider()

    # --------------------------------------------------------
    # Explainability and responsible AI
    # --------------------------------------------------------

    st.subheader(
        "Explainability & Responsible AI"
    )

    xai_col1, xai_col2 = st.columns(
        2
    )

    with xai_col1:

        st.markdown(
            """
            ### 🔍 Explainability

            **SHAP**

            Global model analysis identified the following influential
            transformed features:

            1. Patient survey rating
            2. Mortality better measures
            3. Total worse measures
            4. Mortality no-different measures
            5. Quality coverage
            6. Quality balance score

            **LIME** was additionally used to generate local
            explanations for representative High, Medium, and Low
            predictions.
            """
        )

    with xai_col2:

        st.markdown(
            """
            ### 🛡️ Responsible AI

            Fairness was audited using Fairlearn with **state** as the
            geographic grouping variable.

            The audit identified measurable differences between
            geographic groups, while also highlighting the limitation
            caused by small group sizes.

            The dataset does not contain individual-level race, gender,
            age, or other protected demographic attributes, so the audit
            does not establish demographic fairness.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Engineering & deployment
    # --------------------------------------------------------

    st.subheader("Engineering & Deployment")

    engineering_data = pd.DataFrame(
        {
            "Component": [
                "Model API",
                "Containerization",
                "Testing",
                "Code Quality",
                "CI/CD",
                "Data Versioning",
                "Experiment Tracking",
                "Dashboard",
            ],
            "Technology": [
                "FastAPI",
                "Docker",
                "Pytest",
                "Ruff",
                "GitHub Actions",
                "DVC",
                "MLflow",
                "Streamlit",
            ],
            "Purpose": [
                "Serve hospital quality predictions",
                "Package the application and model",
                "Automated model/API verification",
                "Static code quality checks",
                "Automated testing and Docker build",
                "Version large project datasets",
                "Track ML experiments and metrics",
                "Interactive project monitoring",
            ],
        }
    )

    st.dataframe(
        engineering_data,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Responsible deployment architecture
    # --------------------------------------------------------

    st.subheader("System Architecture")

    architecture = pd.DataFrame(
        {
            "Layer": [
                "Data Layer",
                "ML Layer",
                "Explainability Layer",
                "API Layer",
                "Infrastructure Layer",
                "Monitoring Layer",
            ],
            "Components": [
                "Medicare + CMS + DVC",
                "Scikit-learn + Logistic Regression",
                "SHAP + LIME + Fairlearn",
                "FastAPI",
                "Docker + GitHub Actions",
                "Streamlit + Drift Checks",
            ],
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Project achievements
    # --------------------------------------------------------

    st.subheader("Project Achievements")

    achievements = [
        "Integrated 835 hospitals with a 100% CMS record match rate.",
        "Built a validated 47-column feature-engineered dataset.",
        "Performed statistical analysis and hypothesis testing.",
        "Compared five classification algorithms.",
        "Tracked model experiments using MLflow.",
        "Implemented SHAP and LIME explanations.",
        "Performed a Fairlearn geographic fairness audit.",
        "Packaged the prediction service using FastAPI and Docker.",
        "Implemented automated testing, linting, and Docker builds with GitHub Actions.",
        "Built an interactive Streamlit monitoring and analysis dashboard.",
    ]

    for achievement in achievements:

        st.markdown(
    		"\n".join(
        		f"- {achievement}"
        		for achievement in achievements
    		)
	)

    st.divider()

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    st.subheader("Project Limitations")

    limitations = [
        "239 hospitals do not have an overall rating and therefore cannot be used as labeled samples for the supervised classification task.",
        "The current drift analysis is a baseline comparison within the available dataset rather than production-versus-reference monitoring.",
        "Fairness analysis uses state as a geographic grouping variable because individual-level protected demographic attributes are unavailable.",
        "Some hospital quality measures contain missing values because the source data does not provide them for every hospital.",
        "Statistical associations and model explanations should not be interpreted as causal relationships.",
    ]

    for limitation in limitations:

        st.markdown(
    		"\n".join(
        		f"- {limitation}"
        		for limitation in limitations
    		)
	)

    st.divider()

    # --------------------------------------------------------
    # Project summary
    # --------------------------------------------------------

    st.subheader("Project Summary")

    st.success(
        """
        This project demonstrates an end-to-end machine-learning
        workflow for hospital quality prediction, covering data
        acquisition, integration, validation, statistical analysis,
        machine learning, experiment tracking, explainability,
        fairness auditing, API deployment, containerization, CI/CD,
        and interactive monitoring.
        """
    )

    st.caption(
        "Hospital Quality Prediction — End-to-End Applied Data Science "
        "Project"
    )