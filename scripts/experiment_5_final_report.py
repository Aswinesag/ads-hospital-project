import os
import pandas as pd


# ============================================================
# EXPERIMENT 5 - STAGE 5
# FINAL REPORT, MITIGATION & CONCLUSION
# ============================================================

print("=" * 70)
print("EXPERIMENT 5 - STAGE 5: FINAL REPORT")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

REPORT_DIR = "reports/experiment_5"

FAIRNESS_DIR = os.path.join(
    REPORT_DIR,
    "fairness"
)

SHAP_IMPORTANCE_PATH = os.path.join(
    REPORT_DIR,
    "shap_feature_importance.csv"
)

SHAP_CLASS_PATH = os.path.join(
    REPORT_DIR,
    "shap_class_feature_importance.csv"
)

LIME_PATH = os.path.join(
    REPORT_DIR,
    "lime_explanations.csv"
)

FAIRNESS_SUMMARY_PATH = os.path.join(
    FAIRNESS_DIR,
    "fairness_summary.csv"
)

FAIRNESS_GROUP_PATH = os.path.join(
    FAIRNESS_DIR,
    "fairness_group_metrics.csv"
)

FINAL_REPORT_PATH = os.path.join(
    REPORT_DIR,
    "experiment_5_final_report.txt"
)

FINAL_SUMMARY_PATH = os.path.join(
    REPORT_DIR,
    "experiment_5_summary.csv"
)


# ------------------------------------------------------------
# 2. Load existing results
# ------------------------------------------------------------

print("\nLoading Stage 2 SHAP results...")

shap_df = pd.read_csv(
    SHAP_IMPORTANCE_PATH
)

shap_class_df = pd.read_csv(
    SHAP_CLASS_PATH
)

print(
    "SHAP results loaded:",
    len(shap_df),
    "features"
)


print("\nLoading Stage 3 LIME results...")

lime_df = pd.read_csv(
    LIME_PATH
)

print(
    "LIME results loaded:",
    len(lime_df),
    "local explanations"
)


print("\nLoading Stage 4 fairness results...")

fairness_summary = pd.read_csv(
    FAIRNESS_SUMMARY_PATH
)

fairness_group = pd.read_csv(
    FAIRNESS_GROUP_PATH
)

print(
    "Fairness results loaded:",
    len(fairness_group),
    "group/class records"
)


# ------------------------------------------------------------
# 3. SHAP findings
# ------------------------------------------------------------

top_10_shap = (
    shap_df
    .sort_values(
        "mean_absolute_shap",
        ascending=False
    )
    .head(10)
)


print("\n" + "-" * 70)
print("TOP SHAP FEATURES")
print("-" * 70)

print(
    top_10_shap[
        [
            "rank",
            "feature_name",
            "mean_absolute_shap"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 4. LIME findings
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("LIME EXPLANATIONS")
print("-" * 70)

print(
    lime_df[
        [
            "example_class",
            "actual_class",
            "predicted_class"
        ]
    ]
    .drop_duplicates()
    .to_string(index=False)
)


# ------------------------------------------------------------
# 5. Fairness findings
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("FAIRNESS SUMMARY")
print("-" * 70)

print(
    fairness_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 6. Identify small groups
# ------------------------------------------------------------

small_groups = (
    fairness_group[
        fairness_group["small_group_flag"]
    ]["state"]
    .unique()
    .tolist()
)

small_groups = sorted(
    small_groups
)

print("\nSmall test groups:")
print(small_groups)


# ------------------------------------------------------------
# 7. Determine observed disparities
# ------------------------------------------------------------

max_dp = fairness_summary[
    "demographic_parity_difference"
].max()

max_eo = fairness_summary[
    "equalized_odds_difference"
].max()


if max_dp > 0 or max_eo > 0:

    disparity_observed = True

else:

    disparity_observed = False


# ------------------------------------------------------------
# 8. Build mitigation section
# ------------------------------------------------------------

if disparity_observed:

    mitigation_text = """
BIAS MITIGATION STRATEGIES

The fairness audit identified measurable disparities across
state-based groups. Because the dataset does not contain
individual-level demographic attributes such as gender or race,
these findings represent geographic group disparities rather
than protected-demographic fairness conclusions.

The following mitigation strategies are proposed:

1. PRE-PROCESSING
   - Apply group-aware reweighting during model training.
   - Consider resampling underrepresented geographic groups.
   - Review whether geographic variables such as city, county,
     and state should remain model predictors.

2. IN-PROCESSING
   - Apply fairness-constrained learning methods that optimize
     predictive performance while controlling group disparities.

3. POST-PROCESSING
   - Evaluate group-aware classification threshold adjustments
     using a validation dataset.
   - Compare fairness metrics against predictive performance
     before selecting any modified model.

No mitigation was applied to the final model in this experiment.
The reason is that several state groups have very small test
sample sizes, making direct mitigation decisions from this test
set unreliable. A larger validation dataset should be used before
deploying a fairness-adjusted model.
"""

else:

    mitigation_text = """
BIAS MITIGATION STRATEGIES

No measurable disparity was detected in the evaluated fairness
metrics. Therefore, no mitigation was applied.

Future work could still evaluate reweighting, resampling,
fairness-constrained learning, or post-processing if additional
sensitive attributes become available.
"""


# ------------------------------------------------------------
# 9. Build final report
# ------------------------------------------------------------

report = f"""
======================================================================
EXPERIMENT 5 - EXPLAINABLE AI AND FAIRNESS
======================================================================

AIM
----------------------------------------------------------------------
Apply Explainable AI methods (SHAP and LIME) to interpret the hospital
quality classification model and evaluate prediction disparities using
Fairlearn.

MODEL
----------------------------------------------------------------------
Final model:
Logistic Regression pipeline selected in Experiment 4.

Task:
Multiclass classification of hospitals into:

- Low
- Medium
- High

Dataset:
596 labeled hospitals.

Train/Test split:
476 training hospitals
120 testing hospitals

Original predictors:
34

Transformed model features:
477


======================================================================
STAGE 2 - SHAP EXPLAINABILITY
======================================================================

A SHAP LinearExplainer was applied to the Logistic Regression model.

SHAP output:
120 test samples
477 transformed features
3 classes

The global feature importance was calculated using mean absolute SHAP
values across test samples and classes.

TOP 10 FEATURES
----------------------------------------------------------------------

{top_10_shap[
    [
        "rank",
        "feature_name",
        "mean_absolute_shap"
    ]
].to_string(index=False)}


INTERPRETATION
----------------------------------------------------------------------

Patient survey rating had the largest mean absolute SHAP value among
the transformed features. Mortality, readmission, quality-balance,
quality-coverage, and better/worse-measure features also contributed
substantially to model predictions.

SHAP magnitude indicates the strength of a feature's contribution.
It does not by itself establish whether a feature increases or
decreases a predicted class.

The model contains one-hot encoded categorical variables, so individual
city, county, ownership, and hospital-size categories appear as
separate transformed features.


======================================================================
STAGE 3 - LIME EXPLAINABILITY
======================================================================

LIME was used to explain individual predictions.

Three representative correctly classified examples were selected:

High:
Actual = High
Predicted = High

Low:
Actual = Low
Predicted = Low

Medium:
Actual = Medium
Predicted = Medium

Ten local feature contributions were generated for each example,
resulting in 30 explanation records.

The High example showed a strong contribution from patient survey
rating and mortality-related information.

The Low example showed a strong contribution from a low standardized
patient survey rating together with several geographic categorical
features.

The Medium example was substantially influenced by one-hot encoded
city indicators.

LIME explanations are local explanations for individual predictions
and should not be interpreted as global feature importance.


======================================================================
STAGE 4 - FAIRNESS AUDIT
======================================================================

Sensitive/grouping attribute:
State

The dataset does not contain individual-level demographic attributes
such as gender, race, or age. Therefore, the fairness analysis evaluates
prediction disparities across geographic hospital groups rather than
claiming protected-demographic fairness.

Because the target is multiclass, each class was evaluated using a
one-vs-rest formulation.


FAIRNESS RESULTS
----------------------------------------------------------------------

{fairness_summary.to_string(index=False)}


INTERPRETATION
----------------------------------------------------------------------

The audit identified measurable differences in prediction rates and
error rates across states.

Maximum demographic parity difference:
{max_dp:.4f}

Maximum equalized odds difference:
{max_eo:.4f}

These values indicate observed disparities across the geographic
groups in this particular test sample.

However, four states had fewer than 10 test hospitals:

{", ".join(small_groups)}

Therefore, the group-level fairness results for these states should
not be generalized without a larger evaluation sample.


======================================================================
STAGE 5 - BIAS MITIGATION
======================================================================

{mitigation_text}


======================================================================
LIMITATIONS
======================================================================

1. The dataset contains hospital-level geographic and organizational
   attributes but does not contain individual demographic attributes
   such as gender or race.

2. Therefore, the fairness audit cannot establish fairness with respect
   to those demographic attributes.

3. The test set contains only 120 hospitals.

4. Several geographic groups have small sample sizes.

5. LIME explanations can be affected by the one-hot encoded geographic
   variables.

6. SHAP feature importance describes model contribution rather than
   causation.

7. Correlation or model explanation does not establish that a feature
   causes hospital quality.


======================================================================
CONCLUSION
======================================================================

The experiment successfully applied SHAP and LIME to explain the final
hospital quality classification model and Fairlearn to evaluate
prediction disparities across available geographic groups.

SHAP provided a global view of feature influence, with patient survey
rating having the largest average contribution among the transformed
features. LIME provided local explanations for individual High, Low,
and Medium predictions.

The Fairlearn audit identified measurable geographic disparities in
prediction and error rates. However, small state-level sample sizes
limit the strength of conclusions that can be drawn from these
differences.

The experiment therefore demonstrates both model interpretability and
a structured fairness-audit workflow while explicitly documenting the
limitations of the available fairness attributes and evaluation
sample.


======================================================================
EXPERIMENT 5 COMPLETED
======================================================================
"""


# ------------------------------------------------------------
# 10. Save final report
# ------------------------------------------------------------

with open(
    FINAL_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print("\nFinal report saved to:")
print(
    os.path.abspath(
        FINAL_REPORT_PATH
    )
)


# ------------------------------------------------------------
# 11. Create compact summary table
# ------------------------------------------------------------

summary_records = [
    {
        "component": "SHAP",
        "result": "Global feature importance generated",
        "output": "shap_feature_importance.csv"
    },
    {
        "component": "LIME",
        "result": "3 local explanations generated",
        "output": "lime_explanations.csv"
    },
    {
        "component": "Fairlearn",
        "result": "Geographic fairness audit completed",
        "output": "fairness_summary.csv"
    },
    {
        "component": "Mitigation",
        "result": "Pre-, in-, and post-processing strategies proposed",
        "output": "experiment_5_final_report.txt"
    }
]

summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    FINAL_SUMMARY_PATH,
    index=False
)

print("\nSummary saved to:")
print(
    os.path.abspath(
        FINAL_SUMMARY_PATH
    )
)


# ------------------------------------------------------------
# 12. Final validation
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("FINAL VALIDATION")
print("-" * 70)

print(
    "SHAP results available:",
    os.path.exists(
        SHAP_IMPORTANCE_PATH
    )
)

print(
    "LIME results available:",
    os.path.exists(
        LIME_PATH
    )
)

print(
    "Fairness results available:",
    os.path.exists(
        FAIRNESS_SUMMARY_PATH
    )
)

print(
    "Final report created:",
    os.path.exists(
        FINAL_REPORT_PATH
    )
)

print(
    "Summary created:",
    os.path.exists(
        FINAL_SUMMARY_PATH
    )
)


# ------------------------------------------------------------
# 13. Final status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 5 FINAL REPORT COMPLETED")
print("=" * 70)