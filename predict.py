import os
import sys
import joblib
import pandas as pd
import shap


# ==================================================
# PROJECT PATH
# ==================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from data.preprocessor import clean_data


# ==================================================
# MODEL PATH
# ==================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_risk_model.joblib"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(MODEL_PATH)


# ==================================================
# PREDICT RISK
# ==================================================

def predict_risk(applicant_data):

    applicant = pd.DataFrame([applicant_data])

    # Clean applicant data
    applicant = clean_data(applicant)

    # Features expected by the trained pipeline
    expected_features = model.feature_names_in_

    applicant = applicant.reindex(
        columns=expected_features,
        fill_value=0
    )

    # ==================================================
    # PREDICTION
    # ==================================================

    probability = model.predict_proba(applicant)[0][1]

    if probability < 0.30:
        risk_category = "Low Risk"
    elif probability < 0.60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    print("\n================================")
    print("      CREDIT RISK PREDICTION")
    print("================================")

    print(f"Default Probability: {probability:.2%}")
    print(f"Risk Category: {risk_category}")

    # ==================================================
    # SHAP EXPLANATION
    # ==================================================

    print("\n================================")
    print("        SHAP EXPLANATION")
    print("================================")

    try:

        # Extract preprocessing and Random Forest
        preprocessor = model.named_steps["preprocessor"]
        rf_model = model.named_steps["model"]

        # Apply the same preprocessing used during training
        transformed_applicant = preprocessor.transform(
            applicant
        )

        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        # Convert to DataFrame
        transformed_applicant = pd.DataFrame(
            transformed_applicant,
            columns=feature_names
        )

        # SHAP for Random Forest
        explainer = shap.TreeExplainer(rf_model)

        shap_values = explainer.shap_values(
            transformed_applicant
        )

        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            values = shap_values[1][0]

        elif len(shap_values.shape) == 3:

            values = shap_values[0, :, 1]

        else:

            values = shap_values[0]

        # Make sure lengths match
        values = values[:len(feature_names)]

        explanation = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Value": values,
            "Feature_Value": transformed_applicant.iloc[0].values
        })

        # Absolute importance
        explanation["Absolute_SHAP"] = (
            explanation["SHAP_Value"].abs()
        )

        # Sort by importance
        explanation = explanation.sort_values(
            "Absolute_SHAP",
            ascending=False
        )

        print("\nTop Risk Drivers:\n")

        for _, row in explanation.head(10).iterrows():

            if row["SHAP_Value"] > 0:
                direction = "Increases risk"
            else:
                direction = "Decreases risk"

            print(
                f"{row['Feature']}: "
                f"{row['Feature_Value']} "
                f"-> {direction} "
                f"(SHAP = {row['SHAP_Value']:.4f})"
            )

        return probability, risk_category, explanation

    except Exception as e:

        print("SHAP explanation failed.")
        print("Reason:", e)

        return probability, risk_category, None


# ==================================================
# TEST APPLICANT
# ==================================================

if __name__ == "__main__":

    applicant = {

        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 450000,

        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,

        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_INCOME_TYPE": "Working",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",

        "EXT_SOURCE_1": 0.60,
        "EXT_SOURCE_2": 0.50,
        "EXT_SOURCE_3": 0.55,

        "CREDIT_INCOME_RATIO": 2.78,
        "ANNUITY_INCOME_RATIO": 0.14,

        "BUREAU_COUNT": 4,
        "BUREAU_CREDIT_MEAN": 250000,
        "BUREAU_CREDIT_MAX": 600000,

        "PREV_APPLICATION_COUNT": 3,
        "PREV_APPLICATION_MEAN": 300000,
        "PREV_CREDIT_MEAN": 280000,
        "PREV_CREDIT_MAX": 500000,

        "AGE_YEARS": 32
    }

    try:

        predict_risk(applicant)

    except Exception as e:

        print("\nError:")
        print(e)