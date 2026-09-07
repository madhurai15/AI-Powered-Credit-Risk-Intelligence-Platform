import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "TARGET"


def clean_data(df):
    """
    Perform basic cleaning and feature engineering.
    """

    data = df.copy()

    # --------------------------------------------------
    # 1. Handle DAYS_EMPLOYED anomaly
    # --------------------------------------------------
    if "DAYS_EMPLOYED" in data.columns:

        data["DAYS_EMPLOYED"] = data[
            "DAYS_EMPLOYED"
        ].replace(365243,float("nan"))

    # --------------------------------------------------
    # 2. Create age feature
    # --------------------------------------------------
    if "DAYS_BIRTH" in data.columns:

        data["AGE_YEARS"] = (
            -data["DAYS_BIRTH"] / 365.25
        )

    # --------------------------------------------------
    # 3. Credit-to-income ratio
    # --------------------------------------------------
    if (
        "AMT_CREDIT" in data.columns
        and "AMT_INCOME_TOTAL" in data.columns
    ):

        income = data["AMT_INCOME_TOTAL"].replace(0, pd.NA)

        data["CREDIT_INCOME_RATIO"] = (
            data["AMT_CREDIT"] / income
        )

    # --------------------------------------------------
    # 4. Annuity-to-income ratio
    # --------------------------------------------------
    if (
        "AMT_ANNUITY" in data.columns
        and "AMT_INCOME_TOTAL" in data.columns
    ):

        income = data["AMT_INCOME_TOTAL"].replace(0, pd.NA)

        data["ANNUITY_INCOME_RATIO"] = (
            data["AMT_ANNUITY"] / income
        )

    return data


def split_features_target(df):
    """
    Separate features (X) and target (y).
    """

    data = clean_data(df)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            "TARGET column is missing from the dataset."
        )

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    return X, y


def create_preprocessor(X):
    """
    Create preprocessing pipeline for numerical
    and categorical features.
    """

    # Numerical columns
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Categorical columns
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


if __name__ == "__main__":

    from loader import prepare_training_data

    # Load data
    df = prepare_training_data()

    # Split X and y
    X, y = split_features_target(df)

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X)

    print("Preprocessor created successfully.")
    print("Features:", X.shape)
    print("Target:", y.shape)
    print("Default rate:", y.mean())