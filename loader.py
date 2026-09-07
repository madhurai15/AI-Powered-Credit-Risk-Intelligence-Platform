import os
import pandas as pd
import mysql.connector


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the main Home Credit datasets.
    """

    application_train = pd.read_csv(
        os.path.join(DATA_DIR, "application_train.csv")
    )

    bureau = pd.read_csv(
        os.path.join(DATA_DIR, "bureau.csv")
    )

    previous_application = pd.read_csv(
        os.path.join(DATA_DIR, "previous_application.csv")
    )

    return application_train, bureau, previous_application


# ============================================================
# BUREAU AGGREGATION
# ============================================================

def aggregate_bureau(bureau):
    """
    Aggregate bureau information at applicant level.
    """

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_CREDIT_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_MAX=("AMT_CREDIT_SUM", "max")
    ).reset_index()

    return bureau_agg


# ============================================================
# PREVIOUS APPLICATION AGGREGATION
# ============================================================

def aggregate_previous_applications(previous_application):
    """
    Aggregate previous application information
    at applicant level.
    """

    previous_agg = previous_application.groupby("SK_ID_CURR").agg(
        PREV_APPLICATION_COUNT=("SK_ID_PREV", "count"),
        PREV_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_CREDIT_MAX=("AMT_CREDIT", "max")
    ).reset_index()

    return previous_agg


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """
    Create useful credit-risk features.
    """

    # Age in years
    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365
    ).round(2)

    # Credit-to-income ratio
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"]
        / df["AMT_INCOME_TOTAL"].replace(0, float("nan"))
    )

    # Annuity-to-income ratio
    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"]
        / df["AMT_INCOME_TOTAL"].replace(0, float("nan"))
    )

    return df


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_training_data():
    """
    Load, aggregate, merge and prepare the final
    application-level dataset.
    """

    print("Loading Home Credit datasets...")

    application_train, bureau, previous_application = load_data()

    print(
        "Application train shape:",
        application_train.shape
    )

    print(
        "Bureau shape:",
        bureau.shape
    )

    print(
        "Previous application shape:",
        previous_application.shape
    )

    # --------------------------------------------------------
    # Aggregate bureau
    # --------------------------------------------------------

    print("\nAggregating bureau data...")

    bureau_agg = aggregate_bureau(bureau)

    # --------------------------------------------------------
    # Aggregate previous applications
    # --------------------------------------------------------

    print("Aggregating previous application data...")

    previous_agg = aggregate_previous_applications(
        previous_application
    )

    # --------------------------------------------------------
    # Merge datasets
    # --------------------------------------------------------

    print("Merging datasets...")

    df = application_train.merge(
        bureau_agg,
        on="SK_ID_CURR",
        how="left"
    )

    df = df.merge(
        previous_agg,
        on="SK_ID_CURR",
        how="left"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("Creating derived features...")

    df = create_features(df)

    # --------------------------------------------------------
    # Fix anomalous DAYS_EMPLOYED value
    # --------------------------------------------------------

    if "DAYS_EMPLOYED" in df.columns:

        df["DAYS_EMPLOYED"] = df[
            "DAYS_EMPLOYED"
        ].replace(
            365243,
            float("nan")
        )

    print(
        "\nFinal dataset shape:",
        df.shape
    )

    return df


# ============================================================
# LOAD DATA INTO MYSQL
# ============================================================

def load_to_mysql(df):
    """
    Load selected credit-risk features into MySQL.
    """

    # --------------------------------------------------------
    # MySQL configuration
    # --------------------------------------------------------

    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "Madhu&&2002"
    MYSQL_DATABASE = "credit_risk_db"

    # --------------------------------------------------------
    # Columns matching schema.sql
    # --------------------------------------------------------

    mysql_columns = [
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "BUREAU_COUNT",
        "BUREAU_CREDIT_MEAN",
        "BUREAU_CREDIT_MAX",
        "PREV_APPLICATION_COUNT",
        "PREV_APPLICATION_MEAN",
        "PREV_CREDIT_MEAN",
        "PREV_CREDIT_MAX",
        "AGE_YEARS"
    ]

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in mysql_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns for MySQL: {missing_columns}"
        )

    # --------------------------------------------------------
    # Select required columns
    # --------------------------------------------------------

    mysql_df = df[mysql_columns].copy()

    # --------------------------------------------------------
    # Connect to MySQL
    # --------------------------------------------------------

    print("\nConnecting to MySQL...")

    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Create INSERT query
    # --------------------------------------------------------

    placeholders = ", ".join(
        ["%s"] * len(mysql_columns)
    )

    query = f"""
        INSERT INTO credit_risk_data
        ({", ".join(mysql_columns)})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE
            TARGET = VALUES(TARGET),
            AMT_INCOME_TOTAL = VALUES(AMT_INCOME_TOTAL),
            AMT_CREDIT = VALUES(AMT_CREDIT),
            AMT_ANNUITY = VALUES(AMT_ANNUITY),
            AMT_GOODS_PRICE = VALUES(AMT_GOODS_PRICE),
            DAYS_BIRTH = VALUES(DAYS_BIRTH),
            DAYS_EMPLOYED = VALUES(DAYS_EMPLOYED),
            CODE_GENDER = VALUES(CODE_GENDER),
            NAME_EDUCATION_TYPE = VALUES(NAME_EDUCATION_TYPE),
            NAME_FAMILY_STATUS = VALUES(NAME_FAMILY_STATUS),
            NAME_INCOME_TYPE = VALUES(NAME_INCOME_TYPE),
            NAME_HOUSING_TYPE = VALUES(NAME_HOUSING_TYPE),
            OCCUPATION_TYPE = VALUES(OCCUPATION_TYPE),
            ORGANIZATION_TYPE = VALUES(ORGANIZATION_TYPE),
            EXT_SOURCE_1 = VALUES(EXT_SOURCE_1),
            EXT_SOURCE_2 = VALUES(EXT_SOURCE_2),
            EXT_SOURCE_3 = VALUES(EXT_SOURCE_3),
            CREDIT_INCOME_RATIO = VALUES(CREDIT_INCOME_RATIO),
            ANNUITY_INCOME_RATIO = VALUES(ANNUITY_INCOME_RATIO),
            BUREAU_COUNT = VALUES(BUREAU_COUNT),
            BUREAU_CREDIT_MEAN = VALUES(BUREAU_CREDIT_MEAN),
            BUREAU_CREDIT_MAX = VALUES(BUREAU_CREDIT_MAX),
            PREV_APPLICATION_COUNT = VALUES(PREV_APPLICATION_COUNT),
            PREV_APPLICATION_MEAN = VALUES(PREV_APPLICATION_MEAN),
            PREV_CREDIT_MEAN = VALUES(PREV_CREDIT_MEAN),
            PREV_CREDIT_MAX = VALUES(PREV_CREDIT_MAX),
            AGE_YEARS = VALUES(AGE_YEARS)
    """

    # --------------------------------------------------------
    # Insert data in batches
    # --------------------------------------------------------

    batch_size = 5000
    total_rows = len(mysql_df)

    for start in range(
        0,
        total_rows,
        batch_size
    ):

        batch = mysql_df.iloc[
            start:start + batch_size
        ]

        rows = []

        for row in batch.itertuples(
            index=False,
            name=None
        ):

            cleaned_row = tuple(
                None if pd.isna(value)
                else value
                for value in row
            )

            rows.append(cleaned_row)

        cursor.executemany(
            query,
            rows
        )

        connection.commit()

        current = min(
            start + batch_size,
            total_rows
        )

        print(
            f"Loaded {current} / {total_rows} rows"
        )

    # --------------------------------------------------------
    # Close connection
    # --------------------------------------------------------

    cursor.close()
    connection.close()

    print(
        "\nData successfully loaded into MySQL!"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Prepare dataset
    df = prepare_training_data()

    # Display first 5 rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Load into MySQL
    load_to_mysql(df)