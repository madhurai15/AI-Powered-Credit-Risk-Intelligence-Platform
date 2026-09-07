import os

from sqlalchemy import create_engine, text
import pandas as pd


# ============================================================
# MYSQL CONFIGURATION
# ============================================================

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Madhu&&2002")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "credit_risk_db")


# ============================================================
# CREATE DATABASE ENGINE
# ============================================================

DATABASE_URL = (
    "mysql+mysqlconnector://"
    f"{MYSQL_USER}:{MYSQL_PASSWORD}@"
    f"{MYSQL_HOST}:{MYSQL_PORT}/"
    f"{MYSQL_DATABASE}"
)

engine = create_engine(DATABASE_URL)


# ============================================================
# RUN SQL QUERY
# ============================================================

def run_query(sql):
    """
    Execute a SELECT query and return the result as a DataFrame.
    """

    sql = sql.strip()

    if not sql.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    with engine.connect() as connection:
        result = pd.read_sql(
            text(sql),
            connection
        )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_sql = """
    SELECT
        TARGET,
        COUNT(*) AS applicant_count
    FROM credit_risk_data
    GROUP BY TARGET;
    """

    try:
        result = run_query(test_sql)

        print("\nQuery Result:")
        print(result)

    except Exception as e:
        print("\nDatabase Error:")
        print(e)