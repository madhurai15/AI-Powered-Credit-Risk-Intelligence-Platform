import os
from huggingface_hub import InferenceClient

# =========================================================
# IMPORTS
# =========================================================

try:
    # Used when imported by app.py
    from .prompt_templates import build_nl_to_sql_prompt
    from .query_runner import run_query

except ImportError:
    # Used when running this file directly
    from prompt_templates import build_nl_to_sql_prompt
    from query_runner import run_query


# =========================================================
# HUGGING FACE CONFIGURATION
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# IMPORTANT:
MODEL_NAME = "openai/gpt-oss-20b"


# =========================================================
# CHECK TOKEN
# =========================================================

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is not set."
    )


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

client = InferenceClient(
    api_key=HF_TOKEN
)


# =========================================================
# GENERATE SQL
# =========================================================

def generate_sql(question):

    prompt = build_nl_to_sql_prompt(
        question
    )

    response = client.chat_completion(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=300,

        temperature=0.0
    )

    sql = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    # Remove markdown formatting if LLM adds it
    sql = (
        sql
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    # Security check
    if not sql.lower().startswith("select"):

        raise ValueError(
            "Generated query is not a SELECT statement."
        )

    return sql


# =========================================================
# ASK DATABASE
# =========================================================

def ask_database(question):

    sql = generate_sql(
        question
    )

    print("\nGenerated SQL:")
    print(sql)

    result = run_query(
        sql
    )

    print("\nQuery Result:")
    print(result)

    return result


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    question = input(
        "\nNatural Language: "
    )

    try:

        ask_database(
            question
        )

    except Exception as e:

        print("\nError:")
        print(e)