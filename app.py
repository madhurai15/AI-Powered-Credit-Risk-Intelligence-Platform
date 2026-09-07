import os
import sys
import joblib
import shap
import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "credit_risk_model.joblib"
)

from preprocessor import clean_data


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CreditRisk AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown(
    """
<style>

/* Main application */
.stApp {
    background-color: #f8fafc;
}

/* Content width */
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Sidebar navigation */
section[data-testid="stSidebar"] .stRadio label {
    color: #e5e7eb !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #111827 !important;
}

/* Normal text */
p {
    color: #475569;
}

/* Metric */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background-color: #2563eb;
    border-color: #2563eb;
}

/* Text inputs */
div[data-baseweb="input"] {
    border-radius: 9px;
}

/* Input text */
div[data-baseweb="input"] input {
    color: #111827 !important;
}

/* Select boxes */
div[data-baseweb="select"] {
    border-radius: 9px;
}

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        os.path.join(
            DATA_DIR,
            "application_train.csv"
        )
    )


# =========================================================
# RISK CATEGORY
# =========================================================

def get_risk(probability):

    if probability < 0.30:

        return (
            "Low Risk",
            "Lower estimated probability of default."
        )

    elif probability < 0.60:

        return (
            "Medium Risk",
            "Moderate estimated probability of default."
        )

    else:

        return (
            "High Risk",
            "Higher estimated probability of default."
        )


# =========================================================
# SHAP EXPLANATION
# =========================================================

def generate_shap_explanation(
    model,
    applicant
):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    rf_model = model.named_steps[
        "model"
    ]

    transformed = preprocessor.transform(
        applicant
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    explainer = shap.TreeExplainer(
        rf_model
    )

    shap_values = explainer.shap_values(
        transformed_df
    )

    if isinstance(
        shap_values,
        list
    ):

        values = shap_values[1][0]

    elif len(
        shap_values.shape
    ) == 3:

        values = shap_values[0, :, 1]

    else:

        values = shap_values[0]

    explanation = pd.DataFrame(
        {
            "Feature":
                feature_names,

            "SHAP Value":
                values,

            "Feature Value":
                transformed_df.iloc[0].values
        }
    )

    explanation["Impact"] = (
        explanation["SHAP Value"]
        .apply(
            lambda x:
            "Increases risk"
            if x > 0
            else "Decreases risk"
        )
    )

    explanation["Absolute Impact"] = (
        explanation["SHAP Value"].abs()
    )

    return explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )


# =========================================================
# FORMAT TALK-TO-DATA ANSWER
# =========================================================

def format_database_answer(
    question,
    result
):

    if result.empty:

        return "I couldn't find a result for that question."

    row = result.iloc[0]

    column_name = str(
        result.columns[0]
    ).lower()

    value = row.iloc[0]

    question_lower = question.lower()

    # -----------------------------------------------------
    # DEFAULT RATE
    # -----------------------------------------------------

    if (
        "default rate" in question_lower
        or "default percentage" in question_lower
        or "percentage of default" in question_lower
    ):

        try:

            numeric_value = float(
                value
            )

            if numeric_value <= 1:

                numeric_value *= 100

            return (
                f"The overall loan default rate "
                f"is **{numeric_value:.2f}%**."
            )

        except:

            return (
                f"The default rate is **{value}**."
            )

    # -----------------------------------------------------
    # NUMBER OF DEFAULTS
    # -----------------------------------------------------

    if (
        "how many" in question_lower
        and "default" in question_lower
    ):

        try:

            return (
                f"There are **{int(value):,} "
                f"defaulted applicants**."
            )

        except:

            return (
                f"The result is **{value}**."
            )

    # -----------------------------------------------------
    # AVERAGE INCOME
    # -----------------------------------------------------

    if (
        "average income" in question_lower
        or "avg income" in question_lower
    ):

        try:

            return (
                f"The average income is "
                f"**{float(value):,.2f}**."
            )

        except:

            return (
                f"The average income is **{value}**."
            )

    # -----------------------------------------------------
    # AVERAGE CREDIT
    # -----------------------------------------------------

    if (
        "average credit" in question_lower
        or "avg credit" in question_lower
    ):

        try:

            return (
                f"The average credit amount is "
                f"**{float(value):,.2f}**."
            )

        except:

            return (
                f"The average credit amount is **{value}**."
            )

    # -----------------------------------------------------
    # GENERIC SINGLE RESULT
    # -----------------------------------------------------

    if len(result.columns) == 1:

        if isinstance(
            value,
            float
        ):

            return (
                f"The result is **{value:,.2f}**."
            )

        if isinstance(
            value,
            int
        ):

            return (
                f"The result is **{value:,}**."
            )

        return (
            f"The result is **{value}**."
        )

    # -----------------------------------------------------
    # MULTIPLE RESULTS
    # -----------------------------------------------------

    parts = []

    for column in result.columns:

        value = row[column]

        if isinstance(
            value,
            float
        ):

            value = f"{value:,.2f}"

        elif isinstance(
            value,
            int
        ):

            value = f"{value:,}"

        parts.append(
            f"**{column}:** {value}"
        )

    return " · ".join(parts)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "◈ CreditRisk AI"
    )

    st.caption(
        "Intelligent Credit Risk Platform"
    )

    st.divider()

    section = st.radio(
        "Navigation",
        [
            "Overview",
            "EDA & Insights",
            "Risk Prediction",
            "Explainability",
            "Decision Rules",
            "Talk to Data"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption(
        "Home Credit Default Risk"
    )

    st.caption(
        "AI-powered decision support"
    )


# =========================================================
# OVERVIEW
# =========================================================

if section == "Overview":

    data = load_data()

    total_applicants = len(
        data
    )

    defaults = int(
        data["TARGET"].sum()
    )

    non_defaults = (
        total_applicants -
        defaults
    )

    default_rate = (
        data["TARGET"].mean()
        * 100
    )

    st.title(
        "Credit Risk Intelligence Dashboard"
    )

    st.caption(
        "AI-powered analysis, risk prediction, explainability "
        "and natural-language data access."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Applicants",
            f"{total_applicants:,}"
        )

    with c2:

        st.metric(
            "Default Rate",
            f"{default_rate:.2f}%"
        )

    with c3:

        st.metric(
            "Defaults",
            f"{defaults:,}"
        )

    with c4:

        st.metric(
            "Non-Defaults",
            f"{non_defaults:,}"
        )

    st.divider()

    left, right = st.columns(
        [1.2, 1]
    )

    with left:

        st.subheader(
            "Portfolio Default Distribution"
        )

        distribution = pd.DataFrame(
            {
                "Outcome":
                    [
                        "Non-default",
                        "Default"
                    ],

                "Applicants":
                    [
                        non_defaults,
                        defaults
                    ]
            }
        )

        st.bar_chart(
            distribution.set_index(
                "Outcome"
            )
        )

    with right:

        st.subheader(
            "Key Risk Insights"
        )

        insights = [
            "Default rate is 8.07% across the applicant portfolio.",
            "24,825 applicants are classified as defaults.",
            "Lower external credit scores are associated with higher default risk.",
            "Younger applicant groups show relatively higher observed default rates.",
            "Higher annuity-to-income indicates greater repayment pressure."
        ]

        for insight in insights:
            st.markdown(
                "• " + insight
            )



# =========================================================
# EDA & INSIGHTS
# =========================================================

elif section == "EDA & Insights":

    data = load_data()

    st.title(
        "EDA & Business Insights"
    )

    st.caption(
        "Explore demographics, financial characteristics "
        "and default behavior."
    )

    total = len(data)

    defaults = int(
        data["TARGET"].sum()
    )

    default_rate = (
        data["TARGET"].mean()
        * 100
    )

    missing_cells = int(
        data.isna().sum().sum()
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Dataset Rows",
            f"{total:,}"
        )

    with c2:

        st.metric(
            "Default Rate",
            f"{default_rate:.2f}%"
        )

    with c3:

        st.metric(
            "Missing Cells",
            f"{missing_cells:,}"
        )

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "Default Behavior",
            "External Credit Score",
            "Demographics"
        ]
    )

    # -----------------------------------------------------
    # AGE
    # -----------------------------------------------------

    with tab1:

        st.subheader(
            "Default Rate by Age Group"
        )

        temp = data.copy()

        temp["AGE_YEARS"] = (
            -temp["DAYS_BIRTH"]
            / 365.25
        )

        temp["AGE_GROUP"] = pd.cut(
            temp["AGE_YEARS"],
            bins=[
                18,
                25,
                35,
                45,
                55,
                70,
                100
            ],
            labels=[
                "18–25",
                "26–35",
                "36–45",
                "46–55",
                "56–70",
                "70+"
            ]
        )

        age_data = (
            temp
            .groupby(
                "AGE_GROUP",
                observed=True
            )["TARGET"]
            .mean()
            .mul(100)
            .rename(
                "Default Rate (%)"
            )
            .to_frame()
        )

        st.bar_chart(
            age_data
        )

        st.caption(
            "Younger applicant groups generally show higher observed default rates."
        )

    # -----------------------------------------------------
    # EXT SOURCE
    # -----------------------------------------------------

    with tab2:

        st.subheader(
            "Default Rate by EXT_SOURCE_2"
        )

        temp = data.dropna(
            subset=[
                "EXT_SOURCE_2"
            ]
        ).copy()

        bins = pd.qcut(
            temp["EXT_SOURCE_2"],
            q=5,
            duplicates="drop"
        )

        score_data = (
            temp
            .groupby(
                bins,
                observed=True
            )["TARGET"]
            .mean()
            .mul(100)
            .reset_index()
        )

        score_data["Score Group"] = (
            score_data[
                "EXT_SOURCE_2"
            ].astype(str)
        )

        score_chart = (
            score_data[
                [
                    "Score Group",
                    "TARGET"
                ]
            ]
            .rename(
                columns={
                    "TARGET":
                    "Default Rate (%)"
                }
            )
            .set_index(
                "Score Group"
            )
        )

        st.bar_chart(
            score_chart
        )

        st.caption(
            "Higher external credit scores are associated with lower observed default risk."
        )

    # -----------------------------------------------------
    # EDUCATION
    # -----------------------------------------------------

    with tab3:

        st.subheader(
            "Default Rate by Education"
        )

        education_data = (
            data
            .groupby(
                "NAME_EDUCATION_TYPE"
            )["TARGET"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
            .rename(
                "Default Rate (%)"
            )
            .to_frame()
        )

        st.bar_chart(
            education_data
        )

    st.divider()

    st.subheader(
        "Key Business Insights"
    )

    insights = [

        "The dataset is strongly imbalanced, with defaults representing about 8% of applicants.",

        "EXT_SOURCE_2 shows a strong relationship with default behavior.",

        "Younger applicant groups generally exhibit higher observed default rates.",

        "Education categories show different levels of observed default risk.",

        "Higher annuity-to-income burden can indicate greater repayment pressure.",

        "Credit history and previous application aggregates add applicant-level risk context."
    ]

    for insight in insights:

        st.write(
            "• " + insight
        )


# =========================================================
# RISK PREDICTION
# =========================================================

elif section == "Risk Prediction":

    st.title(
        "Applicant Risk Assessment"
    )

    st.caption(
        "Enter applicant information to estimate "
        "the probability of loan default."
    )

    model = load_model()

    with st.form(
        "risk_form"
    ):

        st.subheader(
            "Financial Information"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=180000.0,
                step=10000.0
            )

        with c2:

            credit = st.number_input(
                "Credit Amount",
                min_value=0.0,
                value=500000.0,
                step=10000.0
            )

        with c3:

            annuity = st.number_input(
                "Annual Annuity",
                min_value=0.0,
                value=25000.0,
                step=1000.0
            )

        st.subheader(
            "Applicant Profile"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

        with c2:

            gender = st.selectbox(
                "Gender",
                [
                    "M",
                    "F"
                ]
            )

        with c3:

            family_status = st.selectbox(
                "Family Status",
                [
                    "Married",
                    "Single / not married",
                    "Civil marriage",
                    "Separated",
                    "Widow"
                ]
            )

        st.subheader(
            "External Credit Scores"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            ext_source_1 = st.number_input(
                "External Source 1",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

        with c2:

            ext_source_2 = st.number_input(
                "External Source 2",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

        with c3:

            ext_source_3 = st.number_input(
                "External Source 3",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01
            )

        education = st.selectbox(
            "Education Type",
            [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree"
            ]
        )

        submitted = st.form_submit_button(
            "Assess Credit Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:

        applicant = pd.DataFrame(
            [
                {
                    "AMT_INCOME_TOTAL":
                        income,

                    "AMT_CREDIT":
                        credit,

                    "AMT_ANNUITY":
                        annuity,

                    "DAYS_BIRTH":
                        -age * 365.25,

                    "CODE_GENDER":
                        gender,

                    "NAME_EDUCATION_TYPE":
                        education,

                    "NAME_FAMILY_STATUS":
                        family_status,

                    "EXT_SOURCE_1":
                        ext_source_1,

                    "EXT_SOURCE_2":
                        ext_source_2,

                    "EXT_SOURCE_3":
                        ext_source_3
                }
            ]
        )

        applicant = clean_data(
            applicant
        )

        applicant = applicant.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        probability = (
            model
            .predict_proba(
                applicant
            )[0][1]
        )

        risk, description = get_risk(
            probability
        )

        st.divider()

        st.subheader(
            "Risk Assessment"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Default Probability",
                f"{probability:.2%}"
            )

        with c2:

            st.metric(
                "Risk Category",
                risk
            )

        st.info(
            description
        )

        st.caption(
            "Low Risk < 30% · "
            "Medium Risk 30–60% · "
            "High Risk ≥ 60%"
        )


# =========================================================
# EXPLAINABILITY
# =========================================================

elif section == "Explainability":

    st.title(
        "Explainable AI"
    )

    st.caption(
        "Understand which features are driving the model prediction."
    )

    model = load_model()

    applicant = pd.DataFrame(
        [
            {
                "AMT_INCOME_TOTAL":
                    180000,

                "AMT_CREDIT":
                    500000,

                "AMT_ANNUITY":
                    25000,

                "DAYS_BIRTH":
                    -35 * 365.25,

                "CODE_GENDER":
                    "M",

                "NAME_EDUCATION_TYPE":
                    "Higher education",

                "NAME_FAMILY_STATUS":
                    "Married",

                "EXT_SOURCE_1":
                    0.60,

                "EXT_SOURCE_2":
                    0.55,

                "EXT_SOURCE_3":
                    0.55
            }
        ]
    )

    applicant = clean_data(
        applicant
    )

    applicant = applicant.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    probability = (
        model
        .predict_proba(
            applicant
        )[0][1]
    )

    risk, _ = get_risk(
        probability
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Default Probability",
            f"{probability:.2%}"
        )

    with c2:

        st.metric(
            "Risk Band",
            risk
        )

    st.divider()

    try:

        explanation = (
            generate_shap_explanation(
                model,
                applicant
            )
        )

        st.subheader(
            "Top Risk Drivers"
        )

        st.dataframe(
            explanation.head(10)[
                [
                    "Feature",
                    "SHAP Value",
                    "Feature Value",
                    "Impact"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "SHAP Impact"
        )

        chart = (
            explanation
            .head(10)
            .set_index(
                "Feature"
            )["SHAP Value"]
            .sort_values()
        )

        st.bar_chart(
            chart
        )

        st.caption(
            "Positive SHAP values increase predicted default risk. "
            "Negative values decrease risk."
        )

    except Exception as e:

        st.error(
            f"Unable to generate SHAP explanation: {e}"
        )


# =========================================================
# DECISION RULES
# =========================================================

elif section == "Decision Rules":

    st.title(
        "Business Decision Rules"
    )

    st.caption(
        "Business-readable rules derived from EDA findings "
        "and model behavior."
    )

    st.divider()

    rules = [

        (
            "Rule 1 — External Credit Score",
            "Lower EXT_SOURCE values indicate higher observed credit risk."
        ),

        (
            "Rule 2 — Applicant Age",
            "Younger applicant groups generally show higher observed default rates."
        ),

        (
            "Rule 3 — Repayment Burden",
            "A higher annuity-to-income ratio indicates greater repayment pressure."
        ),

        (
            "Rule 4 — Credit History",
            "Historical bureau and previous-application information provides additional applicant-level risk context."
        ),

        (
            "Rule 5 — Risk Probability",
            "Default probability below 30% is classified as Low Risk; 30–60% as Medium Risk; and 60% or above as High Risk."
        )
    ]

    for title, description in rules:

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {title}"
            )

            st.write(
                description
            )

    st.divider()

    st.info(
        "These rules are decision-support interpretations "
        "and should not be treated as standalone lending policy."
    )
## Talk to data

elif section == "Talk to Data":

    st.title("Talk to Data")
    st.caption(
        "Ask questions about your credit-risk data and get instant insights."
    )

    st.markdown("")

    with st.container(border=True):

        st.subheader("Ask a Question")

        st.markdown(
            "➜ **Select a question below.**"
        )

        st.markdown("")

        example_questions = [
            "Select an example question",
            "What is the default rate?",
            "How many applicants are defaults?",
            "What is the average income of defaulted applicants?",
            "What is the average credit amount?"
        ]

        selected_question = st.selectbox(
            "Question",
            example_questions,
            index=0
        )

        st.markdown("")

        col1, col2, col3 = st.columns([1.2, 1.2, 4])

        with col1:
            ask_button = st.button(
                "🔍 Ask AI",
                type="primary",
                use_container_width=True
            )

        with col2:
            clear_button = st.button(
                "Clear",
                use_container_width=True
            )

        question = ""

        if selected_question != "Select an example question":
            question = selected_question

        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

        if ask_button:

            if not question:
                st.warning("Please select a question.")

            else:

                try:

                    from nl_to_sql import generate_sql
                    from query_runner import run_query

                    with st.spinner("Analyzing..."):

                        sql = generate_sql(question)
                        result = run_query(sql)

                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []

                    answer = format_database_answer(
                        question,
                        result
                    )

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Unable to process your question: {e}"
                    )

    # IMPORTANT: This must stay INSIDE Talk to Data
    if (
        "chat_history" in st.session_state
        and st.session_state.chat_history
    ):

        st.markdown("")
        st.subheader("Insights")

        for chat in st.session_state.chat_history:

            with st.container(border=True):

                st.markdown(
                    f"**{chat['question']}**"
                )

                st.success(
                    chat["answer"]
                )
