#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

sns.set_theme(style="whitegrid")


# In[107]:


DATA_DIR = Path("../data")

APPLICATION_PATH = DATA_DIR / "application_train.csv"
DESCRIPTION_PATH = DATA_DIR / "HomeCredit_columns_description.csv"


# In[108]:


print("Application file exists:", APPLICATION_PATH.exists())
print("Description file exists:", DESCRIPTION_PATH.exists())


# In[109]:


application = pd.read_csv(APPLICATION_PATH)
column_description = pd.read_csv(
    DESCRIPTION_PATH,
    encoding="latin1"
)

print("Application dataset loaded successfully!")
print("Shape:", application.shape)


# In[110]:


column_description.head()


# In[111]:


application.head()


# In[112]:


application.sample(5, random_state=42)


# In[113]:


application.info()


# In[114]:


print("Number of rows:", application.shape[0])
print("Number of columns:", application.shape[1])


# In[115]:


application.describe().T


# In[116]:


application.describe(include="object").T


# In[117]:


numeric_features = application.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = application.select_dtypes(
    include=["object"]
).columns.tolist()

print("Number of numerical features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))


# In[118]:


feature_categories = {
    "Applicant_Demographics": [
        "CODE_GENDER",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "NAME_FAMILY_STATUS",
        "NAME_EDUCATION_TYPE",
        "NAME_HOUSING_TYPE"
    ],

    "Financial_Profile": [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE"
    ],

    "Employment": [
        "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "DAYS_EMPLOYED"
    ],

    "Credit_Risk_Scores": [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY"
    ],

    "Contact_and_Document": [
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "FLAG_WORK_PHONE",
        "FLAG_DOCUMENT_3"
    ]
}

for category, features in feature_categories.items():
    print(f"\n{category}")
    print("-" * 40)
    print(features)


# In[119]:


missing_values = pd.DataFrame({
    "Missing_Count": application.isnull().sum(),
    "Missing_Percentage": application.isnull().mean() * 100
})

missing_values = missing_values.sort_values(
    "Missing_Percentage",
    ascending=False
)

missing_values.head(20)


# In[120]:


top_missing = missing_values.head(20)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_missing.reset_index(),
    x="Missing_Percentage",
    y="index"
)

plt.title("Top 20 Features by Missing Percentage")
plt.xlabel("Missing Percentage (%)")
plt.ylabel("Feature")

plt.show()


# In[121]:


duplicate_count = application.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# In[122]:


unique_values = pd.DataFrame({
    "Feature": application.columns,
    "Unique_Values": application.nunique().values
})

unique_values.sort_values(
    "Unique_Values"
).head(20)


# In[154]:


feature_summary = pd.DataFrame({
    "Feature": application.columns,
    "Data_Type": application.dtypes.astype(str),
    "Unique_Values": application.nunique(),
    "Missing_Count": application.isnull().sum(),
    "Missing_Percentage": application.isnull().mean() * 100
})

feature_summary.head(20)


# In[123]:


target_counts = application["TARGET"].value_counts()

print(target_counts)


# In[124]:


target_summary = pd.DataFrame({
    "Count": application["TARGET"].value_counts(),
    "Percentage": (
        application["TARGET"]
        .value_counts(normalize=True)
        .mul(100)
    )
})

target_summary


# In[125]:


plt.figure(figsize=(7, 5))

sns.countplot(
    data=application,
    x="TARGET"
)

plt.title("Loan Default Distribution")
plt.xlabel("TARGET (0 = Non-Default, 1 = Default)")
plt.ylabel("Number of Applicants")

plt.show()


# In[126]:


important_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED"
]

application[important_features].describe().T


# In[127]:


application["AGE_YEARS"] = (
    -application["DAYS_BIRTH"] / 365.25
)

application[["DAYS_BIRTH", "AGE_YEARS"]].head()


# In[128]:


plt.figure(figsize=(9, 5))

sns.histplot(
    data=application,
    x="AGE_YEARS",
    bins=30
)

plt.title("Applicant Age Distribution")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Applicants")

plt.show()


# In[129]:


plt.figure(figsize=(9, 5))

income_99 = application["AMT_INCOME_TOTAL"].quantile(0.99)

sns.histplot(
    data=application[
        application["AMT_INCOME_TOTAL"] <= income_99
    ],
    x="AMT_INCOME_TOTAL",
    bins=50
)

plt.title("Applicant Income Distribution (Up to 99th Percentile)")
plt.xlabel("Annual Income")
plt.ylabel("Number of Applicants")

plt.show()


# In[130]:


plt.figure(figsize=(9, 5))

sns.histplot(
    data=application,
    x="AMT_CREDIT",
    bins=50
)

plt.title("Credit Amount Distribution")
plt.xlabel("Credit Amount")
plt.ylabel("Number of Applicants")

plt.show()


# In[131]:


application["INCOME_GROUP"] = pd.qcut(
    application["AMT_INCOME_TOTAL"],
    q=5,
    duplicates="drop"
)

income_default = (
    application
    .groupby("INCOME_GROUP", observed=True)["TARGET"]
    .mean()
    .mul(100)
    .reset_index(name="Default_Rate")
)

income_default


# In[132]:


plt.figure(figsize=(10, 5))

sns.barplot(
    data=income_default,
    x="INCOME_GROUP",
    y="Default_Rate"
)

plt.title("Default Rate by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Default Rate (%)")

plt.xticks(rotation=30)

plt.show()


# In[133]:


education_default = (
    application
    .groupby("NAME_EDUCATION_TYPE")["TARGET"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index(name="Default_Rate")
)

education_default


# In[134]:


plt.figure(figsize=(10, 6))

sns.barplot(
    data=education_default,
    x="Default_Rate",
    y="NAME_EDUCATION_TYPE"
)

plt.title("Default Rate by Education Type")
plt.xlabel("Default Rate (%)")
plt.ylabel("Education Type")

plt.show()


# In[135]:


income_type_default = (
    application
    .groupby("NAME_INCOME_TYPE")
    .agg(
        Applicants=("TARGET", "size"),
        Defaults=("TARGET", "sum"),
        Default_Rate=("TARGET", "mean")
    )
    .reset_index()
)

income_type_default["Default_Rate"] = (
    income_type_default["Default_Rate"] * 100
)

income_type_default = income_type_default.sort_values(
    "Default_Rate",
    ascending=False
)

income_type_default


# In[136]:


plt.figure(figsize=(10, 6))

sns.barplot(
    data=income_type_default,
    x="Default_Rate",
    y="NAME_INCOME_TYPE"
)

plt.title("Default Rate by Income Type")
plt.xlabel("Default Rate (%)")
plt.ylabel("Income Type")

plt.show()


# In[137]:


family_default = (
    application
    .groupby("NAME_FAMILY_STATUS")["TARGET"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index(name="Default_Rate")
)

family_default


# In[138]:


plt.figure(figsize=(10, 6))

sns.barplot(
    data=family_default,
    x="Default_Rate",
    y="NAME_FAMILY_STATUS"
)

plt.title("Default Rate by Family Status")
plt.xlabel("Default Rate (%)")
plt.ylabel("Family Status")

plt.show()


# In[139]:


gender_default = (
    application
    .groupby("CODE_GENDER")["TARGET"]
    .mean()
    .mul(100)
    .reset_index(name="Default_Rate")
)

gender_default


# In[140]:


plt.figure(figsize=(7, 5))

sns.barplot(
    data=gender_default,
    x="CODE_GENDER",
    y="Default_Rate"
)

plt.title("Default Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Default Rate (%)")

plt.show()


# In[141]:


application["AGE_GROUP"] = pd.cut(
    application["AGE_YEARS"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]
)

age_default = (
    application
    .groupby("AGE_GROUP", observed=True)["TARGET"]
    .mean()
    .mul(100)
    .reset_index(name="Default_Rate")
)

age_default


# In[143]:


application["CREDIT_INCOME_RATIO"] = (
    application["AMT_CREDIT"] /
    application["AMT_INCOME_TOTAL"]
)

application["CREDIT_INCOME_RATIO"].describe()


# In[144]:


application["CREDIT_INCOME_GROUP"] = pd.qcut(
    application["CREDIT_INCOME_RATIO"],
    q=5,
    duplicates="drop"
)

credit_income_default = (
    application
    .groupby("CREDIT_INCOME_GROUP", observed=True)["TARGET"]
    .mean()
    .mul(100)
    .reset_index(name="Default_Rate")
)

credit_income_default


# In[145]:


plt.figure(figsize=(10, 5))

sns.barplot(
    data=credit_income_default,
    x="CREDIT_INCOME_GROUP",
    y="Default_Rate"
)

plt.title("Default Rate by Credit-to-Income Ratio")
plt.xlabel("Credit-to-Income Ratio Group")
plt.ylabel("Default Rate (%)")

plt.xticks(rotation=30)

plt.show()


# In[146]:


application["ANNUITY_INCOME_RATIO"] = (
    application["AMT_ANNUITY"] /
    application["AMT_INCOME_TOTAL"]
)


# In[147]:


application["ANNUITY_INCOME_GROUP"] = pd.qcut(
    application["ANNUITY_INCOME_RATIO"],
    q=5,
    duplicates="drop"
)

annuity_income_default = (
    application
    .groupby("ANNUITY_INCOME_GROUP", observed=True)["TARGET"]
    .mean()
    .mul(100)
    .reset_index(name="Default_Rate")
)

annuity_income_default


# In[148]:


plt.figure(figsize=(10, 5))

sns.barplot(
    data=annuity_income_default,
    x="ANNUITY_INCOME_GROUP",
    y="Default_Rate"
)

plt.title("Default Rate by Annuity-to-Income Ratio")
plt.xlabel("Annuity-to-Income Ratio Group")
plt.ylabel("Default Rate (%)")

plt.xticks(rotation=30)

plt.show()


# In[149]:


application["EXT_SOURCE_2_GROUP"] = pd.qcut(
    application["EXT_SOURCE_2"],
    q=5,
    duplicates="drop"
)

ext_source_default = (
    application
    .groupby("EXT_SOURCE_2_GROUP", observed=True)["TARGET"]
    .mean()
    .mul(100)
    .reset_index(name="Default_Rate")
)

ext_source_default


# In[150]:


plt.figure(figsize=(10, 5))

sns.barplot(
    data=ext_source_default,
    x="EXT_SOURCE_2_GROUP",
    y="Default_Rate"
)

plt.title("Default Rate by External Credit Score")
plt.xlabel("EXT_SOURCE_2 Group")
plt.ylabel("Default Rate (%)")

plt.xticks(rotation=30)

plt.show()


# ## Business Insights
# 
# ### 1. External Credit Score
# Applicants in the lowest EXT_SOURCE_2 group have an observed default rate of 15.21%, compared with 3.59% in the highest group. This indicates a strong negative relationship between external credit score and default risk.
# 
# ### 2. Age and Default Risk
# Younger applicants show higher observed default rates. The 18–25 age group has a default rate of 12.29%, compared with 5.22% among applicants aged 56+, indicating age-related differences in observed credit risk.
# 
# ### 3. Education and Default Risk
# Default rates vary substantially across education levels. Applicants with lower secondary education have a default rate of 10.93%, while those with an academic degree have a rate of 1.83%.
# 
# ### 4. Annuity-to-Income Ratio
# Applicants with higher annuity-to-income ratios generally show higher default rates. The default rate increases from 7.20% in the lowest group to 8.70% in the fourth group, suggesting that greater repayment burden relative to income is associated with elevated credit risk.
# 
# ### 5. Family Status
# Default rates differ across family-status groups. Civil marriage and single/not-married applicants have relatively higher observed default rates of 9.94% and 9.81%, respectively, while the rate for widowed applicants is 5.82%.
