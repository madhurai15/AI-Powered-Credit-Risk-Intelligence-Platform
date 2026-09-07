# Credit Risk Platform
AI-powered credit risk platform built using the Home Credit Default Risk dataset for loan default prediction and business risk analysis.

## Key Features
- EDA and business insights
- Data preprocessing and feature engineering
- Random Forest default prediction
- Class imbalance handling
- Risk scoring: Low / Medium / High
- SHAP-based explainability
- Natural-language Talk-to-Data
- MySQL integration
- Business decision rules
- Streamlit UI
- Dockerized deployment

## Dataset
Home Credit Default Risk dataset with **307,511 applicants** and **122 original features**.

Target:
- `0` → Non-default
- `1` → Default
Default rate: **8.07%**

## Machine Learning
Final model: **Random Forest Classifier**
Class imbalance is handled using **stratified splitting** and `class_weight="balanced_subsample"`.

### Performance
| Metric | Score |
| ROC-AUC | 0.732 |
| PR-AUC | 0.210 |
| Default Recall | 0.55 |
| Default F1 | 0.27 |

## Risk Scoring
- `< 30%` → Low Risk
- `30–60%` → Medium Risk
- `≥ 60%` → High Risk

## Explainable AI
SHAP is used to identify the key features contributing to individual risk predictions.

## Talk to Data
Users can ask questions in natural language. The system converts the question into a MySQL `SELECT` query and returns a business-readable answer.

## Decision Rules
Business rules are derived from EDA and model insights, covering credit scores, age, repayment burden, credit history, and risk probability.

## Tech Stack
**Python | Pandas | Scikit-learn | Random Forest | SHAP | MySQL | Hugging Face | Streamlit | Docker**

## Project Structure
credit_risk_platform/
├── data/
├── documents/
├── notebooks/
├── src/
├── sql/
├── models/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

## Configuration
Create a .env file using .env.example and configure the MySQL and Hugging Face credentials.

## Run with Docker
```bash
docker-compose up --build
Open: http://localhost:8501