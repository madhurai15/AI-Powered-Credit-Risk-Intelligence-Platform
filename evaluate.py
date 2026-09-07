from pathlib import Path
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))
from data.loader import prepare_training_data
from data.preprocessor import split_features_target
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.joblib"

# Check model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train.py first.")

# Load data
print("Loading data...")
df = prepare_training_data()
X, y = split_features_target(df)

#train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

# Load trained model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# Predict probabilities
y_probability = model.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= 0.50).astype(int)

# ROC-AUC
roc_auc = roc_auc_score(y_test,y_probability)

# PR-AUC
pr_auc = average_precision_score(y_test,y_probability)

# Results
print("MODEL EVALUATION :")
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test,y_prediction))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_prediction))