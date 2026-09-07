from pathlib import Path
import sys

import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from data.loader import prepare_training_data
from data.preprocessor import split_features_target, create_preprocessor


# --------------------------------------------------
# Model path
# --------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "credit_risk_model.joblib"


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading training data...")

df = prepare_training_data()

print("Dataset shape:", df.shape)


# --------------------------------------------------
# Split features and target
# --------------------------------------------------

X, y = split_features_target(df)

print("Features:", X.shape)
print("Target:", y.shape)


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

print("\nCreating preprocessing pipeline...")

preprocessor = create_preprocessor(X_train)


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# Complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# Train model
# --------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully.")
print("Location:", MODEL_PATH)