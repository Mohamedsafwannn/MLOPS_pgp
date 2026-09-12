import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)

import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier

from mlflow_utils import(setup_mlflow,log_run_info,log_model_artifact)

# =========================
# 1. Load Dataset
# =========================

DATA_PATH = "data/processed/train.csv"

train = pd.read_csv(DATA_PATH)

print("Dataset shape:", train.shape)
print(train.head())

# =========================
# 2. Separate Features and Target
# =========================

TARGET = "is_late"

X = train.drop(columns=[TARGET])
y = train[TARGET]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

# =========================
# 3. Remove Leakage and Irrelevant Columns
# =========================

DROP_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "product_id",
    "seller_id",
    "shipping_limit_date",
    "customer_unique_id",
    "seller_delay_rate"
]

X = X.drop(columns=DROP_COLUMNS)

print("\nFeatures after removing unwanted columns:")
print(X.columns.tolist())

print("\nNumber of features:", X.shape[1])

# =========================
# 4. Identify Feature Types
# =========================

NUMERICAL_FEATURES = [
    "order_item_id",
    "price",
    "freight_value",
    "customer_zip_code_prefix",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "seller_zip_code_prefix",
    "purchase_month",
    "is_weekend",
    "estimated_delivery_days",
    "freight_to_price_ratio",
    "same_state",
    "seller_lat",
    "seller_lng",
    "customer_lat",
    "customer_lng",
    "distance_km"
]

CATEGORICAL_FEATURES = [
    "customer_city",
    "customer_state",
    "product_category_name",
    "seller_city",
    "seller_state",
    "purchase_day"
]

print("\nNumerical features:")
print(NUMERICAL_FEATURES)

print("\nCategorical features:")
print(CATEGORICAL_FEATURES)

# Check that every feature has been assigned a type

all_selected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

print("\nNumber of features in X:", X.shape[1])
print("Number of selected features:", len(all_selected_features))

missing_features = set(X.columns) - set(all_selected_features)
extra_features = set(all_selected_features) - set(X.columns)

print("\nFeatures in X but not assigned a type:")
print(missing_features)

print("\nFeatures assigned a type but not present in X:")
print(extra_features)

# =========================
# 5. Preprocessing Pipeline
# =========================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ]
)

print("\nPreprocessing pipeline created successfully.")

# =========================
# 6. Train / Validation Split
# =========================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nValidation target distribution:")
print(y_val.value_counts(normalize=True))

# =========================
# 7. Define Baseline Models
# =========================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

print("\nBaseline models:")
for name in models:
    print("-", name)

# =========================
# 8. Evaluation Function
# =========================

def evaluate_model(model, X_val, y_val):

    y_pred = model.predict(X_val)

    y_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

# =========================
# 9. Train Baseline Models
# =========================

results = []

setup_mlflow()

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(
        pipeline,
        X_val,
        y_val
    )

    print(f"\n{model_name} Results:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    results.append({
        "model": model_name,
        **metrics
    })

# =========================
# 10. Compare Baseline Models
# =========================

results_df = pd.DataFrame(results)

print("\n==============================")
print("BASELINE MODEL COMPARISON")
print("==============================")

print(
    results_df.sort_values(
        by="f1_score",
        ascending=False
    ).to_string(index=False)
)

# print("\nMissing values in training features:")
# print(X_train.isna().sum().sort_values(ascending=False))

# print("\nMissing values in validation features:")
# print(X_val.isna().sum().sort_values(ascending=False))