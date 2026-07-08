# Member 1 — Feature Engineering | Phase 3: Target Variable Creation

import pandas as pd
import numpy as np

# Load train and test data
train = pd.read_csv(r'data/processed/train.csv')
test  = pd.read_csv(r'data/processed/test.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Date columns to convert
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

# Check missing values in date columns before processing
print("\n=== Missing Values in Date Columns ===")
print(train[date_cols].isna().sum())

# Convert all date columns from string to datetime format
for col in date_cols:
    train[col] = pd.to_datetime(train[col])
    test[col]  = pd.to_datetime(test[col])
print("\n✅ Date columns converted")

# Create target variable — 1 if late, 0 if on time
train['is_late'] = (
    train['order_delivered_customer_date'] > train['order_estimated_delivery_date']
).astype(int)

# Mark rows with missing delivery date as NaN — cant determine if late
missing_delivery = train['order_delivered_customer_date'].isna()
train.loc[missing_delivery, 'is_late'] = np.nan
print("\n=== is_late created ===")
print(train['is_late'].value_counts(dropna=False))

# Check what types of orders were never delivered
print("\n=== Order Status of Undelivered Orders ===")
print(train[missing_delivery]['order_status'].value_counts())

# Drop undelivered orders from train only — test is not touched
before = train.shape[0]
train  = train.dropna(subset=['order_delivered_customer_date'])
after  = train.shape[0]
print(f"\n✅ Removed {before - after} undelivered rows | Remaining: {after}")

# Check class balance — important for model training later
counts = train['is_late'].value_counts()
pct    = train['is_late'].value_counts(normalize=True) * 100
print("\n=== Class Balance ===")
print(f"On time (0): {counts[0]} rows  ({pct[0]:.1f}%)")
print(f"Late    (1): {counts[1]} rows  ({pct[1]:.1f}%)")

# Flag leaky columns — must never be used as model features
leaky_columns = [
    'order_delivered_customer_date',  # used to create is_late
    'order_delivered_carrier_date',   # only known after delivery
    'order_status'                    # reveals the outcome directly
]
print("\n=== LEAKY COLUMNS — NEVER USE AS FEATURES ===")
for col in leaky_columns:
    print(f"  don't use {col}")

# Save updated train with is_late column for other members
train.to_csv('data/processed/train.csv', index=False)
print("\n✅ train.csv saved with is_late column")
print(f"Final shape: {train.shape}")