"""
House Price Prediction - Multiple Linear Regression Training Script
====================================================================
Trains a Multiple Linear Regression model on the Housing Prices Dataset
and exports model coefficients to JSON for client-side web predictions.

Dataset: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ─── Configuration ──────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Housing.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output.json")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── Load Dataset ───────────────────────────────────────────────────────────────
print("=" * 60)
print("  House Price Prediction - Multiple Linear Regression")
print("=" * 60)
print()

print("[1/6] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  - Dataset shape: {df.shape}")
print(f"  - Columns: {list(df.columns)}")
print()

# ─── Exploratory Summary ────────────────────────────────────────────────────────
print("[2/6] Dataset summary:")
print(f"  - Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"  - Mean price:  ${df['price'].mean():,.0f}")
print(f"  - Area range:  {df['area'].min()} - {df['area'].max()} sq ft")
print(f"  - Missing values: {df.isnull().sum().sum()}")
print()

# ─── Preprocessing ──────────────────────────────────────────────────────────────
print("[3/6] Preprocessing features...")

# Store original column stats for the web app (ranges, etc.)
feature_stats = {}

# Binary columns: yes/no → 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
               'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
    feature_stats[col] = {'type': 'binary', 'labels': ['No', 'Yes']}

# Numeric columns: store ranges
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
for col in numeric_cols:
    feature_stats[col] = {
        'type': 'numeric',
        'min': int(df[col].min()),
        'max': int(df[col].max()),
        'mean': round(float(df[col].mean()), 1),
        'median': int(df[col].median())
    }

# One-hot encode furnishingstatus
furnishing_dummies = pd.get_dummies(df['furnishingstatus'], prefix='furnishing', dtype=int)
df = pd.concat([df, furnishing_dummies], axis=1)
df.drop('furnishingstatus', axis=1, inplace=True)

feature_stats['furnishingstatus'] = {
    'type': 'categorical',
    'categories': ['furnished', 'semi-furnished', 'unfurnished'],
    'encoded_columns': list(furnishing_dummies.columns)
}

print(f"  - Binary encoded: {binary_cols}")
print(f"  - One-hot encoded: furnishingstatus -> {list(furnishing_dummies.columns)}")
print()

# ─── Prepare Features & Target ─────────────────────────────────────────────────
print("[4/6] Preparing features and target...")

# Define feature columns (everything except 'price')
feature_columns = [col for col in df.columns if col != 'price']
X = df[feature_columns].values
y = df['price'].values

print(f"  - Feature columns ({len(feature_columns)}): {feature_columns}")
print()

# ─── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Testing samples:  {X_test.shape[0]}")
print()

# ─── Train Multiple Linear Regression ──────────────────────────────────────────
print("[5/6] Training Multiple Linear Regression model...")

model = LinearRegression()
model.fit(X_train, y_train)

# ─── Evaluate Model ─────────────────────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"  - Train R² Score: {train_r2:.4f}")
print(f"  - Test R² Score:  {test_r2:.4f}")
print(f"  - Test RMSE:      ${test_rmse:,.0f}")
print(f"  - Test MAE:       ${test_mae:,.0f}")
print()

# ─── Feature Importance ────────────────────────────────────────────────────────
print("  Feature Importance (Coefficients):")
coeff_importance = []
for name, coeff in zip(feature_columns, model.coef_):
    print(f"    {name:>25s}: {coeff:>12,.2f}")
    coeff_importance.append({
        'name': name,
        'coefficient': round(float(coeff), 4),
        'abs_coefficient': round(abs(float(coeff)), 4)
    })

# Sort by absolute coefficient value
coeff_importance.sort(key=lambda x: x['abs_coefficient'], reverse=True)
print(f"\n  - Intercept: {model.intercept_:,.2f}")
print()

# ─── Export Model to JSON ───────────────────────────────────────────────────────
print("[6/6] Exporting model to JSON...")

model_data = {
    'model_type': 'Multiple Linear Regression',
    'feature_columns': feature_columns,
    'coefficients': {name: round(float(coeff), 4)
                     for name, coeff in zip(feature_columns, model.coef_)},
    'intercept': round(float(model.intercept_), 4),
    'metrics': {
        'train_r2': round(train_r2, 4),
        'test_r2': round(test_r2, 4),
        'test_rmse': round(float(test_rmse), 2),
        'test_mae': round(float(test_mae), 2),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'total_samples': int(len(df))
    },
    'feature_importance': coeff_importance,
    'feature_stats': feature_stats,
    'price_stats': {
        'min': int(df['price'].min()),
        'max': int(df['price'].max()),
        'mean': round(float(df['price'].mean()), 2),
        'median': int(df['price'].median()),
        'std': round(float(df['price'].std()), 2)
    }
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(model_data, f, indent=2)

print(f"  - Model saved to: {OUTPUT_PATH}")
print()
print("=" * 60)
print("  [OK] Training complete! Model exported successfully.")
print("=" * 60)
