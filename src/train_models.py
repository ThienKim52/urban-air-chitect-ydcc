"""
AirWise — Module Huấn luyện Mô hình
=====================================
Huấn luyện 3 mô hình: Linear Regression, Random Forest, XGBoost.

Xem giải thích chi tiết tại: AirWise_Complete_Guide.md → Phần 3
"""
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_baseline(X_train, y_train, X_val, y_val):
    """Huấn luyện Linear Regression (Baseline)."""
    print("\n" + "=" * 60)
    print("🔹 MODEL 1: Linear Regression (Baseline)")
    print("=" * 60)
    
    model = LinearRegression()
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)
    
    print(f"  Train RMSE: {train_rmse:.2f} µg/m³")
    print(f"  Val   RMSE: {val_rmse:.2f} µg/m³")
    print(f"  Val   R²:   {val_r2:.4f}")
    print(f"  Thời gian:  {train_time:.2f}s")
    
    return model


def train_random_forest(X_train, y_train, X_val, y_val):
    """Huấn luyện Random Forest."""
    print("\n" + "=" * 60)
    print("🔹 MODEL 2: Random Forest")
    print("=" * 60)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)
    
    print(f"  Train RMSE: {train_rmse:.2f} µg/m³")
    print(f"  Val   RMSE: {val_rmse:.2f} µg/m³")
    print(f"  Val   R²:   {val_r2:.4f}")
    print(f"  Thời gian:  {train_time:.2f}s")
    
    # Feature importance
    importances = dict(zip(X_train.columns, model.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top-5 features: {[(k, f'{v:.3f}') for k, v in top5]}")
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Huấn luyện XGBoost — mô hình chính của AirWise."""
    print("\n" + "=" * 60)
    print("🔹 MODEL 3: XGBoost (Mô hình chính)")
    print("=" * 60)
    
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
        verbosity=0,
    )
    
    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - start
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)
    
    print(f"  Train RMSE: {train_rmse:.2f} µg/m³")
    print(f"  Val   RMSE: {val_rmse:.2f} µg/m³")
    print(f"  Val   R²:   {val_r2:.4f}")
    print(f"  Best iter:  {model.best_iteration}")
    print(f"  Thời gian:  {train_time:.2f}s")
    
    # Feature importance
    importances = dict(zip(X_train.columns, model.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top-5 features: {[(k, f'{v:.3f}') for k, v in top5]}")
    
    return model
