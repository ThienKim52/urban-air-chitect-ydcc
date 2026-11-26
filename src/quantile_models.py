"""
AirWise — Module Quantile Regression
======================================
Huấn luyện XGBoost với Quantile Loss cho khoảng bất định.

Xem giải thích chi tiết tại: AirWise_Complete_Guide.md → Phần 4.3
"""
import numpy as np
from xgboost import XGBRegressor


def train_quantile_xgboost(X_train, y_train, X_val, y_val, alpha=0.05):
    """
    Huấn luyện XGBoost với Quantile Loss.
    
    Parameters
    ----------
    alpha : float
        Quantile level. 0.05 → cận dưới, 0.95 → cận trên, 0.50 → median.
    """
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=42,
        early_stopping_rounds=30,
        verbosity=0,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    print(f"  Quantile α={alpha:.2f}: best_iteration={model.best_iteration}")
    return model


def predict_with_uncertainty(model_point, model_q05, model_q95, X):
    """
    Dự đoán PM2.5 kèm khoảng bất định 90%.
    
    Returns
    -------
    dict with keys: 'prediction', 'lower', 'upper'
    """
    pred = model_point.predict(X)
    lower = model_q05.predict(X)
    upper = model_q95.predict(X)
    
    # Đảm bảo lower ≤ prediction ≤ upper
    lower = np.minimum(lower, pred)
    upper = np.maximum(upper, pred)
    
    return {
        "prediction": pred,
        "lower": lower,
        "upper": upper,
    }
