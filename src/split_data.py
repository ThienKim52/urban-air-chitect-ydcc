"""
AirWise — Module Chia tập Dữ liệu
===================================
Chia dữ liệu chuỗi thời gian thành Train/Val/Test THEO THỨ TỰ THỜI GIAN.

Xem giải thích chi tiết tại: AirWise_Complete_Guide.md → Phần 2.4
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def time_series_split(df: pd.DataFrame, target_col: str = "PM2.5",
                      train_ratio: float = 0.70,
                      val_ratio: float = 0.15):
    """
    Chia dữ liệu chuỗi thời gian thành Train/Val/Test THEO THỨ TỰ THỜI GIAN.
    
    ⚠️ TUYỆT ĐỐI KHÔNG shuffle ngẫu nhiên dữ liệu chuỗi thời gian!
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã tạo features, sắp xếp theo thời gian.
    target_col : str
        Tên cột mục tiêu.
    train_ratio : float
        Tỷ lệ tập Train (mặc định 70%).
    val_ratio : float
        Tỷ lệ tập Validation (mặc định 15%).
    
    Returns
    -------
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
        Tất cả X đã được chuẩn hóa (StandardScaler fit trên Train).
    """
    # Đảm bảo dữ liệu đã sắp xếp theo thời gian
    df = df.sort_index()
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Chia 3 phần LIÊN TIẾP theo thời gian
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print("═══ CHIA TẬP DỮ LIỆU (Time Series Split) ═══\n")
    print(f"  Train: {len(train_df):>7,} dòng  ({train_df.index.min()} → {train_df.index.max()})")
    print(f"  Val:   {len(val_df):>7,} dòng  ({val_df.index.min()} → {val_df.index.max()})")
    print(f"  Test:  {len(test_df):>7,} dòng  ({test_df.index.min()} → {test_df.index.max()})")
    
    # Tách X (features) và y (target)
    feature_cols = [c for c in df.columns if c != target_col]
    
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]
    
    # ──────────────────────────────────────────────────
    # CHUẨN HÓA (StandardScaler)
    # Quy tắc vàng: FIT chỉ trên tập Train, TRANSFORM cả 3 tập
    # ──────────────────────────────────────────────────
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_cols,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    
    print(f"\n  Features ({len(feature_cols)} cột): {feature_cols}")
    print(f"  StandardScaler: fit trên Train, transform cả 3 tập ✓")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
