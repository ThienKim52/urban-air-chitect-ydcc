"""
AirWise — Module Kỹ thuật Đặc trưng (Feature Engineering)
==========================================================
Tạo toàn bộ features cho mô hình dự báo PM2.5.

Xem giải thích chi tiết tại: AirWise_Complete_Guide.md → Phần 2.2–2.3
"""
import pandas as pd
import numpy as np


# Bảng chuyển đổi hướng gió → góc (độ)
WIND_DIR_TO_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def create_features(df: pd.DataFrame, target_col: str = "PM2.5") -> pd.DataFrame:
    """
    Tạo toàn bộ features cho mô hình dự báo PM2.5.
    
    Features được tạo:
    1. Lag features: PM2.5 ở các thời điểm trước (1h, 2h, 3h, 6h, 12h, 24h)
    2. Rolling statistics: mean và std trong cửa sổ 6h, 12h, 24h
    3. Cyclical time encoding: sin/cos cho giờ trong ngày và tháng trong năm
    4. Wind decomposition: phân rã vector gió thành u, v
    5. Các chỉ số thời tiết gốc giữ nguyên (TEMP, PRES, DEWP, RAIN, WSPM)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame với datetime index, đã xử lý missing values.
    target_col : str
        Tên cột mục tiêu.
    
    Returns
    -------
    pd.DataFrame
        DataFrame với các features mới, đã xóa dòng NaN do tạo lag/rolling.
    """
    df = df.copy()
    
    print("═══ FEATURE ENGINEERING ═══\n")
    
    # ──────────────────────────────────────────────────
    # 1. LAG FEATURES
    # ──────────────────────────────────────────────────
    # "PM2.5 giờ trước là bao nhiêu?" → feature mạnh nhất!
    lag_hours = [1, 2, 3, 6, 12, 24]
    for lag in lag_hours:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
    print(f"  ✓ Lag features: {len(lag_hours)} cột (lags: {lag_hours})")
    
    # ──────────────────────────────────────────────────
    # 2. ROLLING STATISTICS
    # ──────────────────────────────────────────────────

    rolling_windows = [6, 12, 24]

    # Only use historical PM2.5 observations.
    # At time t, the latest available observation is t-1.
    past_pm25 = df[target_col].shift(1)

    for window in rolling_windows:

        df[f"{target_col}_roll_mean_{window}h"] = (
            past_pm25
            .rolling(window=window, min_periods=window)
            .mean()
        )

        df[f"{target_col}_roll_std_{window}h"] = (
            past_pm25
            .rolling(window=window, min_periods=window)
            .std()
        )

    print(
        f"  ✓ Rolling features: {len(rolling_windows) * 2} columns"
    )

    # ──────────────────────────────────────────────────
    # 3. CYCLICAL TIME ENCODING
    # ──────────────────────────────────────────────────
    hour = df.index.hour
    month = df.index.month
    
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["day_of_week"] = df.index.dayofweek
    print(f"  ✓ Time features: 5 cột (hour sin/cos, month sin/cos, day_of_week)")
    
    # ──────────────────────────────────────────────────
    # 4. WIND DECOMPOSITION
    # ──────────────────────────────────────────────────
    wind_features_added = 0
    if "wd" in df.columns and "WSPM" in df.columns:
        wind_deg = df["wd"].map(WIND_DIR_TO_DEGREES)
        wind_rad = np.deg2rad(wind_deg)
        
        df["wind_u"] = -df["WSPM"] * np.sin(wind_rad)  # thành phần Đông-Tây
        df["wind_v"] = -df["WSPM"] * np.cos(wind_rad)  # thành phần Bắc-Nam
        wind_features_added = 2
    print(f"  ✓ Wind features: {wind_features_added} cột (u, v decomposition)")
    
    # ──────────────────────────────────────────────────
    # 5. XÓA CÁC CỘT KHÔNG CẦN & DỌN DẸP
    # ──────────────────────────────────────────────────
    cols_to_drop = ["No", "year", "month", "day", "hour", "wd", "station"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    # Xóa các dòng NaN sinh ra do lag/rolling
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    
    print(f"\n  Xóa {dropped} dòng NaN (do lag/rolling cần dữ liệu trước đó)")
    print(f"  Còn lại: {len(df):,} dòng")
    print(f"  Tổng số features: {len(df.columns) - 1}")  # trừ cột target
    print(f"  Danh sách features:\n    {[c for c in df.columns if c != target_col]}")
    
    return df
