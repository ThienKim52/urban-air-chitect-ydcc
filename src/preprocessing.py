"""
AirWise — Module Tiền xử lý Dữ liệu
=====================================
Xử lý missing values cho dữ liệu chuỗi thời gian chất lượng không khí.

Xem giải thích chi tiết tại: AirWise_Complete_Guide.md → Phần 2.1
"""
import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame, target_col: str = "PM2.5",
                          max_gap_hours: int = 6) -> pd.DataFrame:
    """
    Xử lý missing values cho dữ liệu chuỗi thời gian chất lượng không khí.
    
    Chiến lược:
    1. Nội suy tuyến tính theo thời gian cho khoảng trống ≤ max_gap_hours
    2. Giá trị vẫn NaN sau nội suy (khoảng trống dài) → giữ nguyên NaN → xóa sau
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame với datetime index, chứa các cột số cần xử lý.
    target_col : str
        Tên cột mục tiêu (PM2.5).
    max_gap_hours : int
        Khoảng trống tối đa (số dòng liên tiếp) cho phép nội suy.
        Mặc định: 6 (tương đương 6 giờ với dữ liệu hourly).
    
    Returns
    -------
    pd.DataFrame
        DataFrame đã xử lý missing values.
    """
    df = df.copy()
    
    # Các cột số cần nội suy
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("═══ XỬ LÝ MISSING VALUES ═══")
    print(f"\nMissing TRƯỚC khi xử lý:")
    missing_before = df[numeric_cols].isnull().sum()
    print(missing_before[missing_before > 0].to_string())
    
    # Nội suy tuyến tính theo thời gian, giới hạn khoảng trống
    for col in numeric_cols:
        df[col] = df[col].interpolate(
            method="time",       # nội suy theo khoảng cách thời gian thực
            limit=max_gap_hours  # chỉ nội suy nếu khoảng trống ≤ max_gap_hours dòng liên tục
        )
    
    print(f"\nMissing SAU khi nội suy (limit={max_gap_hours}h):")
    missing_after = df[numeric_cols].isnull().sum()
    print(missing_after[missing_after > 0].to_string() if missing_after.any() else "  Không còn missing!")
    
    # Xóa các dòng vẫn còn NaN ở cột mục tiêu
    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    print(f"\nXóa {dropped} dòng vẫn còn NaN ở '{target_col}'")
    print(f"Còn lại: {len(df):,} dòng")
    
    return df
