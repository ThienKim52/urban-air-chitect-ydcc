"""
AirWise — Module Đánh giá Mô hình
===================================
Đánh giá mô hình hồi quy + phân loại AQI. Vẽ biểu đồ so sánh.

Xem giải thích chi tiết tại: AirWise_Complete_Guide.md → Phần 4
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    f1_score, classification_report, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────
# Quy đổi PM2.5 → 6 mức AQI (US EPA breakpoints)
# ──────────────────────────────────────────────────────────
AQI_LABELS = ["Tốt", "Trung bình", "Kém", "Xấu", "Rất xấu", "Nguy hại"]
AQI_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]


def pm25_to_aqi_level(pm25: np.ndarray) -> np.ndarray:
    """Quy đổi nồng độ PM2.5 (µg/m³) sang 6 mức AQI (0–5)."""
    pm25 = np.asarray(pm25, dtype=float)
    levels = np.zeros_like(pm25, dtype=int)
    levels[pm25 > 12.0] = 1   # Moderate
    levels[pm25 > 35.4] = 2   # Unhealthy for Sensitive Groups
    levels[pm25 > 55.4] = 3   # Unhealthy
    levels[pm25 > 150.4] = 4  # Very Unhealthy
    levels[pm25 > 250.4] = 5  # Hazardous
    return levels


# ──────────────────────────────────────────────────────────
# Đánh giá tổng hợp
# ──────────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Đánh giá mô hình toàn diện: hồi quy + phân loại AQI.
    
    Returns
    -------
    dict : kết quả đánh giá (RMSE, MAE, R², Macro-F1)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # ── Chỉ số Hồi quy ──
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # ── Chỉ số Phân loại (quy đổi sang AQI) ──
    y_true_aqi = pm25_to_aqi_level(y_true)
    y_pred_aqi = pm25_to_aqi_level(y_pred)
    macro_f1 = f1_score(y_true_aqi, y_pred_aqi, average="macro", zero_division=0)
    
    results = {
        "Model": model_name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R²": round(r2, 4),
        "Macro-F1": round(macro_f1, 4),
    }
    
    print(f"\n{'=' * 50}")
    print(f"📊 Đánh giá: {model_name}")
    print(f"{'=' * 50}")
    print(f"  RMSE:     {rmse:.2f} µg/m³")
    print(f"  MAE:      {mae:.2f} µg/m³")
    print(f"  R²:       {r2:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    
    return results


def plot_comparison(y_true, y_pred, model_name, save_path=None):
    """Vẽ biểu đồ so sánh giá trị thực vs dự đoán."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (A) Scatter plot: Thực vs Dự đoán
    axes[0].scatter(y_true, y_pred, alpha=0.1, s=5, color="#4C72B0")
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Hoàn hảo (y = ŷ)")
    axes[0].set_xlabel("PM2.5 thực tế (µg/m³)")
    axes[0].set_ylabel("PM2.5 dự đoán (µg/m³)")
    axes[0].set_title(f"{model_name}: Thực tế vs Dự đoán")
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].legend()
    axes[0].set_aspect("equal")
    
    # (B) Residual plot: Phân phối sai số
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=80, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Sai số (Thực − Dự đoán) µg/m³")
    axes[1].set_ylabel("Số lần")
    axes[1].set_title(f"{model_name}: Phân phối Sai số (Mean={residuals.mean():.1f})")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📄 Biểu đồ lưu tại: {save_path}")
    plt.show()


def plot_confusion_matrix_aqi(y_true, y_pred, model_name, save_path=None):
    """Vẽ Confusion Matrix cho phân loại AQI 6 mức."""
    y_true_aqi = pm25_to_aqi_level(np.asarray(y_true))
    y_pred_aqi = pm25_to_aqi_level(np.asarray(y_pred))
    
    cm = confusion_matrix(y_true_aqi, y_pred_aqi, labels=list(range(6)))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=AQI_LABELS, yticklabels=AQI_LABELS,
    )
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(f"{model_name}: Confusion Matrix (AQI 6 mức)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
