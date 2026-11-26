"""
AirWise — Module Xuất ONNX
============================
Chuyển đổi mô hình XGBoost sang ONNX và kiểm chứng.

Xem giải thích chi tiết tại: AirWise_Complete_Guide.md → Phần 5
"""
import os
import numpy as np


def export_to_onnx(
    model,
    n_features,
    save_path="models/xgboost_pm25.onnx"
):
    """
    Convert an XGBoost model to ONNX.

    Temporarily renames the Booster features to f0, f1, ...
    because the onnxmltools converter expects indexed feature names.
    Original feature names are restored after conversion.
    """

    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    # Create output directory
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define ONNX input:
    # None = flexible batch size
    # n_features = fixed number of input features
    initial_type = [
        ("float_input", FloatTensorType([None, n_features]))
    ]

    # Get the underlying XGBoost Booster
    booster = model.get_booster()

    # Remember original feature names
    original_feature_names = booster.feature_names

    try:
        # Temporarily rename features to f0, f1, ... f29
        if original_feature_names is not None:

            if len(original_feature_names) != n_features:
                raise ValueError(
                    "Number of feature names does not match "
                    "n_features."
                )

            booster.feature_names = [
                f"f{i}" for i in range(n_features)
            ]

        # Convert XGBoost to ONNX
        onnx_model = onnxmltools.convert_xgboost(
            model,
            initial_types=initial_type
        )

    finally:
        # Restore original names even if conversion fails
        if original_feature_names is not None:
            booster.feature_names = original_feature_names

    # Save ONNX model
    onnxmltools.utils.save_model(onnx_model, save_path)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)

    print(
        f"  ONNX saved: {save_path} "
        f"({size_mb:.2f} MB)"
    )

    return save_path



def verify_onnx(original_model, onnx_path, X_sample, atol=1e-3, rtol=1e-3):
    """
    Kiểm chứng: output ONNX ≈ output mô hình gốc.
    
    Parameters
    ----------
    original_model : XGBRegressor
    onnx_path : str
    X_sample : np.ndarray
        Mẫu dữ liệu kiểm tra (nên dùng ~100 dòng từ tập test).
    atol : float
        Ngưỡng sai số tuyệt đối (mặc định 1e-3 = 0.001 µg/m³).
    rtol : float
        Ngưỡng sai số tương đối (mặc định 1e-3 = 0.1%).
    """
    import onnxruntime as ort
    
    # 1. Ép kiểu float32 đồng nhất cho cả 2 để tránh sai lệch do kiểu dữ liệu đầu vào
    X_f32 = np.asarray(X_sample, dtype=np.float32)
    
    # Dự đoán bằng mô hình gốc (với input float32)
    y_original = original_model.predict(X_f32)
    
    # Dự đoán bằng ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    y_onnx = session.run(
        None, {input_name: X_f32}
    )[0].flatten()
    
    # So sánh
    abs_diff = np.abs(y_original - y_onnx)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    # Kiểm tra theo chuẩn số học allclose: |a - b| <= (atol + rtol * |b|)
    is_close = np.allclose(y_original, y_onnx, rtol=rtol, atol=atol)
    
    print(f"\n  🔍 Kiểm chứng ONNX vs Original:")
    print(f"     Max abs diff:  {max_diff:.2e}")
    print(f"     Mean abs diff: {mean_diff:.2e}")
    
    if is_close:
        print("     ✅ PASS — Sai lệch chấp nhận được (nằm trong dung sai float32)")
    else:
        print("     ⚠️  WARN — Sai lệch lớn, cần kiểm tra")
    
    return max_diff
