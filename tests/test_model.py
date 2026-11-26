"""
Tests for Model Training, ONNX Export, and ONNX Runtime Inference
"""
import sys
import json
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_models import train_baseline, train_xgboost
from export_onnx import export_to_onnx, verify_onnx


@pytest.fixture
def synthetic_train_data():
    """Provides a small synthetic dataset for rapid model testing."""
    np.random.seed(42)
    n_samples = 150
    n_features = 10

    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    # y is a linear combination + noise
    y_train = (X_train[:, 0] * 3.0 + X_train[:, 1] * 2.0 + np.random.randn(n_samples) * 0.1).astype(np.float32)

    X_val = np.random.randn(40, n_features).astype(np.float32)
    y_val = (X_val[:, 0] * 3.0 + X_val[:, 1] * 2.0 + np.random.randn(40) * 0.1).astype(np.float32)

    return X_train, y_train, X_val, y_val


class TestModelTraining:
    def test_train_baseline_linear_regression(self, synthetic_train_data):
        """Verifies Linear Regression baseline fits properly and predicts."""
        X_train, y_train, X_val, y_val = synthetic_train_data
        model = train_baseline(X_train, y_train, X_val, y_val)

        preds = model.predict(X_val)
        assert len(preds) == len(y_val)
        # R2 should be high for linear synthetic data
        corr = np.corrcoef(y_val, preds)[0, 1]
        assert corr > 0.95

    def test_train_xgboost_lightweight(self, synthetic_train_data):
        """Verifies XGBoost trains without error and produces sensible predictions."""
        X_train, y_train, X_val, y_val = synthetic_train_data
        model = train_xgboost(X_train, y_train, X_val, y_val)

        preds = model.predict(X_val)
        assert len(preds) == len(y_val)
        corr = np.corrcoef(y_val, preds)[0, 1]
        assert corr > 0.90


class TestModelArtifacts:
    def test_feature_columns_metadata_exists(self):
        """Ensures the feature column metadata file exists and defines 29 features."""
        meta_file = Path(__file__).resolve().parent.parent / "models" / "airwise_feature_columns.json"
        if not meta_file.exists():
            pytest.skip("models/airwise_feature_columns.json not yet generated.")

        with open(meta_file, encoding="utf-8") as f:
            features = json.load(f)

        assert isinstance(features, list)
        assert len(features) == 29
        assert "PM2.5_lag_1h" in features
        assert "PM2.5_change_24h" not in features

    def test_onnx_model_inference(self):
        """If ONNX model exists, verifies onnxruntime loads it and infers shape [batch, 1]."""
        onnx_file = Path(__file__).resolve().parent.parent / "models" / "xgboost_pm25_point.onnx"
        if not onnx_file.exists():
            pytest.skip("models/xgboost_pm25_point.onnx not yet exported.")

        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_file))
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]

        assert input_meta.name == "float_input"
        assert input_meta.shape[1] == 29

        # Run dummy inference with 5 samples
        dummy_input = np.random.randn(5, 29).astype(np.float32)
        out = session.run(None, {input_meta.name: dummy_input})[0]

        assert out.shape == (5, 1) or out.shape == (5,)
