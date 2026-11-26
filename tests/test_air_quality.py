"""
Tests for Air Quality AQI Conversion, Evaluation Metrics, and Uncertainty Quantification
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import pm25_to_aqi_level, evaluate_model, AQI_LABELS
from quantile_models import predict_with_uncertainty


class TestAQIConversion:
    def test_us_epa_breakpoints(self):
        """Validates exact breakpoint transitions for the 6 US EPA AQI levels."""
        # Level 0: Tốt (Good) <= 12.0
        assert pm25_to_aqi_level([0.0, 5.5, 12.0]).tolist() == [0, 0, 0]

        # Level 1: Trung bình (Moderate) 12.1 - 35.4
        assert pm25_to_aqi_level([12.1, 20.0, 35.4]).tolist() == [1, 1, 1]

        # Level 2: Kém (Unhealthy for Sensitive) 35.5 - 55.4
        assert pm25_to_aqi_level([35.5, 45.0, 55.4]).tolist() == [2, 2, 2]

        # Level 3: Xấu (Unhealthy) 55.5 - 150.4
        assert pm25_to_aqi_level([55.5, 100.0, 150.4]).tolist() == [3, 3, 3]

        # Level 4: Rất xấu (Very Unhealthy) 150.5 - 250.4
        assert pm25_to_aqi_level([150.5, 200.0, 250.4]).tolist() == [4, 4, 4]

        # Level 5: Nguy hại (Hazardous) > 250.4
        assert pm25_to_aqi_level([250.5, 350.0, 700.0]).tolist() == [5, 5, 5]

    def test_aqi_labels_count(self):
        """Ensures all 6 AQI levels have corresponding labels."""
        assert len(AQI_LABELS) == 6


class TestEvaluationMetrics:
    def test_perfect_prediction_metrics(self):
        """A perfect prediction (y_pred == y_true) must have RMSE=0, MAE=0, R2=1.0, Macro-F1=1.0."""
        y_true = np.array([5.0, 25.0, 45.0, 80.0, 160.0, 300.0])
        y_pred = y_true.copy()

        metrics = evaluate_model(y_true, y_pred, model_name="PerfectModel")

        assert metrics["RMSE"] == 0.0
        assert metrics["MAE"] == 0.0
        assert metrics["R²"] == 1.0
        assert metrics["Macro-F1"] == 1.0

    def test_imperfect_prediction_metrics(self):
        """Checks sanity of RMSE, MAE, R2 calculations."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])  # constant absolute error of 2.0

        metrics = evaluate_model(y_true, y_pred, model_name="TestModel")

        assert np.isclose(metrics["RMSE"], 2.0, atol=1e-2)
        assert np.isclose(metrics["MAE"], 2.0, atol=1e-2)
        assert 0.9 < metrics["R²"] < 1.0


class TestUncertaintyQuantification:
    class MockPointModel:
        def predict(self, X):
            return np.array([50.0, 80.0])

    class MockQ05Model:
        def predict(self, X):
            # Cận dưới (thấp hơn point)
            return np.array([40.0, 65.0])

    class MockQ95Model:
        def predict(self, X):
            # Cận trên (cao hơn point)
            return np.array([65.0, 110.0])

    def test_predict_with_uncertainty_interval(self):
        """Checks that lower <= prediction <= upper holds consistently."""
        m_point = self.MockPointModel()
        m_q05 = self.MockQ05Model()
        m_q95 = self.MockQ95Model()

        res = predict_with_uncertainty(m_point, m_q05, m_q95, np.zeros((2, 5)))

        assert (res["lower"] <= res["prediction"]).all()
        assert (res["prediction"] <= res["upper"]).all()
        assert np.array_equal(res["lower"], np.array([40.0, 65.0]))
        assert np.array_equal(res["upper"], np.array([65.0, 110.0]))

    def test_predict_with_uncertainty_inversion_guard(self):
        """When quantile model produces inverted bounds, the guard must clamp them."""
        m_point = self.MockPointModel()  # [50.0, 80.0]

        class InvertedQ05:
            def predict(self, X):
                return np.array([60.0, 70.0])  # 60 > 50 (inverted)

        class InvertedQ95:
            def predict(self, X):
                return np.array([45.0, 90.0])  # 45 < 50 (inverted)

        res = predict_with_uncertainty(m_point, InvertedQ05(), InvertedQ95(), np.zeros((2, 5)))

        # Must be strictly clamped: lower <= prediction <= upper
        assert (res["lower"] <= res["prediction"]).all()
        assert (res["prediction"] <= res["upper"]).all()
