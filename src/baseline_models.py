"""
Structured baseline models for net load forecasting.

Contains the A1 moving-average + ARIMA baseline and re-exports the canonical
A3 hybrid implementation to avoid duplicate definitions.
"""

import logging
from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from src.models.a3_hybrid import A3Hybrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover - dependency hint
    logger.warning("statsmodels not available. Install with: pip install statsmodels")
    ARIMA = None


class A1_MA_ARIMA:
    """
    Frequency Separation Baseline: Moving Average + ARIMA.

    Trend is estimated with trailing moving averages (center=False to avoid
    look-ahead); ARIMA is fit on the trend only; residuals are assumed zero-mean.
    """

    def __init__(
        self,
        short_window: int = 24,
        long_window: int = 168,
        auto_arima: bool = False,
        arima_order: tuple | None = None,
        name: str = "A1_MA_ARIMA",
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.auto_arima = auto_arima
        self.arima_order = arima_order or (1, 1, 1)
        self.name = name

        self.model_arima = None
        self.trend_train = None

        logger.info(f"Model initialized: {name}")
        logger.info(f"  Short window: {short_window}h (daily trend)")
        logger.info(f"  Long window: {long_window}h (weekly trend)")
        logger.info(f"  Auto-tune ARIMA: {auto_arima}")
        if not auto_arima:
            logger.info(f"  Fixed ARIMA order: {self.arima_order}")

    def _extract_trend(self, y: pd.Series, window: int) -> pd.Series:
        """Causal moving average (center=False) to avoid leakage."""
        return y.rolling(window=window, center=False, min_periods=1).mean()

    def _multi_scale_decompose(self, y: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Blend short/long trends to form the low-frequency component."""
        trend_short = self._extract_trend(y, self.short_window)
        trend_long = self._extract_trend(y, self.long_window)
        trend = 0.7 * trend_short + 0.3 * trend_long
        residual = y - trend
        return trend, residual

    def _auto_tune_arima(
        self, y: pd.Series, max_p: int = 2, max_d: int = 1, max_q: int = 2
    ) -> tuple[int, int, int]:
        """Small grid-search over (p,d,q) using AIC."""
        best_aic = np.inf
        best_order = (1, 1, 1)

        logger.info(
            f"Auto-tuning ARIMA order (grid search p=0-{max_p}, d=0-{max_d}, q=0-{max_q})..."
        )

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        with catch_warnings():
                            simplefilter("ignore")
                            model = ARIMA(y, order=(p, d, q))
                            results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                            logger.info(f"  ARIMA({p},{d},{q}): AIC={results.aic:.1f}")
                    except Exception:
                        continue

        logger.info(f"Best ARIMA order: {best_order} (AIC={best_aic:.1f})")
        return best_order

    def fit(self, y_train: pd.Series) -> None:
        """Fit MA decomposition and ARIMA on training data."""
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)

        logger.info(f"Fitting {self.name} on {len(y_train)} training samples...")
        self.trend_train, residual_train = self._multi_scale_decompose(y_train)

        trend_for_arima = self.trend_train
        if len(trend_for_arima) > 10000:
            subsample_rate = max(1, len(trend_for_arima) // 5000)
            trend_for_arima = trend_for_arima.iloc[::subsample_rate].reset_index(drop=True)
            logger.info(
                f"  Large dataset detected. Subsampling by {subsample_rate}x for ARIMA fitting "
                f"({len(trend_for_arima)} samples)"
            )

        try:
            with catch_warnings():
                simplefilter("ignore")
                order = (
                    self._auto_tune_arima(trend_for_arima)
                    if self.auto_arima
                    else self.arima_order
                )
                self.model_arima = ARIMA(trend_for_arima, order=order).fit()
            logger.info("ARIMA model fitted successfully")
        except Exception as exc:  # pragma: no cover - fallback path
            logger.error(f"Failed to fit ARIMA: {exc}")
            self.model_arima = None

    def predict(self, y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
        """Forecast trend forward; residual assumed zero-mean (no leakage)."""
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test)

        forecast_horizon = len(y_test)
        logger.info(f"Generating forecasts for {forecast_horizon} test samples...")

        if self.trend_train is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")

        if self.model_arima is not None:
            try:
                trend_pred = (
                    self.model_arima.get_forecast(steps=forecast_horizon)
                    .predicted_mean.values[:forecast_horizon]
                )
            except Exception as exc:
                logger.warning(f"ARIMA forecast failed: {exc}. Using last observed trend value.")
                trend_pred = np.full(forecast_horizon, self.trend_train.iloc[-1])
        else:
            logger.info("Using naive trend forecast (ARIMA unavailable)")
            trend_pred = np.full(forecast_horizon, self.trend_train.iloc[-1])

        return trend_pred

    def save(self, model_dir: Path) -> None:
        """Persist ARIMA state to disk."""
        import pickle

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "arima_order": self.arima_order,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "model_arima": self.model_arima,
            "trend_train": self.trend_train,
        }

        model_path = model_dir / f"{self.name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {model_path}")

    def load(self, model_dir: Path) -> None:
        """Load ARIMA state from disk."""
        import pickle

        model_path = Path(model_dir) / f"{self.name}.pkl"
        with open(model_path, "rb") as f:
            state = pickle.load(f)

        self.arima_order = state["arima_order"]
        self.short_window = state["short_window"]
        self.long_window = state["long_window"]
        self.model_arima = state["model_arima"]
        self.trend_train = state["trend_train"]

        logger.info(f"Model loaded from {model_path}")


# Re-export canonical hybrid to avoid duplicate implementations.
A3_Hybrid = A3Hybrid


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    from data_loader import DataLoader
    from features import FeatureEngineer

    loader = DataLoader(Path(__file__).parent.parent / "data")
    elec_df = loader.load_electricity_data()
    weather_df = loader.load_weather_data()

    engineer = FeatureEngineer(rigid_baseload_mw=2200.0)
    _, df_train, df_test, _, _ = engineer.prepare_features(elec_df, weather_df)

    model = A1_MA_ARIMA(short_window=24, long_window=168, auto_arima=True)
    model.fit(df_train["net_load"].values)
    preds = model.predict(df_train["net_load"].values, df_test["net_load"].values)

    print(f"\nA1_MA_ARIMA Test Complete")
    print(f"Predictions shape: {preds.shape}")
    print(f"Test set shape: {df_test.shape}")
