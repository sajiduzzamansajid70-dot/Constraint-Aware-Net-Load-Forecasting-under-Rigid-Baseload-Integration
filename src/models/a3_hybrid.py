"""
A3 Hybrid Model (proposal-aligned, no leakage).

Definition:
    Net Load = low-frequency component + residual component
    - Low-frequency: trailing moving average (168-hour window) -> ARIMA(2,1,2)
    - Residual: target - low-frequency -> XGBoost on residual lags + calendar/weather

Leakage controls:
    - Moving averages use trailing windows (no centering)
    - ARIMA fit on training low-frequency only
    - Residual lag features for test built by concatenating train tail with test
      before shifting (ensures first test row sees last train residuals only)
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class A3Hybrid:
    """
    Comparative hybrid model: ARIMA on low-frequency, XGBoost on residuals.
    """

    def __init__(
        self,
        ma_window: int = 168,
        arima_order: tuple = (2, 1, 2),
        residual_lags: Optional[List[int]] = None,
        xgb_max_depth: int = 4,
        xgb_n_estimators: int = 200,
        learning_rate: float = 0.1,
        name: str = "A3_Hybrid",
    ):
        self.ma_window = ma_window
        self.arima_order = arima_order
        self.residual_lags = residual_lags or [1, 2, 3, 6, 12, 24, 48, 168]
        self.xgb_max_depth = xgb_max_depth
        self.xgb_n_estimators = xgb_n_estimators
        self.learning_rate = learning_rate
        self.name = name

        self.arima_model = None
        self.xgb_model = None
        self.scaler = None
        self.feature_columns: List[str] = []
        self.exog_columns: List[str] = []
        self.low_freq_train_last = None
        self.residual_train = None

        self.allowed_exog = [
            # Calendar
            "hour",
            "day_of_week",
            "month",
            "day_of_month",
            "is_peak_hour",
            # Weather and engineered weather lags
            "Temperature",
            "Humidity",
            "Rainfall",
            "Sunshine",
            "temperature_lag12h",
            "temperature_lag24h",
            "humidity_lag12h",
            "humidity_lag24h",
            "heat_stress",
            "heat_stress_lag24h",
        ]

    def _moving_average(self, series: pd.Series) -> pd.Series:
        # Causal (trailing) moving average to avoid look-ahead
        return series.rolling(window=self.ma_window, min_periods=1, center=False).mean()

    def _build_lag_frame(self, residual: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(index=residual.index)
        for lag in self.residual_lags:
            frame[f"residual_lag{lag}"] = residual.shift(lag)
        return frame

    def _select_exog(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.allowed_exog if c in df.columns]
        self.exog_columns = cols
        return df[cols].copy() if cols else pd.DataFrame(index=df.index)

    def fit(self, df_train: pd.DataFrame, target_col: str = "net_load") -> None:
        y_train = df_train[target_col].astype(float).reset_index(drop=True)

        # Low-frequency component and residual
        low_freq_train = self._moving_average(y_train)
        self.low_freq_train_last = low_freq_train.iloc[-1]
        residual_train = y_train - low_freq_train
        self.residual_train = residual_train.reset_index(drop=True)

        # Fit ARIMA on low-frequency signal
        try:
            with catch_warnings():
                simplefilter("ignore")
                self.arima_model = ARIMA(low_freq_train, order=self.arima_order).fit()
            logger.info(f"ARIMA{self.arima_order} fitted on low-frequency component")
        except Exception as exc:
            logger.warning(f"ARIMA fitting failed ({exc}); falling back to naive low-frequency hold.")
            self.arima_model = None

        # Residual regression dataset
        lag_frame = self._build_lag_frame(residual_train)
        max_lag = max(self.residual_lags)
        valid_idx = range(max_lag, len(residual_train))

        lag_features = lag_frame.iloc[valid_idx].reset_index(drop=True)
        exog_train = self._select_exog(df_train).iloc[valid_idx].reset_index(drop=True)

        X_residual = pd.concat([lag_features, exog_train], axis=1)
        y_residual = residual_train.iloc[valid_idx].reset_index(drop=True)

        if X_residual.empty:
            logger.warning("No residual features available; residual model disabled.")
            self.xgb_model = None
            return

        X_residual = X_residual.ffill().bfill().fillna(0)

        try:
            import xgboost as xgb

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_residual)

            self.xgb_model = xgb.XGBRegressor(
                n_estimators=self.xgb_n_estimators,
                max_depth=self.xgb_max_depth,
                learning_rate=self.learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
                tree_method="hist",
            )
            self.xgb_model.fit(X_scaled, y_residual)
            self.feature_columns = list(X_residual.columns)
            logger.info(
                f"Residual XGBoost trained on {X_residual.shape[0]} samples "
                f"with {X_residual.shape[1]} features"
            )
        except Exception as exc:
            logger.warning(f"Residual XGBoost fitting failed ({exc}); using zero residual forecast.")
            self.xgb_model = None
            self.feature_columns = []

    def _prepare_residual_features(
        self,
        residual_full: pd.Series,
        df_test: pd.DataFrame,
        train_length: int,
    ) -> pd.DataFrame:
        """
        Build residual lag features for the test horizon.

        Uses concatenated train+test residuals before shifting so the first
        test sample only sees residuals from the training tail.
        """
        lag_frame_full = self._build_lag_frame(residual_full)
        test_idx = range(train_length, len(residual_full))
        lag_features_test = lag_frame_full.iloc[test_idx].reset_index(drop=True)

        # Align exogenous columns to training set
        if self.exog_columns:
            exog_test = df_test[self.exog_columns].reset_index(drop=True)
        else:
            exog_test = pd.DataFrame(index=lag_features_test.index)

        X_test = pd.concat([lag_features_test, exog_test], axis=1)
        # Reindex to training feature order; unseen columns default to 0
        if self.feature_columns:
            X_test = X_test.reindex(columns=self.feature_columns, fill_value=0)

        return X_test.ffill().bfill().fillna(0)

    def predict(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str = "net_load") -> np.ndarray:
        y_train = df_train[target_col].astype(float).reset_index(drop=True)
        y_test = df_test[target_col].astype(float).reset_index(drop=True)
        horizon = len(y_test)

        # Low-frequency forecast
        if self.arima_model is not None:
            try:
                low_freq_pred = self.arima_model.forecast(steps=horizon).values
            except Exception as exc:
                logger.warning(f"ARIMA forecast failed ({exc}); using last observed low-frequency value.")
                low_freq_pred = np.full(horizon, self.low_freq_train_last)
        else:
            low_freq_pred = np.full(horizon, self.low_freq_train_last)

        # Residual forecast
        if self.xgb_model is not None and self.feature_columns:
            res_seed = self.residual_train.tail(self.ma_window) if self.residual_train is not None else pd.Series([], dtype=float)
            res_hist = list(res_seed.values)
            residual_pred = np.zeros(horizon)

            for t in range(horizon):
                row = {col: 0.0 for col in self.feature_columns}

                for lag in self.residual_lags:
                    key = f"residual_lag{lag}"
                    if key in row:
                        if len(res_hist) >= lag:
                            row[key] = res_hist[-lag]
                        elif res_hist:
                            row[key] = res_hist[0]
                        else:
                            row[key] = 0.0

                if self.exog_columns:
                    for col in self.exog_columns:
                        if col in row and col in df_test.columns:
                            row[col] = float(df_test.iloc[t][col])

                X_row = pd.DataFrame([row], columns=self.feature_columns).fillna(0)
                X_scaled = self.scaler.transform(X_row)
                r_pred = float(self.xgb_model.predict(X_scaled)[0])
                residual_pred[t] = r_pred
                res_hist.append(r_pred)
        else:
            residual_pred = np.zeros(horizon)

        return low_freq_pred + residual_pred

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "ma_window": self.ma_window,
            "arima_order": self.arima_order,
            "residual_lags": self.residual_lags,
            "xgb_max_depth": self.xgb_max_depth,
            "xgb_n_estimators": self.xgb_n_estimators,
            "learning_rate": self.learning_rate,
            "feature_columns": self.feature_columns,
            "exog_columns": self.exog_columns,
            "low_freq_train_last": self.low_freq_train_last,
            "arima_model": self.arima_model,
            "xgb_model": self.xgb_model,
            "scaler": self.scaler,
        }

        with open(model_dir / f"{self.name}.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, model_dir: Path) -> None:
        with open(Path(model_dir) / f"{self.name}.pkl", "rb") as f:
            state = pickle.load(f)

        self.ma_window = state["ma_window"]
        self.arima_order = state["arima_order"]
        self.residual_lags = state["residual_lags"]
        self.xgb_max_depth = state["xgb_max_depth"]
        self.xgb_n_estimators = state["xgb_n_estimators"]
        self.learning_rate = state["learning_rate"]
        self.feature_columns = state["feature_columns"]
        self.exog_columns = state["exog_columns"]
        self.low_freq_train_last = state["low_freq_train_last"]
        self.arima_model = state["arima_model"]
        self.xgb_model = state["xgb_model"]
        self.scaler = state["scaler"]
