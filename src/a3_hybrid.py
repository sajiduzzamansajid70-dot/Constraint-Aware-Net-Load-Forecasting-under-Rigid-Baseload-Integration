"""
A3: Decomposition-Informed Hybrid Model

Combines trend decomposition (ARIMA) with ML forecasting on residual component.
Uses moving average decomposition, ARIMA for low-frequency, and XGBoost for residuals.

This is a comparative baseline to understand:
- Pure trend forecasting (ARIMA only) vs. residual learning
- Value of decomposition-based approach vs. end-to-end ML
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from warnings import catch_warnings, simplefilter
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class A3HybridModel:
    """
    A3: Decomposition-Informed Hybrid Model
    
    Architecture:
        1. Decompose net_load = trend + residual
           - Trend: Multi-scale moving average (captures long-term patterns)
           - Residual: High-frequency noise
        2. Fit ARIMA to trend component
        3. Fit XGBoost to residual component with lagged features
        4. Combine: forecast = trend_pred + residual_pred
    
    Design Rationale:
        - Trend (smooth, predictable) → ARIMA appropriate
        - Residual (noisy, learnable structure) → ML-based approach
        - Specialized training allows each model to focus on its domain
        - Provides interpretable decomposition for analysis
    """
    
    def __init__(self,
                 ma_window: int = 168,           # 7-day trend
                 short_window: int = 24,         # Daily trend
                 arima_order: tuple = (2, 1, 2), # ARIMA(2,1,2)
                 xgb_max_depth: int = 3,         # Shallow to prevent overfitting
                 xgb_n_estimators: int = 50,     # Conservative
                 name: str = "A3_Hybrid"):
        """
        Args:
            ma_window: Long-term MA window (hours)
            short_window: Short-term MA window (hours)
            arima_order: ARIMA (p,d,q) tuple
            xgb_max_depth: Max tree depth for residual model
            xgb_n_estimators: Number of boosting rounds
            name: Model identifier
        """
        self.ma_window = ma_window
        self.short_window = short_window
        self.arima_order = arima_order
        self.xgb_max_depth = xgb_max_depth
        self.xgb_n_estimators = xgb_n_estimators
        self.name = name
        
        self.arima_model = None
        self.xgb_model = None
        self.scaler_residual = None
        self.trend_train = None
        self.residual_feature_cols = None
        
        logger.info(f"A3HybridModel initialized: {name}")
        logger.info(f"  MA window: {ma_window}h, Short window: {short_window}h")
        logger.info(f"  ARIMA order: {arima_order}")
        logger.info(f"  XGBoost: depth={xgb_max_depth}, n_estimators={xgb_n_estimators}")
    
    def _moving_average(self, y: pd.Series, window: int) -> pd.Series:
        """Extract trend via centered moving average."""
        return y.rolling(window=window, center=True, min_periods=1).mean()
    
    def _decompose(self, y: pd.Series) -> tuple:
        """
        Multi-scale decomposition: blend short and long trends.
        
        Trend = 0.7 * short_MA + 0.3 * long_MA (emphasizes recent dynamics)
        Residual = y - trend
        """
        trend_short = self._moving_average(y, self.short_window)
        trend_long = self._moving_average(y, self.ma_window)
        
        trend = 0.7 * trend_short + 0.3 * trend_long
        residual = y - trend
        
        return trend, residual
    
    def fit(self, df_train: pd.DataFrame, feature_cols: list, target_col: str = "net_load"):
        """
        Fit ARIMA on trend and XGBoost on residual with lagged features.
        
        Args:
            df_train: Training DataFrame
            feature_cols: Available feature column names
            target_col: Target variable name (default: "net_load")
            
        Returns:
            Dict with fit statistics
        """
        logger.info(f"Fitting {self.name} on {len(df_train)} training samples...")
        
        y = df_train[target_col].astype(float).values
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Step 1: Decompose
        trend, residual = self._decompose(y)
        self.trend_train = trend
        
        logger.info(f"  Trend: mean={trend.mean():.1f} MW, std={trend.std():.1f} MW")
        logger.info(f"  Residual: mean={residual.mean():.2f} MW, std={residual.std():.1f} MW")
        
        # Step 2: Fit ARIMA on trend
        trend_for_arima = trend
        if len(trend_for_arima) > 10000:
            subsample_rate = max(1, len(trend_for_arima) // 5000)
            trend_for_arima = trend_for_arima.iloc[::subsample_rate].reset_index(drop=True)
            logger.info(f"  Subsampling trend {subsample_rate}x for ARIMA fitting")
        
        try:
            with catch_warnings():
                simplefilter("ignore")
                self.arima_model = ARIMA(trend_for_arima, order=self.arima_order).fit()
            logger.info(f"  ARIMA{self.arima_order} fitted: AIC={self.arima_model.aic:.1f}")
        except Exception as e:
            logger.error(f"  ARIMA fitting failed: {e}. Fallback to naive model.")
            self.arima_model = None
        
        # Step 3: Fit XGBoost on residuals with lagged features
        try:
            import xgboost as xgb
            
            # Build residual dataset with lags
            df_res = pd.DataFrame({'residual': residual.values})
            
            # Add residual lags
            for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
                df_res[f'residual_lag{lag}h'] = df_res['residual'].shift(lag)
            
            # Try to add available features from df_train
            available_features = [
                'hour', 'day_of_week', 'month', 'is_peak_hour',
                'Temperature', 'Humidity', 'Rainfall', 'Sunshine',
                'temperature_lag12h', 'temperature_lag24h',
                'humidity_lag12h', 'humidity_lag24h',
                'heat_stress', 'heat_stress_lag24h',
            ]
            
            for feat in available_features:
                if feat in df_train.columns:
                    df_res[feat] = df_train[feat].values
            
            # Clean NaN from lags
            df_res_clean = df_res.dropna().reset_index(drop=True)
            
            # Select feature columns
            residual_lag_cols = [c for c in df_res_clean.columns if c.startswith('residual_lag')]
            other_feat_cols = [c for c in available_features if c in df_res_clean.columns]
            self.residual_feature_cols = residual_lag_cols + other_feat_cols
            
            if len(self.residual_feature_cols) == 0:
                logger.warning("  No residual features available. Using zero-mean residual.")
                self.xgb_model = None
            else:
                X_res = df_res_clean[self.residual_feature_cols]
                y_res = df_res_clean['residual']
                
                # Scale features
                self.scaler_residual = StandardScaler()
                X_res_scaled = self.scaler_residual.fit_transform(X_res)
                
                # Fit shallow XGBoost
                self.xgb_model = xgb.XGBRegressor(
                    n_estimators=self.xgb_n_estimators,
                    max_depth=self.xgb_max_depth,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                    verbosity=0
                )
                self.xgb_model.fit(X_res_scaled, y_res)
                logger.info(f"  XGBoost (depth={self.xgb_max_depth}) fitted on {len(self.residual_feature_cols)} residual features")
        
        except Exception as e:
            logger.warning(f"  XGBoost residual fitting failed: {e}. Using zero-mean fallback.")
            self.xgb_model = None
        
        return {"residual_features": len(self.residual_feature_cols) if self.residual_feature_cols else 0}
    
    def predict(self, y_train: pd.Series, y_test: pd.Series, df_test: pd.DataFrame = None) -> np.ndarray:
        """
        Generate forecasts by combining trend and residual predictions.
        
        Args:
            y_train: Training net load (for context)
            y_test: Test net load (for residual feature construction)
            df_test: Test DataFrame (optional, for additional features)
            
        Returns:
            Predictions array
        """
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test)
        
        n = len(y_test)
        logger.info(f"Generating forecasts for {n} test samples...")
        
        # Step 1: Forecast trend using ARIMA
        if self.arima_model is not None:
            try:
                trend_pred = self.arima_model.get_forecast(steps=n).predicted_mean.values[:n]
                logger.info(f"  Trend forecast: mean={trend_pred.mean():.1f} MW")
            except Exception as e:
                logger.warning(f"  ARIMA forecast failed: {e}. Using naive fallback.")
                trend_pred = np.full(n, self.trend_train.iloc[-1])
        else:
            logger.info("  Using naive trend forecast (last value)")
            trend_pred = np.full(n, self.trend_train.iloc[-1])
        
        # Step 2: Forecast residuals using XGBoost
        if self.xgb_model is not None and self.residual_feature_cols:
            try:
                # Construct residual features for test period
                y_combined = pd.concat([y_train, y_test], ignore_index=True)
                _, residual_combined = self._decompose(y_combined)
                residual_test = residual_combined.iloc[len(y_train):].reset_index(drop=True)
                
                # Build feature matrix
                df_res_test = pd.DataFrame({'residual': residual_test.values})
                
                for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
                    df_res_test[f'residual_lag{lag}h'] = df_res_test['residual'].shift(lag)
                
                # Add features from df_test if provided
                if df_test is not None:
                    for feat in self.residual_feature_cols:
                        if not feat.startswith('residual_lag') and feat in df_test.columns:
                            df_res_test[feat] = df_test[feat].values
                
                # Fill NaN and select features
                df_res_test = df_res_test.fillna(method='bfill').fillna(0)
                X_res_test = df_res_test[self.residual_feature_cols]
                
                # Ensure correct length
                if len(X_res_test) < n:
                    X_res_test = pd.concat([X_res_test, pd.DataFrame(0, index=range(n - len(X_res_test)), columns=self.residual_feature_cols)])
                X_res_test = X_res_test.iloc[:n]
                
                # Predict
                X_res_test_scaled = self.scaler_residual.transform(X_res_test)
                residual_pred = self.xgb_model.predict(X_res_test_scaled)
                logger.info(f"  Residual forecast: mean={residual_pred.mean():.1f} MW")
            
            except Exception as e:
                logger.warning(f"  XGBoost residual forecast failed: {e}. Using zero-mean.")
                residual_pred = np.zeros(n)
        else:
            logger.info("  Using zero-mean residual forecast")
            residual_pred = np.zeros(n)
        
        # Combine
        predictions = trend_pred + residual_pred
        logger.info(f"Forecasts combined: mean={predictions.mean():.1f} MW")
        
        return predictions
    
    def save(self, model_dir: Path) -> None:
        """Save model to disk."""
        import pickle
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'ma_window': self.ma_window,
            'short_window': self.short_window,
            'arima_order': self.arima_order,
            'xgb_max_depth': self.xgb_max_depth,
            'xgb_n_estimators': self.xgb_n_estimators,
            'arima_model': self.arima_model,
            'xgb_model': self.xgb_model,
            'scaler_residual': self.scaler_residual,
            'residual_feature_cols': self.residual_feature_cols,
        }
        
        with open(model_dir / f"{self.name}.pkl", 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {model_dir / self.name}.pkl")
    
    def load(self, model_dir: Path) -> None:
        """Load model from disk."""
        import pickle
        
        with open(Path(model_dir) / f"{self.name}.pkl", 'rb') as f:
            state = pickle.load(f)
        
        self.ma_window = state['ma_window']
        self.short_window = state['short_window']
        self.arima_order = state['arima_order']
        self.xgb_max_depth = state['xgb_max_depth']
        self.xgb_n_estimators = state['xgb_n_estimators']
        self.arima_model = state['arima_model']
        self.xgb_model = state['xgb_model']
        self.scaler_residual = state['scaler_residual']
        self.residual_feature_cols = state['residual_feature_cols']
        logger.info(f"Model loaded from {Path(model_dir) / self.name}.pkl")
