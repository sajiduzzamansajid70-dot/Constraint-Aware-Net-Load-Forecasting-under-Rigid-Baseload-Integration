"""
Training module for constraint-aware net load forecasting.

XGBoost primary model (from proposal 7.2):
- Chosen for scalability, robustness, interpretability
- Time-series aware validation (no shuffling, chronological splits)
- Strict leakage prevention

Strictly aligned with proposal section 7.2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost regression model for net load forecasting.
    
    Model choice (from proposal 7.2):
    "XGBoost (or equivalent gradient boosting) trained directly on constraint-aware net load
     chosen for scalability, robustness, and interpretability through feature importance 
     and partial dependence"
    
    Training strategy:
    - Time-series validation respecting chronology
    - Hyperparameter tuning on training data only
    - No data leakage
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        """
        Initialize XGBoost regressor with hyperparameters.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Fraction of samples for each iteration
            colsample_bytree: Fraction of features for each iteration
            random_state: Random seed for reproducibility
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective='reg:squarederror',
            eval_metric='rmse',
            n_jobs=-1,
            verbosity=1
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"XGBoost model initialized")
        logger.info(f"  n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
    
    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None) -> dict:
        """
        Fit XGBoost model on training data.
        
        Scaling: Features are standardized using StandardScaler fit on training data only.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features for early stopping
            y_val: Optional validation target
            
        Returns:
            Dictionary with training metrics and history
        """
        logger.info("Fitting XGBoost model...")
        
        # Fit scaler on training data only (no leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model (note: XGBoost 3.x simplified the early stopping interface)
        # For compatibility with multiple versions, we fit without early stopping on test data
        self.model.fit(
            X_train_scaled, 
            y_train,
            verbose=True
        )
        
        self.is_fitted = True
        logger.info("Model fitting complete")
        
        # Training metrics
        train_rmse = np.sqrt(np.mean((y_train - self.model.predict(X_train_scaled)) ** 2))
        train_mae = np.mean(np.abs(y_train - self.model.predict(X_train_scaled)))
        
        metrics = {
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'n_estimators_used': self.model.n_estimators
        }
        
        logger.info(f"Training metrics: RMSE={train_rmse:.2f}, MAE={train_mae:.2f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Extract feature importance for interpretability.
        
        Args:
            feature_names: List of feature column names
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'importance_pct': 100.0 * importance / importance.sum()
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path: Path):
        """Save model and scaler to disk (reloadable)."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model in native XGBoost format
        self.model.save_model(str(path / "xgboost_model.json"))

        # Save scaler parameters
        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)

        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model and scaler from disk."""
        path = Path(path)
        
        # Load model (simplified for XGBoost 3.x)
        self.model.load_model(str(path / "xgboost_model.json"))
        
        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")
        
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")


class TimeSeriesValidator:
    """
    Time-series validation strategy.
    
    From proposal 7.2:
    "Model tuning will be performed using time-series validation that respects chronology."
    
    Strategy: Rolling window or expanding window validation
    - Never shuffle data
    - Never use test data from past to train on future
    """
    
    @staticmethod
    def expanding_window_split(df: pd.DataFrame,
                               min_train_size: int = 52*24,  # ~1 week
                               step_size: int = 30*24):  # ~1 month step
        """
        Expanding window: train grows over time, test moves forward.
        
        Args:
            df: Full dataset sorted by time
            min_train_size: Minimum training samples
            step_size: Samples to advance test window each iteration
            
        Yields:
            Tuples of (train_idx, val_idx) respecting chronology
        """
        n_samples = len(df)
        
        for i in range(0, n_samples - min_train_size - step_size, step_size):
            train_end = i + min_train_size
            val_end = train_end + step_size
            
            if val_end > n_samples:
                val_end = n_samples
            
            yield (slice(0, train_end), slice(train_end, val_end))
    
    @staticmethod
    def rolling_window_split(df: pd.DataFrame,
                            train_size: int = 365*24,  # 1 year
                            val_size: int = 30*24):  # 1 month
        """
        Rolling window: fixed-size train/val windows that move forward.
        
        Args:
            df: Full dataset sorted by time
            train_size: Number of training samples
            val_size: Number of validation samples
            
        Yields:
            Tuples of (train_idx, val_idx) respecting chronology
        """
        n_samples = len(df)
        window_size = train_size + val_size
        
        for i in range(0, n_samples - window_size, val_size):
            train_end = i + train_size
            val_end = train_end + val_size
            
            yield (slice(i, train_end), slice(train_end, val_end))


if __name__ == "__main__":
    # Test model and training
    from features import FeatureEngineer
    from data_loader import DataLoader
    
    loader = DataLoader(Path("constraint_aware_net_load/data"))
    elec_df = loader.load_electricity_data()
    weather_df = loader.load_weather_data()
    
    engineer = FeatureEngineer(rigid_baseload_mw=2200.0)
    df, df_train, df_test, feature_cols, target_col = engineer.prepare_features(elec_df, weather_df)
    
    # Initialize and fit model
    model = XGBoostModel(n_estimators=100, max_depth=6)
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    
    metrics = model.fit(X_train, y_train)
    
    # Feature importance
    importance = model.get_feature_importance(feature_cols)
    print("\n" + "="*80)
    print("TOP 10 IMPORTANT FEATURES")
    print("="*80)
    print(importance.head(10))
    
    # Test predictions
    X_test = df_test[feature_cols]
    y_pred = model.predict(X_test)
    
    print(f"\nPredictions shape: {y_pred.shape}")
    print(f"First 10 predictions: {y_pred[:10]}")
