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
                 n_estimators: int = 2000,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 val_fraction: float = 0.2,
                 early_stopping_rounds: int = 50):
        """
        Initialize XGBoost regressor with hyperparameters.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Fraction of samples for each iteration
            colsample_bytree: Fraction of features for each iteration
            random_state: Random seed for reproducibility
            val_fraction: Fraction of training data (chronologically last) reserved for validation
            early_stopping_rounds: Early stopping patience on validation RMSE
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
             verbosity=1,
             early_stopping_rounds=early_stopping_rounds
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.best_iteration = None
        
        logger.info(f"XGBoost model initialized")
        logger.info(f"  n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
    
    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None) -> dict:
        """
        Fit XGBoost model on training data with chronological validation.

        Scaling: Features are standardized using StandardScaler fit on training data only.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features for early stopping (must be chronological tail)
            y_val: Optional validation target

        Returns:
            Dictionary with training and validation metrics
        """
        logger.info("Fitting XGBoost model with chronological validation...")

        # Create a chronological train/validation split if none is provided
        if X_val is None or y_val is None:
            val_size = max(int(len(X_train) * self.val_fraction), 1)
            if val_size >= len(X_train):
                val_size = max(len(X_train) // 5, 1)
            split_idx = len(X_train) - val_size
            X_train_fit = X_train.iloc[:split_idx]
            y_train_fit = y_train.iloc[:split_idx]
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            logger.info(f"  Auto validation split: train={len(X_train_fit)}, val={len(X_val)} (chronological tail)")
        else:
            X_train_fit, y_train_fit = X_train, y_train
            logger.info(f"  Using provided validation set: train={len(X_train_fit)}, val={len(X_val)}")

        # Fit scaler on training split only (no leakage into validation/test)
        X_train_scaled = self.scaler.fit_transform(X_train_fit)
        X_val_scaled = self.scaler.transform(X_val)

        # Train with early stopping on validation RMSE
        self.model.fit(
            X_train_scaled,
            y_train_fit,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )

        self.is_fitted = True
        self.best_iteration = getattr(self.model, "best_iteration", None)
        logger.info("Model fitting complete")
        if self.best_iteration is not None:
            logger.info(f"  Best iteration: {self.best_iteration + 1} (early stopping)")

        # Training and validation metrics at best iteration
        train_pred = self.model.predict(
            X_train_scaled,
            iteration_range=(0, self.best_iteration + 1) if self.best_iteration is not None else None,
        )
        val_pred = self.model.predict(
            X_val_scaled,
            iteration_range=(0, self.best_iteration + 1) if self.best_iteration is not None else None,
        )
        train_rmse = np.sqrt(np.mean((y_train_fit - train_pred) ** 2))
        train_mae = np.mean(np.abs(y_train_fit - train_pred))
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        val_mae = np.mean(np.abs(y_val - val_pred))

        metrics = {
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'val_rmse': float(val_rmse),
            'val_mae': float(val_mae),
            'n_estimators_used': int(self.best_iteration + 1) if self.best_iteration is not None else self.model.n_estimators
        }

        logger.info(f"Training metrics: RMSE={train_rmse:.2f}, MAE={train_mae:.2f}")
        logger.info(f"Validation metrics: RMSE={val_rmse:.2f}, MAE={val_mae:.2f}")

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
        iteration_range = (0, self.best_iteration + 1) if self.best_iteration is not None else None
        return self.model.predict(X_scaled, iteration_range=iteration_range)
    
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
        self.model.get_booster().save_model(str(path / "xgboost_model.json"))

        # Save scaler parameters
        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)
        with open(path / "best_iteration.txt", "w") as f:
            f.write(str(self.best_iteration if self.best_iteration is not None else -1))

        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model and scaler from disk."""
        path = Path(path)
        
        # Load model (simplified for XGBoost 3.x)
        self.model.load_model(str(path / "xgboost_model.json"))
        
        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")
        best_iter_path = path / "best_iteration.txt"
        if best_iter_path.exists():
            try:
                loaded_iter = int(best_iter_path.read_text().strip())
                self.best_iteration = loaded_iter if loaded_iter >= 0 else None
            except Exception:
                self.best_iteration = getattr(self.model, "best_iteration", None)
        else:
            self.best_iteration = getattr(self.model, "best_iteration", None)
        
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
