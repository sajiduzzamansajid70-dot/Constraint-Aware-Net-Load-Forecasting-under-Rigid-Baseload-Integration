"""
Quick test of both models on the full dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.train import XGBoostModel
from src.baseline_models import A1_MA_ARIMA
from src.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
project_root = Path(__file__).parent
data_dir = project_root / "data"
outputs_dir = project_root / "outputs"
models_dir = outputs_dir / "models"
plots_dir = outputs_dir / "plots"

outputs_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

# Load and prepare data
logger.info("Loading data...")
loader = DataLoader(data_dir)
electricity_df = loader.load_electricity_data()
weather_df = loader.load_weather_data()

logger.info("Engineering features...")
engineer = FeatureEngineer(rigid_baseload_mw=2200.0)
df_full, df_train, df_test, feature_cols, target_col = engineer.prepare_features(
    electricity_df,
    weather_df
)

X_train = df_train[feature_cols]
y_train = df_train[target_col]
X_test = df_test[feature_cols]
y_test = df_test[target_col]

results_all = {}

# ========== Model 1: XGBoost ==========
logger.info("\n" + "="*80)
logger.info("MODEL 1: XGBoost (ML-based baseline)")
logger.info("="*80)

model_xgb = XGBoostModel(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

train_metrics = model_xgb.fit(X_train, y_train)
logger.info(f"Training complete: RMSE={train_metrics['train_rmse']:.2f}, MAE={train_metrics['train_mae']:.2f}")

y_pred_xgb = model_xgb.predict(X_test)

# ========== Model 2: A1_MA_ARIMA ==========
logger.info("\n" + "="*80)
logger.info("MODEL 2: A1_MA_ARIMA (Frequency separation baseline)")
logger.info("="*80)

model_a1 = A1_MA_ARIMA(
    short_window=24,
    long_window=168,
    auto_arima=False,
    arima_order=(1, 1, 1),
    name="A1_MA_ARIMA"
)

try:
    model_a1.fit(y_train.values)
    y_pred_a1 = model_a1.predict(y_train.values, y_test.values)
    logger.info(f"A1_MA_ARIMA training complete")
except Exception as e:
    logger.error(f"A1_MA_ARIMA failed: {e}")
    y_pred_a1 = np.full_like(y_pred_xgb, y_train.mean())

# ========== Evaluate Models ==========
logger.info("\n" + "="*80)
logger.info("EVALUATION")
logger.info("="*80)

evaluator = Evaluator(peak_hours=[18, 19, 20, 21, 22])

for model_name, y_pred in [("XGBoost", y_pred_xgb), ("A1_MA_ARIMA", y_pred_a1)]:
    logger.info(f"\nEvaluating: {model_name}")
    
    df_test_eval = df_test.copy()
    df_test_eval['prediction'] = y_pred
    df_test_eval['error'] = y_test.values - y_pred
    
    results = evaluator.evaluate_full(df_test_eval, y_test, y_pred, target_col)
    results_all[model_name] = results
    
    logger.info(f"Full Horizon MAE: {results['full_horizon']['mae']:.2f} MW")
    logger.info(f"Peak Hours MAE: {results['peak_hours']['peak_mae']:.2f} MW")
    logger.info(f"Peak Hours RMSE: {results['peak_hours']['peak_rmse']:.2f} MW")
    logger.info(f"Under-Forecast Rate: {results['peak_hours']['under_forecast_rate']*100:.2f}%")

# ========== Model Comparison ==========
logger.info("\n" + "="*80)
logger.info("MODEL COMPARISON")
logger.info("="*80)

comparison_data = []
for model_name, model_results in results_all.items():
    comparison_data.append({
        'Model': model_name,
        'Peak_MAE_MW': model_results['peak_hours']['peak_mae'],
        'Peak_RMSE_MW': model_results['peak_hours']['peak_rmse'],
        'Full_MAE_MW': model_results['full_horizon']['mae'],
        'Full_RMSE_MW': model_results['full_horizon']['rmse'],
        'Under_Forecast_Rate': model_results['peak_hours']['under_forecast_rate'] * 100
    })

comparison_df = pd.DataFrame(comparison_data)
logger.info("\n" + comparison_df.to_string(index=False))

comparison_df.to_csv(outputs_dir / 'model_comparison.csv', index=False)
logger.info(f"\nComparison saved to: {outputs_dir / 'model_comparison.csv'}")

logger.info("\n✓ Test complete")
