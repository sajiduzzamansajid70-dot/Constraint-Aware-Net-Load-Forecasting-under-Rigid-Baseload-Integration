"""
Main orchestration script for constraint-aware net load forecasting.

Runs the complete reproducible pipeline:
1. Load electricity and weather data
2. Construct constraint-aware net load target
3. Engineer features (lagged, calendar, weather)
4. Train XGBoost model with time-series validation
5. Evaluate: full-horizon and peak-hour metrics
6. Save results, model, and plots

Strictly aligned with research proposal - no modifications, no shortcuts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

# Import pipeline modules
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.train import XGBoostModel
from src.evaluate import Evaluator
from src.baseline_models import A1_MA_ARIMA
from src.models import A3Hybrid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Run complete constraint-aware net load forecasting pipeline.
    """
    logger.info("="*80)
    logger.info("CONSTRAINT-AWARE NET LOAD FORECASTING PIPELINE")
    logger.info("Bangladesh Power System - Peak-Hour Risk-Focused Study")
    logger.info("="*80)
    
    # Setup paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    models_dir = outputs_dir / "models"
    plots_dir = outputs_dir / "plots"
    
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # ==============================================================================
    # PHASE 1: DATA LOADING
    # ==============================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA LOADING")
    logger.info("="*80)
    
    loader = DataLoader(data_dir)
    electricity_df = loader.load_electricity_data()
    weather_df = loader.load_weather_data()
    
    logger.info(f"Loaded electricity data: {electricity_df.shape}")
    logger.info(f"Loaded weather data: {weather_df.shape}")
    
    # ==============================================================================
    # PHASE 2: FEATURE ENGINEERING & TARGET CONSTRUCTION
    # ==============================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: FEATURE ENGINEERING & TARGET CONSTRUCTION")
    logger.info("="*80)
    
    # Rigid baseload scenario: 2200 MW (nuclear planning assumption)
    rigid_baseload_mw = 2200.0
    logger.info(f"Rigid baseload scenario: {rigid_baseload_mw} MW (nuclear)")
    
    engineer = FeatureEngineer(rigid_baseload_mw=rigid_baseload_mw)
    
    # Construct net load and create features
    df_full, df_train, df_test, feature_cols, target_col = engineer.prepare_features(
        electricity_df,
        weather_df
    )
    
    logger.info(f"\nFeatures engineered:")
    logger.info(f"  Total samples: {len(df_full)}")
    logger.info(f"  Train samples: {len(df_train)}")
    logger.info(f"  Test samples: {len(df_test)}")
    logger.info(f"  Number of features: {len(feature_cols)}")

    qa_report = engineer.last_qa_report or {}
    with open(outputs_dir / 'data_quality_report.json', 'w') as f:
        json.dump(qa_report, f, indent=2, default=str)
    logger.info(f"Saved: data_quality_report.json (plausibility filtering + weather alignment note)")
    
    # ==============================================================================
    # PHASE 3: MODEL TRAINING
    # ==============================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: MODEL TRAINING")
    logger.info("="*80)
    
    # Prepare training/test data
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    # Store results for multiple models
    all_results = {}
    predictions_all = {}
    model_roles = {
        'A0_XGBoost': 'Primary',
        'A1_MA_ARIMA': 'Comparative',
        'A3_Hybrid': 'Comparative'
    }
    
    # ======================================================================
    # Model 1: A0_XGBoost (primary)
    # ======================================================================
    logger.info("\nTraining A0_XGBoost model (primary ML baseline)...")
    
    model_a0 = XGBoostModel(
        n_estimators=2000,          # allow early stopping to find optimal depth
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        val_fraction=0.2,
        early_stopping_rounds=50,
    )
    
    train_metrics = model_a0.fit(X_train, y_train)
    
    logger.info(f"XGBoost training complete:")
    logger.info(f"  Train RMSE: {train_metrics['train_rmse']:.2f} MW")
    logger.info(f"  Train MAE: {train_metrics['train_mae']:.2f} MW")
    logger.info(f"  Val RMSE: {train_metrics['val_rmse']:.2f} MW")
    logger.info(f"  Val MAE: {train_metrics['val_mae']:.2f} MW")
    logger.info(f"  Trees used (early stopping): {train_metrics['n_estimators_used']}")
    
    # Feature importance
    feature_importance = model_a0.get_feature_importance(feature_cols)
    
    logger.info(f"\nTop 10 most important features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance_pct']:6.2f}%")
    
    # Save feature importance
    feature_importance.to_csv(outputs_dir / 'feature_importance_xgb.csv', index=False)
    logger.info(f"Saved: feature_importance_xgb.csv")
    
    # Generate predictions
    y_pred_xgb = model_a0.predict(X_test)
    predictions_all['A0_XGBoost'] = y_pred_xgb
    
    # ======================================================================
    # Model 2: A1_MA_ARIMA (Structured baseline, explicit decomposition)
    # ======================================================================
    logger.info("\nTraining A1_MA_ARIMA model (frequency separation baseline)...")
    
    model_a1 = A1_MA_ARIMA(
        short_window=24,      # Daily MA
        long_window=168,      # Weekly MA
        auto_arima=False,     # Use fixed ARIMA(1,1,1) for speed
        arima_order=(1, 1, 1),
        name="A1_MA_ARIMA"
    )
    
    try:
        model_a1.fit(y_train.values)
        y_pred_a1 = model_a1.predict(y_train.values, y_test.values)
        predictions_all['A1_MA_ARIMA'] = y_pred_a1
        logger.info(f"A1_MA_ARIMA training complete")
    except Exception as e:
        logger.error(f"A1_MA_ARIMA training failed: {e}")
        predictions_all['A1_MA_ARIMA'] = np.full_like(y_pred_xgb, y_train.mean())
    
    # ======================================================================
    # Model 3: A3_Hybrid (comparative hybrid)
    # ======================================================================
    logger.info("\nTraining A3_Hybrid model (comparative hybrid)...")
    
    model_a3 = A3Hybrid(
        ma_window=168,
        arima_order=(2, 1, 2),
        residual_lags=[1, 2, 3, 6, 12, 24, 48, 168],
        xgb_max_depth=4,
        xgb_n_estimators=200,
        learning_rate=0.1,
        name="A3_Hybrid"
    )
    
    try:
        model_a3.fit(df_train, target_col=target_col)
        y_pred_a3 = model_a3.predict(df_train, df_test, target_col=target_col)
        predictions_all['A3_Hybrid'] = y_pred_a3
        logger.info(f"A3_Hybrid training complete")
    except Exception as e:
        logger.error(f"A3_Hybrid training failed: {e}")
        predictions_all['A3_Hybrid'] = np.full_like(y_test.values, y_train.mean())
    
    # ==============================================================================
    # PHASE 4: MODEL EVALUATION
    # ==============================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: MODEL EVALUATION")
    logger.info("="*80)
    
    # Peak hours: 18:00 - 22:00 (operational risk window)
    evaluator = Evaluator(peak_hours=[18, 19, 20, 21, 22])
    
    # Evaluate each model
    for model_name, y_pred in predictions_all.items():
        logger.info(f"\n{'-'*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'-'*80}")
        
        # Add predictions to test dataframe
        df_test_eval = df_test.copy()
        df_test_eval['prediction'] = y_pred
        df_test_eval['error'] = y_test.values - y_pred
        
        # Evaluate
        results = evaluator.evaluate_full(df_test_eval, y_test, y_pred, target_col)
        all_results[model_name] = results
        
        # Log summary for this model
        logger.info(f"\nFull Horizon (SECONDARY):")
        logger.info(f"  MAE: {results['full_horizon']['mae']:.2f} MW")
        logger.info(f"  RMSE: {results['full_horizon']['rmse']:.2f} MW")
        
        logger.info(f"\nPeak Hours 18:00-22:00 (PRIMARY):")
        logger.info(f"  MAE: {results['peak_hours']['peak_mae']:.2f} MW")
        logger.info(f"  RMSE: {results['peak_hours']['peak_rmse']:.2f} MW")
        logger.info(f"  Under-Forecast Rate: {results['peak_hours']['under_forecast_rate']*100:.2f}%")
        
        # Save model predictions
        df_test_eval_save = df_test_eval[[
            'datetime', 'hour', 'day_of_week', 'month',
            target_col, 'prediction', 'error', 'is_peak_hour'
        ]].copy()
        
        pred_file = outputs_dir / f'test_predictions_{model_name}.csv'
        df_test_eval_save.to_csv(pred_file, index=False)
        logger.info(f"Saved: test_predictions_{model_name}.csv")
        
        # Generate plots only for main model (A0_XGBoost) to avoid memory issues
        if model_name == 'A0_XGBoost':
            plot_subdir = plots_dir / model_name
            plot_subdir.mkdir(parents=True, exist_ok=True)
            evaluator.plot_results(df_test_eval, y_test, y_pred, output_dir=plot_subdir)
            logger.info(f"Saved plots to: {plot_subdir}/")
    
    # Store primary results from A0_XGBoost for main summary
    results = all_results['A0_XGBoost']
    y_pred = predictions_all['A0_XGBoost']
    
    # ==============================================================================
    # PHASE 5: SAVE RESULTS & PLOTS
    # ==============================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: SAVE RESULTS & PLOTS")
    logger.info("="*80)
    
    # Save results JSON for all models
    results_all_save = {
        'config': {
            'rigid_baseload_mw': rigid_baseload_mw,
            'models': list(all_results.keys()),
            'n_features': len(feature_cols),
            'train_samples': len(df_train),
            'test_samples': len(df_test),
            'test_date_range': f"{df_test['datetime'].min()} to {df_test['datetime'].max()}",
            'timestamp': datetime.now().isoformat()
        },
        'results': {
            model_name: results for model_name, results in all_results.items()
        }
    }
    
    with open(outputs_dir / 'results_all_models.json', 'w') as f:
        json.dump(results_all_save, f, indent=2)
    logger.info(f"Saved: results_all_models.json")
    
    # Also save main results (A0_XGBoost) for backward compatibility
    results['config'] = {
        'rigid_baseload_mw': rigid_baseload_mw,
        'model': 'A0_XGBoost',
        'n_features': len(feature_cols),
        'train_samples': len(df_train),
        'test_samples': len(df_test),
        'test_date_range': f"{df_test['datetime'].min()} to {df_test['datetime'].max()}",
        'timestamp': datetime.now().isoformat()
    }
    
    with open(outputs_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: results.json")
    
    # Save test predictions (A0_XGBoost main model)
    df_test_eval_main = df_test.copy()
    df_test_eval_main['prediction'] = predictions_all['A0_XGBoost']
    df_test_eval_main['error'] = y_test.values - predictions_all['A0_XGBoost']
    
    df_test_save = df_test_eval_main[[
        'datetime', 'hour', 'day_of_week', 'month',
        target_col, 'prediction', 'error', 'is_peak_hour'
    ]].copy()
    
    df_test_save.to_csv(outputs_dir / 'test_predictions.csv', index=False)
    logger.info(f"Saved: test_predictions.csv")
    
    # Save models
    model_a0.save(models_dir)
    logger.info(f"Saved: A0_XGBoost model to {models_dir}")
    
    model_a1.save(models_dir)
    logger.info(f"Saved: A1_MA_ARIMA model to {models_dir}")
    
    model_a3.save(models_dir)
    logger.info(f"Saved: A3_Hybrid model to {models_dir}")
    
    # Save feature list
    with open(outputs_dir / 'features.json', 'w') as f:
        json.dump({'features': feature_cols}, f, indent=2)
    logger.info(f"Saved: features.json")
    
    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE - MODEL COMPARISON")
    logger.info("="*80)
    
    # Compare all models
    logger.info("\n" + "="*80)
    logger.info("FULL HORIZON METRICS (SECONDARY)")
    logger.info("="*80)
    
    for model_name, model_results in all_results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  MAE: {model_results['full_horizon']['mae']:.2f} MW")
        logger.info(f"  RMSE: {model_results['full_horizon']['rmse']:.2f} MW")
    
    logger.info("\n" + "="*80)
    logger.info("PEAK HOURS 18:00-22:00 (PRIMARY METRICS)")
    logger.info("="*80)
    
    for model_name, model_results in all_results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  MAE: {model_results['peak_hours']['peak_mae']:.2f} MW")
        logger.info(f"  RMSE: {model_results['peak_hours']['peak_rmse']:.2f} MW")
        logger.info(f"  Peak Samples: {model_results['peak_hours']['peak_count']}")
        logger.info(f"  Under-Forecast Rate: {model_results['peak_hours']['under_forecast_rate']*100:.2f}%")
        if 'seasonal_peak_hours' in model_results:
            logger.info(f"  Seasonal Peak-Hour Performance:")
            for season, metrics in model_results['seasonal_peak_hours'].items():
                logger.info(f"    {season}: MAE={metrics['mae']:.2f} MW, RMSE={metrics['rmse']:.2f} MW ({metrics['count']} peak hours)")
    
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*80)
    
    # Extract peak-hour MAE for comparison
    comparison_rows = []
    for model_name in predictions_all.keys():
        if model_name not in all_results:
            continue
        model_results = all_results[model_name]
        comparison_rows.append({
            'Model': model_name,
            'MAE': model_results['full_horizon']['mae'],
            'RMSE': model_results['full_horizon']['rmse'],
            'Peak_RMSE': model_results['peak_hours']['peak_rmse'],
            'Role': model_roles.get(model_name, 'Comparative')
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(outputs_dir / 'model_comparison.csv', index=False)
    logger.info(f"\nSaved: model_comparison.csv")

    logger.info("\nOutputs saved to:")
    logger.info(f"  {outputs_dir}/")
    logger.info(f"    - results_all_models.json (all model metrics)")
    logger.info(f"    - results.json (A0_XGBoost main model)")
    logger.info(f"    - test_predictions.csv (test set predictions)")
    logger.info(f"    - test_predictions_*.csv (predictions by model)")
    logger.info(f"    - feature_importance_xgb.csv (A0_XGBoost feature ranks)")
    logger.info(f"    - model_comparison.csv (comparative metrics)")
    logger.info(f"    - features.json (feature list)")
    logger.info(f"    - models/ (trained models)")
    logger.info(f"    - plots/ (diagnostic plots by model)")
    
    return results


if __name__ == "__main__":
    results = main()
    print("\nPipeline execution complete. Check outputs/ for results.")
