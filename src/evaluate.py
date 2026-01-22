"""
Evaluation module for constraint-aware net load forecasting.

Peak-hour risk-focused evaluation (from proposal 7.3):
- Full-horizon metrics (MAE, RMSE) are SECONDARY
- Peak-hour metrics (18:00-22:00) are PRIMARY
- Additional: under-forecast rate, error distributions

Strictly aligned with proposal section 7.3
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluation for net load forecasting.
    
    From proposal 7.3:
    "Evaluation will use two layers:
    1. Full-horizon metrics: MAE, RMSE
    2. Peak-hour metrics (PRIMARY): MAE, RMSE during 18:00-22:00"
    
    Additional reporting:
    - Error distributions during peak hours
    - Under-forecast rate (frequency of negative errors beyond thresholds)
    """
    
    def __init__(self, peak_hours: list = None):
        """
        Args:
            peak_hours: List of hours (0-23) considered as peak.
                       Default: [18, 19, 20, 21, 22] (18:00-22:59)
        """
        if peak_hours is None:
            peak_hours = [18, 19, 20, 21, 22]
        
        self.peak_hours = peak_hours
        logger.info(f"Evaluator initialized with peak hours: {peak_hours}")
    
    def evaluate_full_horizon(self,
                             y_true: pd.Series,
                             y_pred: np.ndarray) -> dict:
        """
        Compute full-horizon metrics (secondary).
        
        Metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute Percentage Error
        
        Args:
            y_true: True values
            y_pred: Predictions
            
        Returns:
            Dictionary with metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE only if all values are non-zero
        if np.all(y_true != 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true))
        else:
            mape = np.nan
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape) if not np.isnan(mape) else None,
            'mean_true': float(np.mean(y_true)),
            'std_true': float(np.std(y_true))
        }
        
        return metrics
    
    def evaluate_peak_hours(self,
                           y_true: pd.Series,
                           y_pred: np.ndarray,
                           hours: pd.Series) -> dict:
        """
        Compute peak-hour metrics (PRIMARY).
        
        From proposal: "The peak-hour window is selected because it aligns with 
        the most operationally constrained period and the highest risk of imbalance."
        
        Peak hours: 18:00-22:00 (18:59)
        
        Args:
            y_true: True values
            y_pred: Predictions
            hours: Hour of day (0-23)
            
        Returns:
            Dictionary with peak-hour metrics
        """
        peak_mask = hours.isin(self.peak_hours)
        
        y_true_peak = np.asarray(y_true[peak_mask])
        y_pred_peak = np.asarray(y_pred[peak_mask])
        
        mae = np.mean(np.abs(y_true_peak - y_pred_peak))
        rmse = np.sqrt(np.mean((y_true_peak - y_pred_peak) ** 2))
        
        # Error definition: error = y_pred - y_true
        # Under-forecast: error < 0 (predicted too low)
        errors = y_pred_peak - y_true_peak
        threshold_levels = [0, 250, 500]
        under_forecast_rate = (errors < 0).mean()
        under_forecast_rates = {
            f"under_forecast_rate_{thr}mw": float((errors < -thr).mean())
            for thr in threshold_levels
        }

        # Large under-forecast events (e.g., > 500 MW underprediction)
        large_under_forecast_threshold = 500  # MW
        large_under_forecast_rate = (errors < -large_under_forecast_threshold).mean()
        
        metrics = {
            'peak_mae': float(mae),
            'peak_rmse': float(rmse),
            'peak_count': int(peak_mask.sum()),
            'peak_mean_true': float(np.mean(y_true_peak)),
            'peak_std_true': float(np.std(y_true_peak)),
            'under_forecast_rate': float(under_forecast_rate),
            'large_under_forecast_rate': float(large_under_forecast_rate),
            'under_forecast_rates': under_forecast_rates,
            'max_under_forecast': float(np.min(errors)),  # Most negative error (most severe underprediction)
            'max_over_forecast': float(np.max(errors))    # Most positive error (most severe overprediction)
        }
        
        return metrics
    
    def evaluate_by_season(self,
                          y_true: pd.Series,
                          y_pred: np.ndarray,
                          months: pd.Series,
                          hours: pd.Series) -> dict:
        """
        Seasonal performance computed on PEAK HOURS only (18:00–22:00).
        """
        peak_mask = hours.isin(self.peak_hours)

        y_true_peak = np.asarray(y_true[peak_mask])
        y_pred_peak = np.asarray(y_pred[peak_mask])
        months_peak = months[peak_mask].reset_index(drop=True)

        seasons = {
            'Winter (Dec-Feb)': [12, 1, 2],
            'Spring (Mar-May)': [3, 4, 5],
            'Summer (Jun-Aug)': [6, 7, 8],
            'Fall (Sep-Nov)': [9, 10, 11]
        }

        season_metrics = {}
        for season_name, season_months in seasons.items():
            season_mask = months_peak.isin(season_months)

            y_true_season = y_true_peak[season_mask.values]
            y_pred_season = y_pred_peak[season_mask.values]

            if len(y_true_season) == 0:
                continue

            mae = float(np.mean(np.abs(y_true_season - y_pred_season)))
            rmse = float(np.sqrt(np.mean((y_true_season - y_pred_season) ** 2)))

            season_metrics[season_name] = {
                'mae': mae,
                'rmse': rmse,
                'count': int(season_mask.sum())
            }

        return season_metrics
    
    def evaluate_full(self,
                     df_test: pd.DataFrame,
                     y_true: pd.Series,
                     y_pred: np.ndarray,
                     target_col: str = 'net_load') -> dict:
        """
        Full evaluation pipeline.
        
        Args:
            df_test: Test DataFrame with hour and month columns
            y_true: True target values
            y_pred: Model predictions
            target_col: Name of target column
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Evaluating model...")
        
        # Full horizon (secondary)
        full_metrics = self.evaluate_full_horizon(y_true, y_pred)
        
        # Peak hours (PRIMARY)
        peak_metrics = self.evaluate_peak_hours(y_true, y_pred, df_test['hour'])
        
        # Seasonal peak-hour metrics
        seasonal_metrics = self.evaluate_by_season(y_true, y_pred, df_test['month'], df_test['hour'])
        
        results = {
            'full_horizon': full_metrics,
            'peak_hours': peak_metrics,
            'seasonal_peak_hours': seasonal_metrics
        }
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        
        logger.info("\nFull Horizon (SECONDARY):")
        logger.info(f"  MAE: {full_metrics['mae']:.2f} MW")
        logger.info(f"  RMSE: {full_metrics['rmse']:.2f} MW")
        
        logger.info("\nPeak Hours 18:00-22:00 (PRIMARY):")
        logger.info(f"  MAE: {peak_metrics['peak_mae']:.2f} MW")
        logger.info(f"  RMSE: {peak_metrics['peak_rmse']:.2f} MW")
        logger.info(f"  Under-forecast rate (<0 MW error): {peak_metrics['under_forecast_rate']*100:.2f}%")
        uf_rates = peak_metrics.get('under_forecast_rates', {})
        for thr_label, rate in uf_rates.items():
            logger.info(f"  {thr_label.replace('_', ' ').title()}: {rate*100:.2f}%")
        logger.info(f"  Large under-forecast rate (<-500MW): {peak_metrics['large_under_forecast_rate']*100:.2f}%")
        logger.info(f"  Most negative error (worst under-forecast): {peak_metrics['max_under_forecast']:.2f} MW")
        logger.info(f"  Most positive error (worst over-forecast): {peak_metrics['max_over_forecast']:.2f} MW")
        
        logger.info("\nSeasonal Peak-Hour Performance (18:00-22:00):")
        for season, metrics in seasonal_metrics.items():
            logger.info(f"  {season}: MAE={metrics['mae']:.2f} MW, RMSE={metrics['rmse']:.2f} MW ({metrics['count']} peak hours)")
        
        return results
    
    def plot_results(self,
                    df_test: pd.DataFrame,
                    y_true: pd.Series,
                    y_pred: np.ndarray,
                    output_dir: Path = None):
        """
        Generate diagnostic plots.
        
        Plots:
        1. Time series: true vs predicted
        2. Residuals by hour
        3. Error distribution (peak vs non-peak)
        4. Quantile-quantile plot
        
        Args:
            df_test: Test DataFrame with datetime
            y_true: True values
            y_pred: Predictions
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = Path("outputs/plots")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        errors = np.asarray(y_true) - y_pred
        
        # Plot 1: Time series
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df_test['datetime'], y_true, label='True', linewidth=1, alpha=0.7)
        ax.plot(df_test['datetime'], y_pred, label='Predicted', linewidth=1, alpha=0.7)
        
        # Highlight peak hours
        peak_mask = df_test['hour'].isin(self.peak_hours)
        ax.scatter(df_test.loc[peak_mask, 'datetime'], y_true[peak_mask], 
                   label='True (Peak)', alpha=0.3, s=10, color='red')
        
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Net Load (MW)')
        ax.set_title('Constraint-Aware Net Load: True vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'timeseries.png', dpi=150)
        logger.info(f"Saved: timeseries.png")
        plt.close()
        
        # Plot 2: Residuals by hour
        fig, ax = plt.subplots(figsize=(12, 5))
        hours_range = range(24)
        residuals_by_hour = [errors[df_test['hour'] == h] for h in hours_range]
        
        bp = ax.boxplot(residuals_by_hour, labels=hours_range, patch_artist=True)
        
        # Color peak hours
        for i, hour in enumerate(hours_range):
            if hour in self.peak_hours:
                bp['boxes'][i].set_facecolor('lightcoral')
            else:
                bp['boxes'][i].set_facecolor('lightblue')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Prediction Error (MW)')
        ax.set_title('Prediction Error by Hour (Red=Peak Hours)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'errors_by_hour.png', dpi=150)
        logger.info(f"Saved: errors_by_hour.png")
        plt.close()
        
        # Plot 3: Error distribution (peak vs non-peak)
        fig, ax = plt.subplots(figsize=(10, 5))
        peak_mask = df_test['hour'].isin(self.peak_hours)
        
        ax.hist(errors[~peak_mask], bins=50, alpha=0.6, label='Non-Peak', color='blue')
        ax.hist(errors[peak_mask], bins=50, alpha=0.6, label='Peak Hours', color='red')
        
        ax.set_xlabel('Prediction Error (MW)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution: Peak vs Non-Peak Hours')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=150)
        logger.info(f"Saved: error_distribution.png")
        plt.close()
        
        # Plot 4: Q-Q plot (residuals vs normal)
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 8))
        stats.probplot(errors, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot: Residuals vs Normal Distribution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'qq_plot.png', dpi=150)
        logger.info(f"Saved: qq_plot.png")
        plt.close()
        
        logger.info(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    # Test evaluation
    from features import FeatureEngineer
    from train import XGBoostModel
    from data_loader import DataLoader
    
    loader = DataLoader(Path("constraint_aware_net_load/data"))
    elec_df = loader.load_electricity_data()
    weather_df = loader.load_weather_data()
    
    engineer = FeatureEngineer(rigid_baseload_mw=2200.0)
    df, df_train, df_test, feature_cols, target_col = engineer.prepare_features(elec_df, weather_df)
    
    # Train model
    model = XGBoostModel(n_estimators=100, max_depth=6)
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = Evaluator(peak_hours=[18, 19, 20, 21, 22])
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    y_pred = model.predict(X_test)
    
    results = evaluator.evaluate_full(df_test, y_test, y_pred, target_col)
    
    # Save results
    output_dir = Path("constraint_aware_net_load/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to outputs/results.json")
