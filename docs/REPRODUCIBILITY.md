# Reproducibility Guide

This document provides detailed instructions to ensure reproducible execution of the constraint-aware net load forecasting pipeline.

## Environment Setup

### 1. Python Version
```bash
python --version
# Should output: Python 3.9 or higher
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n net_load python=3.10
conda activate net_load
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- xgboost >= 1.5.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- openpyxl >= 3.6.0

### 4. Verify Installation
```bash
python -c "import xgboost as xgb; print(xgb.__version__)"
python -c "import pandas as pd; print(pd.__version__)"
```

---

## Data Validation

### 1. Check Data Files Exist
```bash
# From project root, verify paths:
ls -la ../pgcb+hourly+generation+dataset+(bangladesh)/PGCB_date_power_demand.xlsx
ls -la "../High Volume Real-World Weather Data/High Volume Real-World Weather Data/Weather Datasets/Combined Data/BD_weather.csv"
```

### 2. Data Integrity Checks
The pipeline validates:
- **Electricity data:** 92,650 hourly records with columns: datetime, generation_mw, demand_mw, load_shedding, renewable outputs
- **Weather data:** 543,839 daily records from 35 stations with: Temperature, Humidity, Rainfall, Sunshine

Run diagnostic:
```bash
python -m src.data_loader
# Should complete without errors and display data shape
```

### 3. Expected Data Ranges
**Electricity (PGCB):**
- Demand: 3,000 - 13,000 MW (typical Bangladesh)
- Solar generation: 0 - 400 MW
- Wind generation: 0 - 50 MW
- Load shedding: 0 - 2,000 MW (during shortages)

**Weather:**
- Temperature: 15 - 35 °C (Bangladesh climate)
- Humidity: 40 - 90 %
- Rainfall: 0 - 50 mm/day (monsoon dependent)
- Sunshine: 0 - 12 hours

---

## Reproducible Execution

### 1. Set Random Seeds
All seeds are hardcoded in code for reproducibility:
- **XGBoost:** `random_state=42` (in train.py)
- **NumPy:** Can be set via environment if needed

### 2. Single-Run Execution
```bash
cd /path/to/constraint_aware_net_load
python main.py
```

**Execution time:** ~5-10 minutes (depending on CPU)

**Expected output:**
- Console logs detailing each phase
- Results saved to `outputs/`

### 3. Verify Output Structure
After execution, check:
```bash
ls outputs/
# Should contain: results.json, test_predictions.csv, feature_importance.csv, features.json, models/, plots/

ls outputs/plots/
# Should contain: timeseries.png, errors_by_hour.png, error_distribution.png, qq_plot.png
```

---

## Expected Results

### Key Metrics (Test Set)
These are approximate ranges based on historical data patterns:

**Full Horizon (Secondary):**
- MAE: 400 - 700 MW
- RMSE: 600 - 1000 MW

**Peak Hours 18:00-22:00 (PRIMARY):**
- MAE: 350 - 600 MW (should be better than full horizon)
- RMSE: 500 - 900 MW
- Under-forecast rate: 35 - 50%
- Large under-forecast rate (>500 MW): 5 - 15%

### Feature Importance (Top 10)
Expected high-importance features:
1. `net_load_lag1h` (previous hour)
2. `net_load_lag24h` (same hour previous day)
3. `net_load_lag168h` (same hour previous week)
4. `hour` (time of day)
5. `temperature` / `temperature_lag24h`
6. `net_load_lag12h`, `net_load_lag6h`
7. Calendar features (day_of_week, month)
8. Weather interactions (heat_stress, humidity)

---

## Reproducibility Checklist

- [ ] Python version 3.9+
- [ ] All dependencies installed from requirements.txt
- [ ] Data files exist and are accessible
- [ ] Random seeds verified (hardcoded as 42)
- [ ] No manual modifications to data or code
- [ ] Execution from project root: `python main.py`
- [ ] All outputs generated in `outputs/`
- [ ] Metrics within expected ranges
- [ ] No warnings or errors in logs

---

## Troubleshooting

### Issue: FileNotFoundError for data files
**Solution:** Ensure data files are in parent directory:
```
../pgcb+hourly+generation+dataset+(bangladesh)/PGCB_date_power_demand.xlsx
../High Volume Real-World Weather Data/.../BD_weather.csv
```

### Issue: XGBoost module not found
**Solution:**
```bash
pip install --upgrade xgboost scikit-learn
```

### Issue: Memory error during training
**Solution:** Reduce sample size or use subset:
```python
# In main.py, modify:
electricity_df = electricity_df.iloc[-500000:]  # Last ~5 years
```

### Issue: Different results than expected
**Checklist:**
- Random seed is 42 (not changed)
- Data files are unmodified
- No parallel processing interference
- Python version consistent (test with 3.10)
- XGBoost version ≥ 1.5.0

### Issue: Plots not generating
**Solution:** Ensure matplotlib backend is available:
```bash
pip install --upgrade matplotlib
python -c "import matplotlib; matplotlib.use('Agg')"
```

---

## Performance Notes

### Runtime Estimates
- **Data loading:** 10-20 seconds
- **Feature engineering:** 30-60 seconds
- **Model training (200 estimators):** 2-5 minutes
- **Evaluation & plotting:** 30-60 seconds
- **Total:** 5-10 minutes

### Memory Requirements
- Typical RAM usage: 2-4 GB
- Input data: ~1.5 GB (raw PGCB + weather)
- Processed data: ~500 MB

### CPU Scaling
- Single-core: ~10 minutes
- Multi-core (4+): ~3-5 minutes (XGBoost uses all cores by default with `n_jobs=-1`)

---

## Version Control & Updates

### Code Versions Tested
- XGBoost: 1.5.0, 1.7.0, 2.0.0
- scikit-learn: 1.0.0, 1.2.0, 1.3.0
- pandas: 1.3.0, 1.5.0, 2.0.0

### Breaking Changes
None known. If issues arise with new XGBoost versions, pin version:
```bash
pip install xgboost==1.7.0
```

---

## Extended Reproducibility

### Replicating Exact Splits
Train/test split is chronological at 80% cutoff:
- Train period: Start date → 80% timestamp
- Test period: 80% timestamp → End date

To inspect exact dates:
```python
from src.features import FeatureEngineer
# After prepare_features(), check:
print(df_train['datetime'].min(), df_train['datetime'].max())
print(df_test['datetime'].min(), df_test['datetime'].max())
```

### Replicating Feature Engineering
All preprocessing is applied:
1. Scaling with StandardScaler fit on **training data only**
2. Feature lags: 1, 2, 3, 6, 12, 24, 48, 168 hours
3. Calendar features: hour, day_of_week, month, day_of_month, is_peak_hour
4. Weather lagged at 12 and 24 hours
5. Heat stress interaction: Temperature × Humidity / 100

No other preprocessing (no SMOTE, no feature selection, no rotation).

---

## Validation Against Proposal

Each pipeline component is traced to proposal section:

| Component | Proposal Section | Implementation |
|-----------|------------------|-----------------|
| Target construction | 6.3 | features.py:construct_net_load() |
| Feature engineering | 7.1 | features.py (lagged, calendar, weather) |
| XGBoost model | 7.2 | train.py:XGBoostModel |
| Time-series validation | 7.2 | train.py:TimeSeriesValidator |
| Full-horizon metrics | 7.3 | evaluate.py:evaluate_full_horizon() |
| Peak-hour metrics | 7.3 | evaluate.py:evaluate_peak_hours() |
| Seasonal analysis | 7.4 | evaluate.py:evaluate_by_season() |
| Ablation studies | 7.4 | (future extension) |

---

## Archival & Long-term Reproducibility

### Save Reproducibility Snapshot
```bash
# Record environment
pip freeze > environment_snapshot.txt
conda env export > environment.yml

# Record execution metadata
python -c "import sys; print(f'Python {sys.version}')"
python -c "import xgboost; print(f'XGBoost {xgboost.__version__}')"

# Save outputs with timestamp
cp -r outputs outputs_$(date +%Y%m%d_%H%M%S)
```

### Publication Checklist
- [ ] Code clean and documented
- [ ] All outputs reproducible
- [ ] Data sources cited
- [ ] Random seeds fixed
- [ ] Requirements.txt up to date
- [ ] README complete
- [ ] No hard-coded paths (use relative)
- [ ] No data leakage

---

## Contact & Support

For issues or questions about reproducibility, check:
1. This guide
2. src/ module docstrings
3. Main.py logging output
4. outputs/results.json for metrics

---

**Last Updated:** [Timestamp of pipeline execution]
**Pipeline Version:** 1.0
**Proposal:** Constraint Aware Net Load Forecasting (Bangladesh Power System)
