# Constraint-Aware Net Load Forecasting: Bangladesh Power System
## Peak-Hour Risk-Focused Machine Learning Study

This repository contains a clean, reproducible research pipeline for short-term net load forecasting under rigid baseload integration, aligned with the research proposal:

**Title:** *Constraint Aware Net Load Forecasting under Rigid Baseload Integration: A Peak-Hour Risk-Focused Machine Learning Study for the Bangladesh Power System*

---

## Overview

### Research Problem
Bangladesh's power system faces a critical operational shift with the planned integration of large, non-dispatchable baseload generation (nuclear). When baseload output is treated as must-run, forecasting errors during peak demand hours become costly because the system has fewer corrective actions available.

### Research Solution
This study reframes the forecasting problem to match operational reality:
- **Target:** Constraint-aware net load (residual demand after baseload and renewables)
- **Evaluation:** Peak-hour (18:00-22:00) performance is PRIMARY metric
- **Model:** Strong ML baseline (XGBoost) trained on long-horizon historical data
- **No shortcuts:** Reproducible pipeline with strict leakage prevention

---

## Project Structure

```
constraint_aware_net_load/
├── data/                           # Input data directory
├── src/                            # Core modules
│   ├── data_loader.py             # Load electricity and weather data
│   ├── features.py                # Target construction and feature engineering
│   ├── train.py                   # XGBoost model training
│   └── evaluate.py                # Evaluation (full-horizon + peak-hour)
├── main.py                         # Pipeline orchestration
├── outputs/                        # Results, models, plots
│   ├── results.json               # Full evaluation metrics
│   ├── test_predictions.csv       # Test set predictions
│   ├── feature_importance.csv     # Feature rankings
│   ├── models/                    # Saved model and scaler
│   └── plots/                     # Diagnostic plots
├── docs/                           # Documentation
├── README.md                       # This file
├── REPRODUCIBILITY.md             # Detailed reproducibility guide
└── requirements.txt               # Python dependencies
```

---

## Data Sources

### Electricity System Data
- **Source:** PGCB hourly generation dataset (Bangladesh)
- **Coverage:** ~92,650 hourly observations (2015+)
- **Columns:** demand, generation, load shedding, fuel breakdown, imports, renewables
- **Used for:** Constructing served load and net load target

### Meteorological Data
- **Source:** Bangladesh weather stations (national records)
- **Coverage:** Daily observations from 35 stations (1961-2023)
- **Columns:** Temperature, Humidity, Rainfall, Sunshine
- **Used for:** Feature engineering (lagged weather, heat stress interactions)

---

## Methodology (Aligned with Proposal)

### Target Variable Definition
```
Net Load(t) = Served Load(t) - Rigid Baseload(t) - Renewable Output(t)

where:
- Served Load = Demand - Load Shedding
- Rigid Baseload = 2200 MW (nuclear planning scenario)
- Renewable Output = Solar + Wind
```

The net load represents the residual demand that must be served by flexible generation and reserves.

### Feature Engineering (Section 7.1)
1. **Lagged net load features** (short: 1-48h, medium: 7 days)
2. **Calendar features** (hour, day of week, month, peak hour indicator)
3. **Weather features** (lagged temperature, lagged humidity, heat stress)
4. **No leakage:** All preprocessing fit on training data only

### Model: XGBoost Baseline (Section 7.2)
- **Model:** XGBoost regression
- **Rationale:** Scalability, robustness, interpretability via feature importance
- **Training:** Time-series validation respecting chronology (80/20 split)
- **Hyperparameters:** 200 estimators, max_depth=6, lr=0.1

### Evaluation: Peak-Hour Risk Focus (Section 7.3)
**PRIMARY Metrics (Peak Hours 18:00-22:00):**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Under-forecast rate (frequency of negative errors)
- Large under-forecast rate (>500 MW threshold)

**SECONDARY Metrics (Full Horizon):**
- MAE, RMSE across all hours
- Error distributions and seasonal breakdown

---

## Installation & Setup

### Requirements
- Python 3.9+
- Dependencies: pandas, numpy, xgboost, scikit-learn, matplotlib, seaborn, scipy

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Setup
Ensure the following data files are in the parent directory:
```
../
├── pgcb+hourly+generation+dataset+(bangladesh)/
│   └── PGCB_date_power_demand.xlsx
└── High Volume Real-World Weather Data/
    └── High Volume Real-World Weather Data/
        └── Weather Datasets/
            └── Combined Data/
                └── BD_weather.csv
```

---

## Running the Pipeline

### Complete Pipeline (Recommended)
```bash
# From project root
python main.py
```

This runs all phases:
1. Load and validate data
2. Construct net load target
3. Engineer features
4. Train XGBoost model
5. Evaluate (full-horizon + peak-hour)
6. Save results and plots

### Individual Modules (Testing)
```bash
# Test data loading
python -m src.data_loader

# Test feature engineering
python -m src.features

# Test model training
python -m src.train

# Test evaluation
python -m src.evaluate
```

---

## Results

The pipeline generates:

### Metrics (outputs/results.json)
- Full-horizon MAE, RMSE
- Peak-hour MAE, RMSE (PRIMARY)
- Under-forecast rate and large under-forecast frequency
- Seasonal breakdown

### Artifacts
- **test_predictions.csv:** True vs predicted net load on test set
- **feature_importance.csv:** Ranked feature contributions
- **models/:** Serialized XGBoost model and StandardScaler
- **plots/:**
  - `timeseries.png` - True vs predicted time series
  - `errors_by_hour.png` - Residuals by hour (peak hours highlighted)
  - `error_distribution.png` - Peak vs non-peak error distributions
  - `qq_plot.png` - Residuals vs normal distribution

---

## Key Design Choices

### Why XGBoost?
From proposal 7.2: "chosen for scalability, robustness, and interpretability through feature importance and partial dependence"

No decomposition methods (EMD, CEEMDAN) unless explicitly justified by hypothesis tests.

### Why Peak-Hour Evaluation?
From proposal 7.3: "The peak-hour window is selected because it aligns with the most operationally constrained period and the highest risk of imbalance."

Under rigid baseload, system flexibility is most limited during 18:00-22:00 when demand peaks.

### Why Chronological Train-Test Split?
Time series forecasting must respect temporal order. No shuffling, no future leakage.

### Why 2200 MW Baseload?
Scenario modeling for future nuclear integration in Bangladesh (~2200 MW planned).

---

## Reproducibility

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for:
- Environment setup instructions
- Seed configuration
- Data validation checks
- Expected output ranges
- Troubleshooting

---

## Authors & Attribution
Research and implementation strictly aligned with proposal:
- Problem framing: Constraint-aware net load under rigid baseload
- Evaluation: Peak-hour risk focus
- Methods: No shortcuts, no unnecessary complexity

---

## References

1. Bangladesh Power System Operator (PGCB) data
2. National meteorological records
3. XGBoost: Chen & Guestrin (2016)
4. Time-series evaluation best practices: Hyndman et al.

---

## License & Data Usage

Data access respects institutional agreements. Published results will comply with sharing permissions.

---

## Questions or Issues?

See REPRODUCIBILITY.md for common issues or contact the research team.
