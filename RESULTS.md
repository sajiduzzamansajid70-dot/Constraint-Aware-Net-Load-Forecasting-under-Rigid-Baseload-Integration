## Final Model Comparison (Including Persistence Baselines)

All models were evaluated on an identical chronological test set.
Peak-hour (18:00–22:00) RMSE is treated as the primary operational metric.

| Model              | MAE (MW) | RMSE (MW) | Peak RMSE (MW) | Role        |
|-------------------|----------|-----------|----------------|-------------|
| A0_XGBoost        | 365.94   | 618.12    | 750.26         | Primary     |
| A1_MA_ARIMA       | 1674.27  | 2067.93   | 2015.85        | Comparative |
| A3_Hybrid         | 1763.52  | 2090.70   | 2023.79        | Comparative |
| A4_Persist_24h    | 790.43   | 1193.76   | 1369.38        | Baseline    |
| A5_Persist_168h   | 1514.22  | 1966.42   | 1920.97        | Baseline    |

Persistence baselines demonstrate that naive repetition fails during
peak hours, confirming task difficulty and the value of learning-based models.
