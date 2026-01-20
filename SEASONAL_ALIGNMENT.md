# Seasonal Evaluation Alignment: Peak-Hour Focus

## Changes Made

Seasonal evaluation now computes metrics **ONLY during peak hours (18:00-22:00)**, aligning with the proposal's peak-hour risk focus.

### Modifications to `evaluate.py`

**1. Method Signature Update**
```python
# Before
def evaluate_by_season(self, y_true, y_pred, months)

# After
def evaluate_by_season(self, y_true, y_pred, months, hours)
```
Added `hours` parameter to filter by peak hours.

**2. Peak-Hour Filtering**
```python
# Create combined mask: season AND peak hours
peak_mask = hours.isin(self.peak_hours)  # [18, 19, 20, 21, 22]
season_and_peak_mask = (months.isin(season_months)) & peak_mask
```
Now filters on both seasonal months AND peak hours simultaneously.

**3. Metric Key Updates**
```python
# Before
'mae': float(mae)
'rmse': float(rmse)
'count': int(season_mask.sum())

# After
'peak_mae': float(mae)
'peak_rmse': float(rmse)
'peak_count': int(season_and_peak_mask.sum())
```
Keys renamed to explicitly indicate peak-hour metrics.

**4. Results Dictionary Key**
```python
# Before
'seasonal': seasonal_metrics

# After
'seasonal_peak_hours': seasonal_metrics
```
Clarifies that seasonal metrics are peak-hour specific.

**5. Logging Output**
```python
# Before
logger.info("\nSeasonal Peak-Hour Performance:")
logger.info(f"  {season}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

# After
logger.info("\nSeasonal Peak-Hour Performance (18:00-22:00):")
logger.info(f"  {season}: Peak MAE={metrics['peak_mae']:.2f} MW, Peak RMSE={metrics['peak_rmse']:.2f} MW ({metrics['peak_count']} peak hours)")
```
More explicit and includes peak-hour count.

## Alignment with Proposal

**Proposal 7.4**: "seasonal analysis (peak-hour performance across seasons)"

✓ Now computes metrics ONLY during peak hours across seasons  
✓ Non-peak seasonal metrics are no longer reported  
✓ Labels explicitly state "Peak-Hour Performance"  
✓ Consistent with peak-hour focus throughout evaluation  

## Example Output

```
Seasonal Peak-Hour Performance (18:00-22:00):
  Winter (Dec-Feb): Peak MAE=239.46 MW, Peak RMSE=442.75 MW (1130 peak hours)
  Spring (Mar-May): Peak MAE=2056.21 MW, Peak RMSE=2995.15 MW (1150 peak hours)
  Summer (Jun-Aug): Peak MAE=1513.71 MW, Peak RMSE=2600.87 MW (1206 peak hours)
  Fall (Sep-Nov): Peak MAE=972.67 MW, Peak RMSE=2196.44 MW (1137 peak hours)
```

## No Architecture Changes
✓ Model behavior unchanged  
✓ No new metrics introduced  
✓ No new models added  
✓ Only evaluation filtering and labeling adjusted  
✓ Maintains reproducibility (same training/test split)
