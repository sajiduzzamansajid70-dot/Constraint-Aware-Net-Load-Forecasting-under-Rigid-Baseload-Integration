# Metric Correction: Under-Forecast Logic

## Issue Fixed
The under-forecast rate calculation was inverted, using the wrong inequality operator and wrong error extrema.

## Correction Summary

### Definition Clarity
```
error = y_true - y_pred

Under-forecast (error > 0): Predicted too LOW
  - Actual demand > Predicted demand
  - System at risk of under-capacity
  
Over-forecast (error < 0): Predicted too HIGH
  - Actual demand < Predicted demand
  - System has excess capacity
```

### Changes Made

**1. Under-forecast rate**
- **Before**: `(errors < 0).sum() / len(errors)` ❌ WRONG (inverted)
- **After**: `(errors > 0).sum() / len(errors)` ✓ CORRECT

**2. Large under-forecast rate**
- **Before**: `(errors < -large_under_forecast_threshold).sum()` ❌ WRONG
- **After**: `(errors > large_under_forecast_threshold).sum()` ✓ CORRECT

**3. Max under-forecast value**
- **Before**: `np.min(errors)` ❌ WRONG (gives most negative = overprediction)
- **After**: `np.max(errors)` ✓ CORRECT (gives largest positive error)

**4. Max over-forecast value**
- **Before**: `np.max(errors)` ❌ WRONG (gives most positive = underprediction)
- **After**: `np.min(errors)` ✓ CORRECT (gives most negative error)

## Impact on Results

The corrected logic now properly identifies:
- **Under-forecast events**: When model predicts demand too low (operational risk)
- **Over-forecast events**: When model predicts demand too high (capacity surplus)

### Example
If at peak hour:
- Actual load: 10,000 MW
- Predicted load: 8,500 MW
- Error: 10,000 - 8,500 = **+1,500 MW** (positive)
- Interpretation: **Under-forecast** by 1,500 MW
  - System expected to supply ~1,500 MW less than actually needed
  - This is an operational risk event ⚠️

## Code Location
- **File**: `src/evaluate.py`
- **Method**: `evaluate_peak_hours()`
- **Lines**: 115-137

## Model Behavior
✓ **No change to model predictions**
✓ **No change to training or data**
✓ **Only metric interpretation corrected**
✓ **Results now accurately reflect operational risk**

## Validation
The corrected metrics now properly reflect:
- System vulnerability during peak hours
- Frequency of under-capacity events
- Severity of demand forecasting misses
- Operational planning implications
