import json

with open('outputs/results_all_models.json', 'r') as f:
    results = json.load(f)
    
xgb = results['results']['XGBoost']['peak_hours']
print('XGBoost Peak-Hour Metrics:')
print(f'  Under-forecast rate: {xgb["under_forecast_rate"]*100:.2f}%')
print(f'  Large under-forecast rate (>500MW): {xgb["large_under_forecast_rate"]*100:.2f}%')
print(f'  Max under-forecast: {xgb["max_under_forecast"]:.2f} MW')
print(f'  Max over-forecast: {xgb["max_over_forecast"]:.2f} MW')
print(f'  Peak MAE: {xgb["peak_mae"]:.2f} MW')
