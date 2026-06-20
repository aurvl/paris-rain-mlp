# Paris Rain MLP

Educational multilayer perceptron that estimates the probability of meaningful rain in Paris tomorrow from the seven
previous completed days of weather observations.

The model is designed for a portfolio blog widget. Training runs offline in Python; browser inference will use the
small exported JSON artifact without a Python server or a JavaScript ML framework.

## Temporal protocol

- Train targets: 2015-01-08 through 2022-12-31
- Validation targets: 2023-01-01 through 2024-12-31
- Locked test targets: 2025-01-01 through 2025-12-31
- Input window: seven previous days
- Positive target: next-day precipitation of at least 1 mm

No target from validation or test is used for fitting the neural-network weights. The validation period is used for
architecture selection and probability calibration. The 2025 test period is evaluated once at the end.

## Run

```powershell
python train.py
```

Generated outputs:

- `artifacts/paris_rain_mlp.json`: scaler, MLP weights, biases, calibration, schema, and metrics
- `reports/metrics.json`: complete machine-readable experiment results
- `reports/model_report.md`: readable comparison and generalization report

Raw Open-Meteo data is cached under `data/raw/` and intentionally ignored by Git.

## Runtime contract

The future widget must fetch the same seven daily variables, order the previous seven days from oldest to newest,
derive target-date seasonality, standardize with the exported scaler, run each dense layer, and apply the exported
sigmoid calibration.

This is a pedagogical model, not an official weather forecast.

