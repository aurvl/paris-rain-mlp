# Paris Rain MLP

Machine-learning project that estimates the probability of meaningful rain in Paris tomorrow from the seven previous
completed days of weather observations.

The repository covers data acquisition, temporal feature engineering, baseline comparison, MLP architecture selection,
probability calibration, locked out-of-time evaluation, and portable model export.

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
python verify_artifact.py
```

Generated outputs:

- `artifacts/paris_rain_mlp.json`: scaler, MLP weights, biases, calibration, schema, and metrics
- `reports/metrics.json`: complete machine-readable experiment results
- `reports/model_report.md`: readable comparison and generalization report

Raw Open-Meteo data is cached under `data/raw/` and intentionally ignored by Git.

## Portable inference contract

Any inference client must provide the same seven daily variables, order the previous seven days from oldest to newest,
derive target-date seasonality, standardize with the exported scaler, run each dense layer, and apply the exported
sigmoid calibration.

This is a pedagogical model, not an official weather forecast.
