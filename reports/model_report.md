# Paris Rain MLP - Model Report

## Data protocol

- Train: 2015-01-08 to 2022-12-31 (2915 windows)
- Validation: 2023-01-01 to 2024-12-31 (731 windows)
- Locked test: 2025-01-01 to 2025-12-31 (365 windows)
- Positive target: next-day precipitation >= 1.0 mm

## Selected model

- Configuration: `mlp_8` (8,)
- Seed: `17`
- Validation Brier score before calibration: `0.2158`
- Validation ROC-AUC before calibration: `0.7165`

## Locked 2025 test metrics

- ROC-AUC: `0.7052`
- Average precision: `0.5420`
- Brier score: `0.1976`
- Log-loss: `0.5812`
- Accuracy: `0.7041`
- Balanced accuracy: `0.6395`
- Precision: `0.5567`
- Recall: `0.4538`
- F1: `0.5000`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[203, 43], [65, 54]]`

## Baseline comparison on 2025 test

| Model | ROC-AUC | Brier | Log-loss | F1 |
|---|---:|---:|---:|---:|
| Climatology | 0.5000 | 0.2207 | 0.6336 | 0.0000 |
| Logistic regression | 0.7194 | 0.1911 | 0.5645 | 0.3750 |
| Selected calibrated MLP | 0.7052 | 0.1976 | 0.5812 | 0.5000 |

## Generalization diagnosis

- Train ROC-AUC: `0.7770`
- Validation ROC-AUC: `0.7165`
- ROC-AUC gap: `0.0605`
- Log-loss gap: `0.0922`
- Assessment: No strong overfitting or underfitting signal under the configured thresholds.

See `metrics.json` for every candidate, seed, learning-curve point, and metric.
