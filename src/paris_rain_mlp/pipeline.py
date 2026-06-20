"""Reproducible temporal training pipeline for the Paris rain MLP."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 48.8566
LONGITUDE = 2.3522
TIMEZONE = "Europe/Paris"
WINDOW_DAYS = 7
RAIN_THRESHOLD_MM = 1.0
DATA_START = "2015-01-01"
DATA_END = "2025-12-31"
TRAIN_END = pd.Timestamp("2022-12-31")
VALIDATION_END = pd.Timestamp("2024-12-31")

DAILY_FEATURES = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "sunshine_duration",
]


@dataclass(frozen=True)
class ModelConfig:
    """MLP candidate configuration."""

    name: str
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    learning_rate_init: float = 0.001


MODEL_CONFIGS = [
    ModelConfig("mlp_8", (8,), 0.001),
    ModelConfig("mlp_16", (16,), 0.001),
    ModelConfig("mlp_16_8", (16, 8), 0.001),
    ModelConfig("mlp_32_16", (32, 16), 0.003),
]
RANDOM_SEEDS = [17, 42, 73]


def download_daily_weather(cache_path: Path) -> pd.DataFrame:
    """Download or load cached daily Open-Meteo weather data for Paris."""

    if cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["date"])
        return _validate_weather_frame(frame)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": DATA_START,
        "end_date": DATA_END,
        "daily": ",".join(DAILY_FEATURES),
        "timezone": TIMEZONE,
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise ValueError("Open-Meteo response does not contain daily weather data.")

    frame = pd.DataFrame(daily).rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["sunshine_duration"] = frame["sunshine_duration"] / 3600.0
    frame = _validate_weather_frame(frame)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def _validate_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, ordering, and missing values in daily weather data."""

    required_columns = ["date", *DAILY_FEATURES]
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing weather columns: {missing_columns}")

    validated = frame[required_columns].copy().sort_values("date").reset_index(drop=True)
    if validated["date"].duplicated().any():
        raise ValueError("Weather data contains duplicate dates.")
    if validated[DAILY_FEATURES].isna().any().any():
        missing = validated[DAILY_FEATURES].isna().sum()
        raise ValueError(f"Weather data contains missing values: {missing[missing > 0].to_dict()}")
    return validated


def build_supervised_windows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create seven-day flattened features and next-day rain targets."""

    rows: list[dict[str, float]] = []
    targets: list[int] = []
    target_dates: list[pd.Timestamp] = []

    for target_index in range(WINDOW_DAYS, len(frame)):
        history = frame.iloc[target_index - WINDOW_DAYS : target_index]
        target_row = frame.iloc[target_index]
        feature_row: dict[str, float] = {}

        for history_offset, (_, day) in enumerate(history.iterrows()):
            lag = WINDOW_DAYS - history_offset
            for feature in DAILY_FEATURES:
                feature_row[f"{feature}_lag_{lag}"] = float(day[feature])

        target_date = pd.Timestamp(target_row["date"])
        day_angle = 2.0 * math.pi * target_date.dayofyear / 365.25
        feature_row["target_day_sin"] = math.sin(day_angle)
        feature_row["target_day_cos"] = math.cos(day_angle)

        rows.append(feature_row)
        targets.append(int(float(target_row["precipitation_sum"]) >= RAIN_THRESHOLD_MM))
        target_dates.append(target_date)

    features = pd.DataFrame(rows)
    labels = pd.Series(targets, name="rain_tomorrow", dtype="int64")
    dates = pd.Series(target_dates, name="target_date")
    return features, labels, dates


def temporal_split(
    features: pd.DataFrame,
    labels: pd.Series,
    dates: pd.Series,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """Split samples by target date into train, validation, and locked test sets."""

    masks = {
        "train": dates <= TRAIN_END,
        "validation": (dates > TRAIN_END) & (dates <= VALIDATION_END),
        "test": dates > VALIDATION_END,
    }
    splits: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}
    for name, mask in masks.items():
        splits[name] = (
            features.loc[mask].reset_index(drop=True),
            labels.loc[mask].reset_index(drop=True),
            dates.loc[mask].reset_index(drop=True),
        )
    return splits


def make_mlp(config: ModelConfig, seed: int) -> Pipeline:
    """Create a standardized MLP classifier for one configuration and seed."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=config.hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=config.alpha,
                    batch_size=64,
                    learning_rate_init=config.learning_rate_init,
                    max_iter=600,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=35,
                    random_state=seed,
                ),
            ),
        ]
    )


def probability_metrics(labels: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Calculate probability and threshold classification metrics."""

    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    predictions = (probabilities_array >= 0.5).astype(int)
    matrix = confusion_matrix(labels_array, predictions, labels=[0, 1])

    return {
        "n_samples": int(len(labels_array)),
        "positive_rate": float(labels_array.mean()),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_array, probabilities_array)),
        "average_precision": float(average_precision_score(labels_array, probabilities_array)),
        "brier_score": float(brier_score_loss(labels_array, probabilities_array)),
        "log_loss": float(log_loss(labels_array, probabilities_array, labels=[0, 1])),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def fit_probability_calibrator(probabilities: np.ndarray, labels: pd.Series) -> LogisticRegression:
    """Fit Platt calibration on validation probabilities."""

    logits = _probability_to_logit(probabilities).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=42)
    calibrator.fit(logits, labels)
    return calibrator


def apply_probability_calibrator(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    """Apply fitted Platt calibration to model probabilities."""

    logits = _probability_to_logit(probabilities).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def evaluate_candidates(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
) -> tuple[Pipeline, ModelConfig, int, list[dict[str, Any]]]:
    """Evaluate all MLP configurations and return the best validation run."""

    results: list[dict[str, Any]] = []
    fitted_models: dict[tuple[str, int], Pipeline] = {}

    for config in MODEL_CONFIGS:
        for seed in RANDOM_SEEDS:
            model = make_mlp(config, seed)
            model.fit(train_features, train_labels)
            classifier = model.named_steps["classifier"]
            train_probabilities = model.predict_proba(train_features)[:, 1]
            validation_probabilities = model.predict_proba(validation_features)[:, 1]
            train_metrics = probability_metrics(train_labels, train_probabilities)
            validation_metrics = probability_metrics(validation_labels, validation_probabilities)
            result = {
                "config": asdict(config),
                "seed": seed,
                "iterations": int(classifier.n_iter_),
                "train": train_metrics,
                "validation": validation_metrics,
                "roc_auc_gap": train_metrics["roc_auc"] - validation_metrics["roc_auc"],
                "log_loss_gap": validation_metrics["log_loss"] - train_metrics["log_loss"],
            }
            results.append(result)
            fitted_models[(config.name, seed)] = model

    best_result = min(
        results,
        key=lambda item: (
            item["validation"]["brier_score"],
            item["validation"]["log_loss"],
            -item["validation"]["roc_auc"],
        ),
    )
    best_config = next(config for config in MODEL_CONFIGS if config.name == best_result["config"]["name"])
    best_seed = int(best_result["seed"])
    return fitted_models[(best_config.name, best_seed)], best_config, best_seed, results


def evaluate_baselines(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
    test_features: pd.DataFrame,
    test_labels: pd.Series,
) -> dict[str, Any]:
    """Fit climatology and logistic-regression baselines."""

    estimators = {
        "climatology": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("classifier", LogisticRegression(C=1.0, max_iter=2_000))]
        ),
    }
    results: dict[str, Any] = {}
    for name, estimator in estimators.items():
        fitted = clone(estimator).fit(train_features, train_labels)
        results[name] = {
            "train": probability_metrics(train_labels, fitted.predict_proba(train_features)[:, 1]),
            "validation": probability_metrics(
                validation_labels,
                fitted.predict_proba(validation_features)[:, 1],
            ),
            "test": probability_metrics(test_labels, fitted.predict_proba(test_features)[:, 1]),
        }
    return results


def calculate_learning_curve(
    config: ModelConfig,
    seed: int,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
) -> list[dict[str, Any]]:
    """Calculate a chronological learning curve for the selected architecture."""

    curve: list[dict[str, Any]] = []
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        sample_count = max(365, int(len(train_features) * fraction))
        subset_features = train_features.iloc[:sample_count]
        subset_labels = train_labels.iloc[:sample_count]
        model = make_mlp(config, seed)
        model.fit(subset_features, subset_labels)
        curve.append(
            {
                "fraction": fraction,
                "n_samples": sample_count,
                "train": probability_metrics(subset_labels, model.predict_proba(subset_features)[:, 1]),
                "validation": probability_metrics(
                    validation_labels,
                    model.predict_proba(validation_features)[:, 1],
                ),
            }
        )
    return curve


def diagnose_fit(train_metrics: dict[str, Any], validation_metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize overfitting and underfitting signals from temporal metrics."""

    roc_gap = train_metrics["roc_auc"] - validation_metrics["roc_auc"]
    loss_gap = validation_metrics["log_loss"] - train_metrics["log_loss"]
    overfitting = roc_gap > 0.08 or loss_gap > 0.10
    underfitting = train_metrics["roc_auc"] < 0.65

    if overfitting:
        summary = "Material train-validation gap: overfitting risk detected."
    elif underfitting:
        summary = "Low train discrimination: underfitting risk detected."
    else:
        summary = "No strong overfitting or underfitting signal under the configured thresholds."

    return {
        "roc_auc_gap": float(roc_gap),
        "log_loss_gap": float(loss_gap),
        "overfitting_flag": overfitting,
        "underfitting_flag": underfitting,
        "summary": summary,
    }


def export_browser_artifact(
    path: Path,
    model: Pipeline,
    calibrator: LogisticRegression,
    feature_names: list[str],
    config: ModelConfig,
    seed: int,
    metrics: dict[str, Any],
    verification_features: np.ndarray,
    verification_probabilities: np.ndarray,
) -> None:
    """Export scaler, dense layers, calibration, and metadata for TypeScript inference."""

    scaler: StandardScaler = model.named_steps["scaler"]
    classifier: MLPClassifier = model.named_steps["classifier"]
    layers = []
    for index, (weights, biases) in enumerate(zip(classifier.coefs_, classifier.intercepts_, strict=True)):
        is_output = index == len(classifier.coefs_) - 1
        layers.append(
            {
                "weights": weights.tolist(),
                "biases": biases.tolist(),
                "activation": "sigmoid" if is_output else "relu",
            }
        )

    artifact = {
        "schemaVersion": 1,
        "modelType": "mlp_binary_classifier",
        "location": {
            "name": "Paris",
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": TIMEZONE,
        },
        "training": {
            "dataStart": DATA_START,
            "trainEnd": str(TRAIN_END.date()),
            "validationEnd": str(VALIDATION_END.date()),
            "testEnd": DATA_END,
            "trainedAt": datetime.now(UTC).isoformat(),
            "config": asdict(config),
            "seed": seed,
        },
        "task": {
            "windowDays": WINDOW_DAYS,
            "rainThresholdMm": RAIN_THRESHOLD_MM,
            "target": "precipitation_sum_next_day_gte_threshold",
        },
        "dailyFeatures": DAILY_FEATURES,
        "featureNames": feature_names,
        "preprocessing": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "layers": layers,
        "calibration": {
            "method": "platt_on_validation_period",
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        },
        "metrics": metrics,
        "verification": {
            "features": verification_features.tolist(),
            "calibratedProbabilities": verification_probabilities.tolist(),
        },
        "disclaimer": "Educational model; not an official weather forecast.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")


def write_reports(
    reports_dir: Path,
    experiment: dict[str, Any],
) -> None:
    """Write JSON metrics and a concise Markdown model report."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected = experiment["selected_model"]
    test = selected["test_calibrated"]
    validation = selected["validation_uncalibrated"]
    train = selected["train_uncalibrated"]
    baseline = experiment["baselines"]
    diagnosis = selected["fit_diagnosis"]

    lines = [
        "# Paris Rain MLP - Model Report",
        "",
        "## Data protocol",
        "",
        f"- Train: 2015-01-08 to 2022-12-31 ({train['n_samples']} windows)",
        f"- Validation: 2023-01-01 to 2024-12-31 ({validation['n_samples']} windows)",
        f"- Locked test: 2025-01-01 to 2025-12-31 ({test['n_samples']} windows)",
        f"- Positive target: next-day precipitation >= {RAIN_THRESHOLD_MM:.1f} mm",
        "",
        "## Selected model",
        "",
        f"- Configuration: `{selected['config']['name']}` {tuple(selected['config']['hidden_layer_sizes'])}",
        f"- Seed: `{selected['seed']}`",
        f"- Validation Brier score before calibration: `{validation['brier_score']:.4f}`",
        f"- Validation ROC-AUC before calibration: `{validation['roc_auc']:.4f}`",
        "",
        "## Locked 2025 test metrics",
        "",
        f"- ROC-AUC: `{test['roc_auc']:.4f}`",
        f"- Average precision: `{test['average_precision']:.4f}`",
        f"- Brier score: `{test['brier_score']:.4f}`",
        f"- Log-loss: `{test['log_loss']:.4f}`",
        f"- Accuracy: `{test['accuracy']:.4f}`",
        f"- Balanced accuracy: `{test['balanced_accuracy']:.4f}`",
        f"- Precision: `{test['precision']:.4f}`",
        f"- Recall: `{test['recall']:.4f}`",
        f"- F1: `{test['f1']:.4f}`",
        f"- Confusion matrix [[TN, FP], [FN, TP]]: `{test['confusion_matrix']}`",
        "",
        "## Baseline comparison on 2025 test",
        "",
        "| Model | ROC-AUC | Brier | Log-loss | F1 |",
        "|---|---:|---:|---:|---:|",
        _baseline_row("Climatology", baseline["climatology"]["test"]),
        _baseline_row("Logistic regression", baseline["logistic_regression"]["test"]),
        _baseline_row("Selected calibrated MLP", test),
        "",
        "## Generalization diagnosis",
        "",
        f"- Train ROC-AUC: `{train['roc_auc']:.4f}`",
        f"- Validation ROC-AUC: `{validation['roc_auc']:.4f}`",
        f"- ROC-AUC gap: `{diagnosis['roc_auc_gap']:.4f}`",
        f"- Log-loss gap: `{diagnosis['log_loss_gap']:.4f}`",
        f"- Assessment: {diagnosis['summary']}",
        "",
        "See `metrics.json` for every candidate, seed, learning-curve point, and metric.",
    ]
    (reports_dir / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {metrics['roc_auc']:.4f} | {metrics['brier_score']:.4f} | "
        f"{metrics['log_loss']:.4f} | {metrics['f1']:.4f} |"
    )


def run_experiment(project_root: Path) -> None:
    """Run the full experiment from download through browser-artifact export."""

    raw_path = project_root / "data" / "raw" / "paris_daily_2015_2025.csv"
    weather = download_daily_weather(raw_path)
    features, labels, dates = build_supervised_windows(weather)
    splits = temporal_split(features, labels, dates)
    train_features, train_labels, train_dates = splits["train"]
    validation_features, validation_labels, validation_dates = splits["validation"]
    test_features, test_labels, test_dates = splits["test"]

    print(
        f"Windows: train={len(train_features)}, validation={len(validation_features)}, "
        f"test={len(test_features)}, features={train_features.shape[1]}"
    )
    print(
        f"Positive rates: train={train_labels.mean():.3f}, validation={validation_labels.mean():.3f}, "
        f"test={test_labels.mean():.3f}"
    )

    baselines = evaluate_baselines(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    )
    selected_model, selected_config, selected_seed, candidates = evaluate_candidates(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )

    train_probabilities = selected_model.predict_proba(train_features)[:, 1]
    validation_probabilities = selected_model.predict_proba(validation_features)[:, 1]
    test_probabilities = selected_model.predict_proba(test_features)[:, 1]
    calibrator = fit_probability_calibrator(validation_probabilities, validation_labels)
    calibrated_validation = apply_probability_calibrator(calibrator, validation_probabilities)
    calibrated_test = apply_probability_calibrator(calibrator, test_probabilities)

    train_metrics = probability_metrics(train_labels, train_probabilities)
    validation_metrics = probability_metrics(validation_labels, validation_probabilities)
    test_uncalibrated_metrics = probability_metrics(test_labels, test_probabilities)
    test_calibrated_metrics = probability_metrics(test_labels, calibrated_test)
    validation_calibrated_metrics = probability_metrics(validation_labels, calibrated_validation)
    fit_diagnosis = diagnose_fit(train_metrics, validation_metrics)
    learning_curve = calculate_learning_curve(
        selected_config,
        selected_seed,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )

    experiment: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "source": OPEN_METEO_ARCHIVE_URL,
            "daily_rows": len(weather),
            "feature_count": train_features.shape[1],
            "window_days": WINDOW_DAYS,
            "rain_threshold_mm": RAIN_THRESHOLD_MM,
            "splits": {
                "train": {
                    "start": str(train_dates.min().date()),
                    "end": str(train_dates.max().date()),
                    "n_samples": len(train_features),
                },
                "validation": {
                    "start": str(validation_dates.min().date()),
                    "end": str(validation_dates.max().date()),
                    "n_samples": len(validation_features),
                },
                "test": {
                    "start": str(test_dates.min().date()),
                    "end": str(test_dates.max().date()),
                    "n_samples": len(test_features),
                },
            },
        },
        "baselines": baselines,
        "candidates": candidates,
        "learning_curve": learning_curve,
        "selected_model": {
            "config": asdict(selected_config),
            "seed": selected_seed,
            "train_uncalibrated": train_metrics,
            "validation_uncalibrated": validation_metrics,
            "validation_calibrated": validation_calibrated_metrics,
            "test_uncalibrated": test_uncalibrated_metrics,
            "test_calibrated": test_calibrated_metrics,
            "fit_diagnosis": fit_diagnosis,
        },
    }

    export_browser_artifact(
        project_root / "artifacts" / "paris_rain_mlp.json",
        selected_model,
        calibrator,
        features.columns.tolist(),
        selected_config,
        selected_seed,
        {
            "validation_uncalibrated": validation_metrics,
            "test_uncalibrated": test_uncalibrated_metrics,
            "test_calibrated": test_calibrated_metrics,
        },
        test_features.tail(5).to_numpy(dtype=float),
        calibrated_test[-5:],
    )
    write_reports(project_root / "reports", experiment)

    print(f"Selected: {selected_config.name}, seed={selected_seed}")
    print(
        "Locked 2025 test: "
        f"ROC-AUC={test_calibrated_metrics['roc_auc']:.4f}, "
        f"Brier={test_calibrated_metrics['brier_score']:.4f}, "
        f"log-loss={test_calibrated_metrics['log_loss']:.4f}, "
        f"F1={test_calibrated_metrics['f1']:.4f}"
    )
    print(f"Fit diagnosis: {fit_diagnosis['summary']}")
