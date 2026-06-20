"""Verify that the exported browser artifact reproduces Python MLP inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paris_rain_mlp.pipeline import (  # noqa: E402
    build_supervised_windows,
    download_daily_weather,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid."""

    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def run_artifact_inference(artifact: dict[str, object], features: np.ndarray) -> np.ndarray:
    """Run preprocessing, dense layers, and Platt calibration from exported JSON."""

    preprocessing = artifact["preprocessing"]
    activations = (features - np.asarray(preprocessing["mean"])) / np.asarray(preprocessing["scale"])

    for layer in artifact["layers"]:
        activations = activations @ np.asarray(layer["weights"]) + np.asarray(layer["biases"])
        activations = sigmoid(activations) if layer["activation"] == "sigmoid" else np.maximum(activations, 0.0)

    probabilities = activations.reshape(-1)
    calibration = artifact["calibration"]
    logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
    return sigmoid(calibration["coefficient"] * logits + calibration["intercept"])


def main() -> None:
    """Validate artifact schema, dimensions, and probability range."""

    artifact_path = PROJECT_ROOT / "artifacts" / "paris_rain_mlp.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    weather = download_daily_weather(PROJECT_ROOT / "data" / "raw" / "paris_daily_2015_2025.csv")
    features, _, _ = build_supervised_windows(weather)
    ordered = features[artifact["featureNames"]].tail(32).to_numpy(dtype=float)
    probabilities = run_artifact_inference(artifact, ordered)

    assert probabilities.shape == (32,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert len(artifact["featureNames"]) == ordered.shape[1] == 51
    assert artifact["task"]["windowDays"] == 7

    verification = artifact["verification"]
    verification_features = np.asarray(verification["features"], dtype=float)
    expected = np.asarray(verification["calibratedProbabilities"], dtype=float)
    actual = run_artifact_inference(artifact, verification_features)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    print(
        f"Artifact verification passed: {len(probabilities)} predictions, range={probabilities.min():.4f}-{probabilities.max():.4f}"
    )


if __name__ == "__main__":
    main()
