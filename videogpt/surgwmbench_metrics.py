from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


EMPTY_IMAGE_METRICS: Dict[str, Optional[float]] = {
    "mse": None,
    "psnr": None,
    "ssim": None,
    "lpips": None,
    "fvd": None,
}

EMPTY_TRAJECTORY_METRICS: Dict[str, Optional[float]] = {
    "ade": None,
    "fde": None,
    "frechet": None,
    "hausdorff": None,
    "endpoint_error": None,
    "trajectory_length_error": None,
    "smoothness": None,
}


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    clean = [(float(value), float(weight)) for value, weight in zip(values, weights) if value is not None]
    if not clean:
        return None
    total_weight = sum(weight for _, weight in clean)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in clean) / total_weight


def _euclidean(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=-1)


def _trajectory_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())


def _smoothness(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(np.linalg.norm(np.diff(points, n=2, axis=0), axis=-1).mean())


def _discrete_frechet(pred: np.ndarray, target: np.ndarray) -> float:
    ca = np.full((len(pred), len(target)), -1.0, dtype=np.float64)

    def compute(i: int, j: int) -> float:
        if ca[i, j] > -1:
            return float(ca[i, j])
        dist = float(np.linalg.norm(pred[i] - target[j]))
        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i > 0 and j == 0:
            ca[i, j] = max(compute(i - 1, 0), dist)
        elif i == 0 and j > 0:
            ca[i, j] = max(compute(0, j - 1), dist)
        else:
            ca[i, j] = max(
                min(compute(i - 1, j), compute(i - 1, j - 1), compute(i, j - 1)),
                dist,
            )
        return float(ca[i, j])

    if len(pred) == 0 or len(target) == 0:
        return math.nan
    return compute(len(pred) - 1, len(target) - 1)


def _hausdorff(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 0 or len(target) == 0:
        return math.nan
    distances = np.linalg.norm(pred[:, None, :] - target[None, :, :], axis=-1)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def trajectory_metrics(
    prediction: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> Dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"Prediction and target trajectory shapes differ: {pred.shape} vs {tgt.shape}")
    distances = _euclidean(pred, tgt)
    return {
        "ade": float(distances.mean()),
        "fde": float(distances[-1]),
        "frechet": _discrete_frechet(pred, tgt),
        "hausdorff": _hausdorff(pred, tgt),
        "endpoint_error": float(distances[-1]),
        "trajectory_length_error": abs(_trajectory_length(pred) - _trajectory_length(tgt)),
        "smoothness": abs(_smoothness(pred) - _smoothness(tgt)),
    }


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, Optional[float]]:
    pred = prediction.astype(np.float64) / 255.0
    tgt = target.astype(np.float64) / 255.0
    mse = float(np.mean((pred - tgt) ** 2))
    psnr = None if mse == 0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    if mse == 0:
        psnr = float("inf")
    ssim = _simple_ssim(pred, tgt)
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": None,
        "fvd": None,
    }


def _simple_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = float(pred.mean())
    mu_y = float(target.mean())
    sigma_x = float(pred.var())
    sigma_y = float(target.var())
    sigma_xy = float(((pred - mu_x) * (target - mu_y)).mean())
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def aggregate_metric_dicts(items: Iterable[Dict[str, Any]], weights: Iterable[float]) -> Dict[str, Optional[float]]:
    rows = list(items)
    row_weights = list(weights)
    if not rows:
        return {}
    keys = rows[0].keys()
    aggregated: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        if any(value is None for value in values):
            aggregated[key] = None if all(value is None for value in values) else _weighted_mean(
                [value for value in values if value is not None],
                [weight for value, weight in zip(values, row_weights) if value is not None],
            )
        else:
            aggregated[key] = _weighted_mean(values, row_weights)
    return aggregated


def empty_image_metrics() -> Dict[str, Optional[float]]:
    return dict(EMPTY_IMAGE_METRICS)


def empty_trajectory_metrics() -> Dict[str, Optional[float]]:
    return dict(EMPTY_TRAJECTORY_METRICS)
