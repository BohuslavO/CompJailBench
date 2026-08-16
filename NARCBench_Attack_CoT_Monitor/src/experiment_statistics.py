"""Dependency-light statistics for matched attack/defense experiments."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    total_positives = sum(labels)
    if total_positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / total_positives


def confusion_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "n": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(labels)),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "recall": recall,
        "false_positive_rate": safe_div(fp, fp + tn),
        "specificity": specificity,
        "precision": safe_div(tp, tp + fp),
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "threshold": threshold,
        "recall_ci95": wilson_interval(tp, tp + fn),
        "fpr_ci95": wilson_interval(fp, fp + tn),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_interval(
    pairs: Sequence[Any],
    statistic: Callable[[list[Any]], float | None],
    *,
    iterations: int = 10_000,
    seed: int = 20260814,
) -> tuple[float | None, float | None]:
    if not pairs:
        return None, None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    return percentile(values, 0.025), percentile(values, 0.975)


def exact_mcnemar_pvalue(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Two-sided exact McNemar test for paired binary outcomes."""

    b = sum(first == 1 and second == 0 for first, second in pairs)
    c = sum(first == 0 and second == 1 for first, second in pairs)
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)
