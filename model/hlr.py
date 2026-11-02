"""Half-life regression, and the prerequisite coupling built on top of it.

Base model
----------
Settles and Meeder (ACL 2016), "A Trainable Spaced Repetition Model for Language
Learning", predict a memory half-life from features of the practice history:

    h_hat = 2 ** (theta . x)

and recall decays exponentially in the time since the last practice:

    p = 2 ** (-delta / h)

Fitting minimises squared error on both the observed recall and the half-life
that observation implies, plus L2:

    loss = (p - p_hat)^2 + alpha * (h - h_hat)^2 + lambda * ||theta||^2

The implied half-life of one observation, -delta / log2(p), blows up as p
approaches 1 and collapses as it approaches 0, so it is clamped. The clamp is a
real modelling choice and its bounds are reported, not hidden.

The coupling extension
----------------------
Every spaced repetition system treats items as independent. Course concepts are
not: recall of the chain rule and recall of implicit differentiation are not two
unrelated facts. So let a concept's effective half-life shrink when its
prerequisites have decayed:

    h_eff(i) = h_hat(i) * prod over prereqs j of p(j) ** gamma

One scalar gamma for the whole model, fitted alongside theta. gamma = 0 recovers
the base model exactly, which matters: the extension is nested inside the thing
it is being compared against, so the comparison is honest by construction and a
gamma that comes out near zero is a real null result rather than a failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# Observations at the extremes imply an infinite or zero half-life; clamp both.
MIN_HALF_LIFE = 15.0 / (60 * 24)   # 15 minutes, in days
MAX_HALF_LIFE = 274.0              # about nine months
MIN_P, MAX_P = 0.0001, 0.9999


def recall(delta_days, half_life):
    """p = 2 ** (-delta / h), the exponential forgetting curve."""
    half_life = np.maximum(half_life, MIN_HALF_LIFE)
    return np.exp2(-np.asarray(delta_days, dtype=float) / half_life)


def recall_scalar(delta_days: float, half_life: float) -> float:
    """The same curve for one value. The simulations call this millions of
    times, and routing single floats through numpy costs more than the sum."""
    return math.exp2(-delta_days / max(half_life, MIN_HALF_LIFE))


def clamp(value: float) -> float:
    return MIN_HALF_LIFE if value < MIN_HALF_LIFE else MAX_HALF_LIFE if value > MAX_HALF_LIFE else value


def implied_half_life(delta_days, observed_p):
    """Invert the decay for a single observation, clamped at both ends."""
    p = np.clip(np.asarray(observed_p, dtype=float), MIN_P, MAX_P)
    delta = np.asarray(delta_days, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -delta / np.log2(p)
    return np.clip(np.nan_to_num(h, nan=MAX_HALF_LIFE, posinf=MAX_HALF_LIFE), MIN_HALF_LIFE, MAX_HALF_LIFE)


@dataclass
class Fitted:
    theta: np.ndarray
    gamma: float
    feature_names: tuple[str, ...]
    alpha: float
    lam: float
    iterations: int
    loss: float

    def half_life(self, features: np.ndarray) -> np.ndarray:
        return np.clip(np.exp2(features @ self.theta), MIN_HALF_LIFE, MAX_HALF_LIFE)

    def predict(self, features: np.ndarray, delta_days, prereq_recall=None) -> np.ndarray:
        """Predicted recall. prereq_recall is the product term already reduced.

        prereq_recall is prod(p_j) over the concept's prerequisites at review
        time, or None when the concept has none. gamma = 0 makes it irrelevant,
        which is exactly what the base model is.
        """
        h = self.half_life(features)
        if prereq_recall is not None and self.gamma != 0.0:
            coupling = np.clip(np.asarray(prereq_recall, dtype=float), MIN_P, 1.0) ** self.gamma
            h = np.clip(h * coupling, MIN_HALF_LIFE, MAX_HALF_LIFE)
        return recall(delta_days, h)


def _loss(params, features, delta, observed, prereq, alpha, lam, couple):
    theta = params[:-1] if couple else params
    gamma = params[-1] if couple else 0.0
    h = np.clip(np.exp2(features @ theta), MIN_HALF_LIFE, MAX_HALF_LIFE)
    if couple:
        h = np.clip(h * np.clip(prereq, MIN_P, 1.0) ** gamma, MIN_HALF_LIFE, MAX_HALF_LIFE)
    p_hat = recall(delta, h)
    h_observed = implied_half_life(delta, observed)
    # Half-life error is compared on a log scale: half-lives span minutes to
    # months, so a squared error in days would be dominated by the long tail.
    error_p = np.mean((observed - p_hat) ** 2)
    error_h = np.mean((np.log2(h_observed) - np.log2(h)) ** 2)
    return error_p + alpha * error_h + lam * float(theta @ theta)


def fit(features, delta_days, observed, prereq_recall=None, *, couple=False,
        alpha=0.0, lam=0.01, feature_names=(), seed=20261121) -> Fitted:
    """Fit theta, and gamma too when couple is True.

    alpha weights the half-life term and defaults to 0 here, which departs from
    the published loss on purpose. Settles and Meeder fit on session recall, a
    proportion over several items, so the half-life a session implies is a real
    quantity. A single review is binary: the implied half-life of a correct
    answer is unbounded and of a wrong answer is zero, so after clamping the
    term is almost pure noise. Left on, it drags the coupling estimate toward
    zero; study.py measures exactly that and reports both settings.

    L-BFGS on a smooth objective. The base and coupled models differ by one
    parameter and nothing else: same features, same loss, same optimiser.
    """
    features = np.asarray(features, dtype=float)
    delta_days = np.asarray(delta_days, dtype=float)
    observed = np.clip(np.asarray(observed, dtype=float), MIN_P, MAX_P)
    prereq = np.ones_like(delta_days) if prereq_recall is None else np.asarray(prereq_recall, dtype=float)

    rng = np.random.default_rng(seed)
    start = rng.normal(0, 0.1, features.shape[1])
    # Start from a half-life near the median gap, so the optimiser does not
    # begin somewhere the exponential has already saturated.
    start[0] = np.log2(max(np.median(delta_days), MIN_HALF_LIFE))
    params = np.append(start, 0.0) if couple else start

    result = minimize(
        _loss, params, args=(features, delta_days, observed, prereq, alpha, lam, couple),
        method="L-BFGS-B", options={"maxiter": 500},
        bounds=([(None, None)] * features.shape[1] + [(-3.0, 3.0)]) if couple else None,
    )
    theta = result.x[:-1] if couple else result.x
    gamma = float(result.x[-1]) if couple else 0.0
    return Fitted(theta=theta, gamma=gamma, feature_names=tuple(feature_names),
                  alpha=alpha, lam=lam, iterations=int(result.nit), loss=float(result.fun))


def mean_absolute_error(predicted, observed) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(observed))))


def auc(scores, labels) -> float:
    """Area under the ROC curve, by rank. Ties get the average rank."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(int)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied groups so a constant predictor scores 0.5.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
