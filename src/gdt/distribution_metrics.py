"""Seeded alpha-precision and beta-recall for MAGNET evaluation.

This follows the EvaGeM one-class SVM protocol used by DiffeoCFM, with
explicit random states and a Nyström rank bounded by the sample count.
"""

from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import NearestNeighbors


def _curve(base: np.ndarray, test: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    base = base.reshape(len(base), -1)
    test = test.reshape(len(test), -1)
    if len(base) < 5:
        raise ValueError("Alpha/beta metrics need at least five base samples")

    distances, _ = NearestNeighbors(n_neighbors=5, n_jobs=1).fit(base).kneighbors(base)
    gamma = 1.0 / np.median(distances[:, 4]) ** 2
    transform = Nystroem(
        gamma=gamma,
        n_components=min(base.shape),
        n_jobs=1,
        random_state=random_state,
    ).fit(base)
    base_features = transform.transform(base)
    test_features = transform.transform(test)

    points = []
    for nu in 1 - np.linspace(1e-3, 1 - 1e-3, 50):
        classifier = SGDOneClassSVM(
            nu=nu,
            shuffle=True,
            fit_intercept=True,
            tol=1e-3,
            random_state=random_state,
        ).fit(base_features)
        points.append(((classifier.predict(base_features) == 1).mean(),
                       (classifier.predict(test_features) == 1).mean()))
    true_inliers, test_inliers = np.asarray(sorted(points)).T
    return true_inliers, test_inliers


def _score(base: np.ndarray, test: np.ndarray, random_state: int) -> float:
    x, y = _curve(base, test, random_state)
    if x[0] > 0:
        x = np.r_[0.0, x]
        y = np.r_[y[0], y]
    if x[-1] < 1:
        x = np.r_[x, 1.0]
        y = np.r_[y, y[-1]]
    return float(1 - 2 * np.trapz(np.abs(y - x), x))


def alpha_precision(real: np.ndarray, generated: np.ndarray, *, random_state: int = 42) -> float:
    return _score(real, generated, random_state)


def beta_recall(real: np.ndarray, generated: np.ndarray, *, random_state: int = 42) -> float:
    return _score(generated, real, random_state)
