"""Serializable probability wrappers and actual equal-weight soft voting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def probability_logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


@dataclass
class ProbabilityModel:
    name: str
    pipeline: object
    calibrator: object | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, predictors):
        raw = self.pipeline.predict_proba(predictors)[:, 1]
        positive = self.calibrator.predict_proba(probability_logit(raw))[:, 1] if self.calibrator is not None else raw
        return np.column_stack([1 - positive, positive])

    def predict(self, predictors):
        return (self.predict_proba(predictors)[:, 1] >= 0.5).astype(int)


@dataclass
class SoftVotingEnsemble:
    members: dict
    weights: tuple = (1 / 3, 1 / 3, 1 / 3)
    name: str = "Hybrid Ensemble"

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, predictors):
        weights = np.asarray(self.weights, dtype=float)
        if len(weights) != len(self.members) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError("Ensemble weights must be non-negative and sum to one.")
        positive = sum(weight * member.predict_proba(predictors)[:, 1] for weight, member in zip(weights, self.members.values()))
        return np.column_stack([1 - positive, positive])

    def predict(self, predictors):
        return (self.predict_proba(predictors)[:, 1] >= 0.5).astype(int)
