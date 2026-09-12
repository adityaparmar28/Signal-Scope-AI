"""
SignalScope Probability Calibration & Threshold Logic
Implements Temperature Scaling and honest 3-tier verdicts.
"""

import torch
import torch.nn as nn
import numpy as np


class TemperatureCalibrator:
    """
    Applies temperature scaling to raw model logits to align predicted probabilities
    with empirical true frequencies, preventing overconfident misclassifications.
    """
    def __init__(self, temperature=1.0, low_threshold=0.35, high_threshold=0.65):
        self.temperature = float(temperature)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def calibrate_probability(self, logit: float) -> float:
        """Applies learned temperature scaling to a scalar logit."""
        scaled_logit = logit / max(self.temperature, 1e-4)
        prob = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(prob)

    def get_verdict(self, prob: float) -> dict:
        """
        Maps a calibrated probability to a responsible verdict string
        avoiding definitive accusations (Section 1 & 3.1 compliance).
        """
        if prob > self.high_threshold:
            category = "Likely AI-Generated"
            confidence = prob
            recommendation = "Visual & structural cues strongly align with generative synthesis."
        elif prob < self.low_threshold:
            category = "Likely Real Photo"
            confidence = 1.0 - prob
            recommendation = "Sensor noise, optics, and texture patterns are consistent with authentic camera captures."
        else:
            category = "Inconclusive"
            confidence = 1.0 - 2.0 * abs(prob - 0.5)  # uncertainty measure
            recommendation = "Forensics cues are ambiguous or degraded. Secondary human verification advised."

        return {
            "verdict": category,
            "calibrated_probability": round(prob, 4),
            "display_confidence": round(confidence, 4),
            "recommendation": recommendation,
        }

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        """
        Optimizes temperature T on a held-out validation set using negative log-likelihood.
        """
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.float32)
        temp_param = nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = torch.optim.LBFGS([temp_param], lr=0.01, max_iter=50)

        loss_fn = nn.BCEWithLogitsLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = loss_fn(logits_t / temp_param, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        self.temperature = float(temp_param.item())
        return self.temperature
