"""Decay-window tau: predictive R^2 of a feature vs forward returns across horizons."""
from __future__ import annotations
import numpy as np


def predictivity_curve(feature, returns_by_h: dict) -> dict:
    f = np.asarray(feature, float)
    return {h: float(np.corrcoef(f, np.asarray(r, float))[0, 1] ** 2)
            for h, r in returns_by_h.items()}


def estimate_tau(curve: dict, *, frac: float = 0.3679) -> float:
    hs = sorted(curve); vals = [curve[h] for h in hs]
    thresh = frac * max(vals)
    for i in range(1, len(hs)):
        if vals[i] < thresh <= vals[i - 1]:
            x0, x1, y0, y1 = hs[i - 1], hs[i], vals[i - 1], vals[i]
            return float(x0 + (x1 - x0) * (y0 - thresh) / (y0 - y1 + 1e-12))
    return float(hs[-1])
