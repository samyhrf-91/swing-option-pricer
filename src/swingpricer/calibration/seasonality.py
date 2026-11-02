"""Estimation de la composante saisonniere deterministe du log-prix.

Decomposition usuelle : log S_t = f(t) + X_t, avec f harmonique annuelle
(+ eventuellement hebdomadaire pour l'electricite) et X_t le residu
stochastique a modeliser.

A IMPLEMENTER (jalon 4) : moindres carres sur harmoniques, puis calibration
de kappa et sigma sur le residu (regression AR(1) ou maximum de vraisemblance).
"""

from __future__ import annotations


def fit_seasonality(dates, log_prices, n_harmonics: int = 2):
    raise NotImplementedError("Jalon 4")
