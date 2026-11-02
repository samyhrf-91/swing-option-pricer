"""Ornstein-Uhlenbeck sur le log-prix, avec saisonnalite deterministe.

Premier palier credible cote energie : retour a la moyenne, saisonnalite
annuelle et hebdomadaire. Le modele reste markovien en dimension 1, ce qui
permet le benchmark par arbre (engines/tree_1f.py).

A IMPLEMENTER : simulation exacte du schema OU (pas d'Euler, la solution
exacte est disponible), saisonnalite f(t) additive sur le log-prix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PriceModel


@dataclass
class OneFactorOU(PriceModel):
    s0: float
    kappa: float       # vitesse de retour a la moyenne
    theta: float       # niveau de long terme (log-prix)
    vol: float
    seasonality: object | None = None   # callable f(t) -> float, additive sur log S

    def _seasonal(self, t):
        if self.seasonality is None:
            return np.zeros_like(np.asarray(t, dtype=float))
        return self.seasonality(t)

    def simulate(self, dates, n_paths, seed=None):
        raise NotImplementedError("Jalon 4 : schema exact OU + saisonnalite")

    def forward(self, t):
        raise NotImplementedError("Jalon 4 : E[S_t] sous log-normale OU")
