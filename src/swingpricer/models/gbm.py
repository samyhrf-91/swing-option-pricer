"""Mouvement brownien geometrique.

Ce modele n'est PAS un modele energie. Il n'existe ici que pour disposer de
cas limites a solution analytique (Black-76) et valider les moteurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PriceModel


@dataclass
class GBM(PriceModel):
    s0: float
    vol: float
    drift: float = 0.0

    def simulate(self, dates: np.ndarray, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dates = np.asarray(dates, dtype=float)
        dt = np.diff(np.concatenate(([0.0], dates)))
        z = rng.standard_normal((n_paths, len(dates)))
        increments = (self.drift - 0.5 * self.vol**2) * dt + self.vol * np.sqrt(dt) * z #renvoie un tableau avec tout les prix à toutes les dates et tout les chemins
        return self.s0 * np.exp(np.cumsum(increments, axis=1))

    def forward(self, t: float) -> float:
        return self.s0 * np.exp(self.drift * t)
