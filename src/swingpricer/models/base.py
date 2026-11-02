"""Interface commune a tous les modeles de prix spot."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PriceModel(ABC):
    """Tout modele expose la meme signature de simulation.

    Les moteurs de pricing ne doivent jamais dependre du modele concret :
    c'est ce qui permet de rejouer la meme batterie de validation sur GBM,
    OU un facteur et Schwartz-Smith sans toucher au code des moteurs.
    """

    @abstractmethod
    def simulate(self, dates: np.ndarray, n_paths: int, seed: int | None = None) -> np.ndarray:
        """Renvoie un tableau (n_paths, len(dates)) de prix spot simules."""

    @abstractmethod
    def forward(self, t: float) -> float:
        """Prix forward vu de t=0 pour livraison en t. Sert aux tests limites."""
