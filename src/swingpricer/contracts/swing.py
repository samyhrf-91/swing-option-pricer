"""Definition du contrat swing et de l'ensemble des volumes admissibles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SwingContract:
    """Swing option a exercice discret.

    Parameters
    ----------
    exercise_dates : temps (en annees) des dates d'exercice t_1 ... t_N.
    strike : prix d'exercice K (constant ; indexation a implementer plus tard).
    q_min, q_max : bornes de volume par date d'exercice.
    Q_min, Q_max : bornes de volume cumule sur la duree du contrat.
    penalty : penalite lineaire alpha appliquee au deficit (Q_min - Q_N)^+.
              Si penalty > 0, Q_min devient une contrainte souple.
    rate : taux d'actualisation continu.
    """

    exercise_dates: np.ndarray
    strike: float
    q_min: float
    q_max: float
    Q_min: float
    Q_max: float
    penalty: float = 0.0
    rate: float = 0.0

    def __post_init__(self) -> None:
        if self.q_min > self.q_max:
            raise ValueError("q_min > q_max")
        if self.Q_min > self.Q_max:
            raise ValueError("Q_min > Q_max")
        if self.Q_max > self.n_dates * self.q_max + 1e-12:
            raise ValueError("Q_max inatteignable : superieur a N * q_max")

    @property
    def n_dates(self) -> int:
        return len(self.exercise_dates)

    def admissible_volume(self, i: int, Q):
        """Bornes [q_lo, q_hi] a la date d'indice i (0-base) pour un volume cumule Q.

        La borne basse tient compte du volume qu'il reste obligatoirement a
        consommer pour atteindre Q_min compte tenu des dates restantes ; la
        borne haute tient compte du plafond global Q_max.

        NOTE : source d'erreur numero 1 du projet. Testee isolement dans
        tests/test_admissible.py avant toute utilisation dans un moteur.
        """
        Q = np.asarray(Q, dtype=float)
        remaining_after = self.n_dates - i - 1  # dates restantes APRES celle-ci

        q_hi = np.minimum(self.q_max, self.Q_max - Q)

        if self.penalty > 0.0:
            # Q_min souple : le deficit est paye, pas impose.
            q_lo = np.full_like(Q, self.q_min)
        else:
            must_take = self.Q_min - Q - remaining_after * self.q_max
            q_lo = np.maximum(self.q_min, must_take)

        q_lo = np.minimum(q_lo, q_hi)
        return q_lo, q_hi

    def terminal_penalty(self, Q):
        """Penalite lineaire pour deficit de volume a maturite."""
        return self.penalty * np.maximum(self.Q_min - np.asarray(Q, dtype=float), 0.0)

    def discount(self, t: float) -> float:
        return float(np.exp(-self.rate * t))
