"""Solutions fermees pour les cas limites de la batterie de validation.

Ces formules sont la reference contre laquelle le LSMC est teste. Elles ne
servent jamais au pricing du contrat reel.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black76_call(forward: float, strike: float, vol: float, t: float, df: float = 1.0) -> float:
    """Call europeen sur forward."""
    if t <= 0 or vol <= 0:
        return df * max(forward - strike, 0.0)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    return df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def strip_of_calls(contract, model, vol: float) -> float:
    """Cas limite 1 : q_min=0, Q_max=N*q_max, pas de Q_min.

    Les dates se decouplent : la valeur est la somme de N calls europeens
    de nominal q_max.
    """
    total = 0.0
    for t in contract.exercise_dates:
        total += contract.q_max * black76_call(
            model.forward(t), contract.strike, vol, t, contract.discount(t)
        )
    return total


def forced_strip(contract, model) -> float:
    """Cas limite 2 : Q_min = Q_max = N*q_max, exercice force.

    La valeur est un strip de forwards, sans optionalite.
    """
    total = 0.0
    for t in contract.exercise_dates:
        total += contract.q_max * contract.discount(t) * (model.forward(t) - contract.strike)
    return total
