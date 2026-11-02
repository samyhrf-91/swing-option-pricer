"""Bases de fonctions pour l'estimation de la valeur de continuation.

Garder la base modeste (degre 3 a 4). Au-dela, le sur-ajustement fait remonter
artificiellement la borne basse : c'est un test a mener explicitement dans le
notebook de validation (sensibilite du prix au degre).
"""

from __future__ import annotations

import numpy as np


def polynomial_basis(s, degree=3) -> np.ndarray:
    """Base polynomiale, centree-reduite pour le conditionnement.

    Diviser par la moyenne ne suffit pas : avec S autour de 20, les
    colonnes 1, x, x^2, x^3 sont alors toutes concentrees autour de 1
    et quasi colineaires. Centrer sur la moyenne et reduire par
    l'ecart-type separe les colonnes.
    """
    s = np.asarray(s, dtype=float)
    mu, sigma = s.mean(), s.std()
    x = (s - mu) / (sigma if sigma > 0 else 1.0)
    return np.vstack([x**k for k in range(degree + 1)]).T


def laguerre_basis(s: np.ndarray, degree: int = 3) -> np.ndarray:
    """Base de Laguerre ponderee, celle de l'article original de 2001."""
    s = np.asarray(s, dtype=float)
    scale = np.mean(s) if np.mean(s) != 0 else 1.0
    x = s / scale
    w = np.exp(-x / 2.0)
    cols = [np.ones_like(x)]
    for k in range(1, degree + 1):
        cols.append(w * np.polynomial.laguerre.Laguerre.basis(k)(x))
    return np.vstack(cols).T


def fit_continuation(basis: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Moindres carres robustes (lstsq) : renvoie les coefficients."""
    coef, *_ = np.linalg.lstsq(basis, np.asarray(y, dtype=float), rcond=None)
    return coef
