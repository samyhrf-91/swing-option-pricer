"""Modele de Schwartz-Smith a deux facteurs.

log S_t = chi_t + xi_t
  chi : facteur court terme, OU de moyenne nulle (deviations transitoires)
  xi  : facteur long terme, mouvement brownien arithmetique (niveau d'equilibre)
correlation rho entre les deux browniens.

C'est le palier qui rend le projet credible cote energie : il reproduit
l'effet Samuelson et une structure par terme de volatilite realiste.

A IMPLEMENTER : simulation jointe, formule fermee du forward, et la
decomposition de la volatilite par maturite a afficher dans le notebook 04.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import PriceModel


@dataclass
class SchwartzSmith(PriceModel):
    s0: float
    kappa: float
    sigma_chi: float
    mu_xi: float
    sigma_xi: float
    rho: float = 0.0

    def simulate(self, dates, n_paths, seed=None):
        raise NotImplementedError("Jalon 4 : simulation jointe (chi, xi)")

    def forward(self, t):
        raise NotImplementedError("Jalon 4 : formule fermee du forward")
