"""Borne HAUTE par methode duale (Andersen-Broadie, version multi-exercice).

Principe : a toute martingale M on associe la borne superieure
    V <= E[ max_politiques ( payoff - M ) ]
La martingale est construite par sous-simulations le long de la politique
LSMC. Couteux, mais c'est l'atout du projet : on affiche un intervalle
[borne basse, borne haute] et un duality gap en pourcentage, plutot qu'un
chiffre unique.

A IMPLEMENTER (jalon 3) :
  - increments de martingale par sous-simulation depuis chaque noeud
  - controle du nombre de sous-trajectoires (le gap se referme lentement)
  - reporting du gap relatif et de son erreur-type
"""

from __future__ import annotations


class DualUpperBound:
    def __init__(self, contract, policy, n_subpaths: int = 200) -> None:
        self.contract = contract
        self.policy = policy
        self.n_subpaths = n_subpaths

    def price(self, model, n_paths: int, seed: int = 0):
        raise NotImplementedError("Jalon 3")
