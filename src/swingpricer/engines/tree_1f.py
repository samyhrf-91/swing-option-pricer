"""Benchmark par arbre / grille, valable uniquement en modele 1 facteur.

Sert a valider le LSMC sur le cas ou la solution numerique "exacte" est
accessible : arbre trinomial sur le log-prix x grille de volume cumule,
programmation dynamique arriere.

Si le LSMC et l'arbre ne convergent pas vers la meme valeur en 1 facteur,
le probleme vient du LSMC, pas du modele.

A IMPLEMENTER (jalon 2).
"""

from __future__ import annotations


class TreeSwingEngine:
    def __init__(self, contract, n_steps: int = 200, n_volume_nodes: int = 51) -> None:
        self.contract = contract
        self.n_steps = n_steps
        self.n_volume_nodes = n_volume_nodes

    def price(self, model):
        raise NotImplementedError("Jalon 2")
