"""Sensibilites et couverture.

Regle : toujours utiliser des nombres aleatoires communs (meme seed) entre le
cas de base et le cas choque. Sans cela le bruit Monte Carlo domine largement
la sensibilite et le delta est inexploitable.

A IMPLEMENTER (jalon 5) :
  - delta par differences finies centrees sur S0, CRN
  - vega par bump de volatilite
  - strategie de couverture delta sur forwards, P&L residuel a maturite
"""

from __future__ import annotations


def delta_fd(engine, model, bump: float = 0.01, seed: int = 0):
    raise NotImplementedError("Jalon 5")


def hedge_pnl(engine, model, n_paths: int = 10_000, seed: int = 0):
    raise NotImplementedError("Jalon 5")
