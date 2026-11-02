"""Proprietes structurelles : le prix doit etre monotone.

Decroissant en K, croissant en Q_max, croissant en volatilite.
Un pricer peut passer les cas limites et violer ces proprietes dans le
regime intermediaire : c'est la que se logent les erreurs d'interpolation
sur la grille de volume.
"""

import pytest
import numpy as np
#pytestmark = pytest.mark.skip(reason="Jalon 1 non implemente")
from swingpricer.contracts.swing import SwingContract
from swingpricer.engines.lsmc import LSMCSwingEngine
from swingpricer.models.gbm import GBM

N_DATES = 4
Q_MAX = 1.0
S0 = 20.0
K = 20.0
VOL = 0.40
RATE = 0.03
DATES = np.arange(1, N_DATES +1)/N_DATES

MODEL = GBM(s0 = S0, vol = VOL, drift = RATE)


def test_decroissant_en_strike():

    contract1 = SwingContract(DATES, 18, 0.0, Q_MAX, 0, N_DATES*Q_MAX, 0.0, RATE)
    contract2 = SwingContract(DATES, 24, 0.0, Q_MAX, 0, N_DATES*Q_MAX, 0.0, RATE)

    price_1 = LSMCSwingEngine(contract1).price(MODEL, 20_000, 20_000, seed=7).price
    price_2 = LSMCSwingEngine(contract2).price(MODEL, 20_000, 20_000, seed=7).price
    assert price_1 >= price_2 


def test_croissant_en_Q_max():

    contract1 = SwingContract(DATES, K, 0.0, Q_MAX, 0, 1, 0.0, RATE)
    contract2 = SwingContract(DATES, K, 0.0, Q_MAX, 0, 3, 0.0, RATE)
    
    price_1 = LSMCSwingEngine(contract1).price(MODEL, 20_000, 20_000, seed=7).price
    price_2 = LSMCSwingEngine(contract2).price(MODEL, 20_000, 20_000, seed=7).price
    assert price_1 < price_2


def test_croissant_en_volatilite():

    contract = SwingContract(DATES, K, 0.0, Q_MAX, 0, N_DATES*Q_MAX, 0.0, RATE)
    MODEL_1 = GBM(s0 = S0, vol = VOL - 0.15, drift = RATE)
    MODEL_2 = GBM(s0 = S0, vol = VOL + 0.15, drift = RATE)

    price_1 = LSMCSwingEngine(contract).price(MODEL_1, 20_000, 20_000, seed=7).price
    price_2 = LSMCSwingEngine(contract).price(MODEL_2, 20_000, 20_000, seed=7).price

    assert price_1 < price_2
