"""Convergence : nombre de trajectoires, finesse de la grille de volume,
degre de la base de regression.

Le test sur le degre est le plus instructif : au-dela de 4, la borne basse
remonte artificiellement (sur-ajustement). A documenter dans la note.
"""

import pytest
import numpy as np
#pytestmark = pytest.mark.skip(reason="Jalon 2 non implemente")
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

def test_convergence_en_trajectoires():
    contract = SwingContract(DATES, K, 0.0, Q_MAX, 0, 2.5, 0.0, RATE)

    r1 = LSMCSwingEngine(contract).price(MODEL, 20_000, 20_000, seed=7)
    r2 = LSMCSwingEngine(contract).price(MODEL, 80_000, 80_000, seed=7)

    tol = 3 * np.sqrt(r1.stderr**2 + r2.stderr**2)
    assert abs(r1.price - r2.price) < tol

def test_erreur_type_decroit_en_racine():
    """Quadrupler les trajectoires doit diviser l'erreur-type par ~2."""
    c = SwingContract(DATES, K, 0.0, Q_MAX, 0.0, N_DATES * Q_MAX, 0.0, RATE)
    engine = LSMCSwingEngine(c)

    r1 = engine.price(MODEL, 10_000, 10_000, seed=7)
    r2 = engine.price(MODEL, 40_000, 40_000, seed=7)

    assert 1.7 < r1.stderr / r2.stderr < 2.3

def test_convergence_en_grille_de_volume():
    c = SwingContract(DATES, K, 0.0, Q_MAX, 0.0, 2.3, 0.0, RATE)
    
    prices = [
        LSMCSwingEngine(c, n_volume_nodes=n).price(MODEL, 20_000, 20_000, seed=7).price
        for n in (11, 21, 41, 81)
    ]

    print(f"\n11 noeuds: {prices[0]:.4f} | 21: {prices[1]:.4f} | 41: {prices[2]:.4f} | 81: {prices[3]:.4f}")
    print(f"ecarts successifs: {abs(prices[1] - prices[0]):.4f} -> {abs(prices[2] - prices[1]):.4f} -> {abs(prices[3] - prices[2]):.4f}")


    assert abs(prices[2] - prices[1]) <= abs(prices[1] - prices[0])

def test_sensibilite_au_degre_de_base():
    """Le biais optimiste in-sample croit avec le degre, a condition que
    l'echantillon de calage soit assez petit pour etre sur-ajuste.

    Calage volontairement reduit a 1 000 trajectoires : avec 100 000, le
    rapport observations/parametres est trop favorable et le phenomene
    n'apparait pas, meme au degre 12 (voir notebooks/03_validation.ipynb).
    Jeu d'evaluation large, pour que le prix out-of-sample soit precis.
    """
    c = SwingContract(DATES, K, 0.0, Q_MAX, 0.0, 2.5, 0.0, RATE)
    paths_fit = MODEL.simulate(DATES, 1000, seed=7)
    paths_eval = MODEL.simulate(DATES, 100_000, seed=10_007) #

    gaps = []
    for d in (2, 3, 5, 8, 12):
        engine = LSMCSwingEngine(c, degree=d)
        policy = engine.fit_policy(paths_fit) #on n'appelle pas price car price renvoit le prix out of sample
        p_in, _ = engine.evaluate_policy(policy, paths_fit) #prix in-sample
        p_out, _ = engine.evaluate_policy(policy, paths_eval) #prix out-sample
        gaps.append(p_in - p_out)

        print(f"degre {d}: in={p_in:.4f} out={p_out:.4f} ecart={p_in - p_out:+.4f} "
                f"({(p_in - p_out) / p_out:+.2%})")

    assert gaps[-1] > gaps[0]
    
