"""Batterie de cas limites a solution connue. Coeur de la validation.

| Cas                                          | Reference attendue        |
|----------------------------------------------|---------------------------|
| q_min=0, Q_max=N*q_max, pas de Q_min          | strip de calls (Black-76) |
| Q_min=Q_max=N*q_max                           | strip de forwards         |
| N=1, Q_max=q_max                              | call europeen             |
| Q_max=q_max, N>1                              | bermudeenne -> arbre 1F   |

Ces tests echouent tant que les jalons 1 a 2 ne sont pas faits : c'est voulu,
ils servent de definition d'achevement.
"""
import numpy as np
import pytest

#pytestmark = pytest.mark.skip(reason="Jalons 1-2 non implementes")

from swingpricer.contracts.swing import SwingContract
from swingpricer.engines.analytic import forced_strip, strip_of_calls
from swingpricer.engines.lsmc import LSMCSwingEngine
from swingpricer.models.gbm import GBM


N_DATES = 4
Q_MAX = 1.0
S0 = 20.0
K = 20.0
VOL = 0.40
RATE = 0.03
DATES = np.arange(1, N_DATES +1)/N_DATES

#drift = RATE: simulation sous la mesure risque-neutre
MODEL = GBM(s0 = S0, vol = VOL, drift = RATE)

def contract(Qmin, Qmax):
    return SwingContract(DATES, K, 0.0, Q_MAX, Qmin, Qmax, 0.0, RATE)


# --- strip de forwards : exercice force, valeur deterministe --------

FORCED = contract(N_DATES * Q_MAX, N_DATES * Q_MAX)

def test_forced_singleton():
    """Prealable : l'ensemble admissible doit etre reduit a {q_max}."""
    for i in range(N_DATES):
        lo, hi = FORCED.admissible_volume(i, np.array([i * Q_MAX]))
        assert lo[0] == pytest.approx(Q_MAX)
        assert hi[0] == pytest.approx(Q_MAX)

def test_strip_de_forwards():
    engine = LSMCSwingEngine(FORCED)
    r = engine.price(MODEL, 100_000, 100_000, seed=42)
    ref = forced_strip(FORCED, MODEL)
    print(f"\nstrip de forwards : prix={r.price:.4f} ± {r.stderr:.4f} | ref={ref:.4f} "
          f"| ecart={(r.price - ref) / r.stderr:+.2f} sigma")
    assert ref - 3 * r.stderr <= r.price <= ref + 3 * r.stderr


# --- strip de calls : dates decouplees -----------------------------

FREE = contract(0.0, N_DATES * Q_MAX)

def test_free_sans_contrainte():
    """Prealable : aucune contrainte ne doit mordre, quel que soit Q."""
    for i in range(N_DATES):
        lo, hi = FREE.admissible_volume(i, np.linspace(0.0, i * Q_MAX, 5))
        assert np.allclose(lo, 0.0)
        assert np.allclose(hi, Q_MAX)

def test_strip_de_calls():
    engine = LSMCSwingEngine(FREE)
    r = engine.price(MODEL, 40_000, 40_000, seed=42)
    ref = strip_of_calls(FREE, MODEL, VOL)
    print(f"\nstrip de calls : prix={r.price:.4f} ± {r.stderr:.4f} | ref={ref:.4f} "
          f"| ecart={(r.price - ref) / r.stderr:+.2f} sigma")
    assert ref - 3 * r.stderr <= r.price <= ref + 3 * r.stderr

def contract_at(i):
    """Contrat a une seule date : la strategie 'exercer en t_i, quoi qu'il arrive'."""
    return SwingContract(DATES[i:i+1], K, 0.0, Q_MAX, 0.0, Q_MAX, 0.0, RATE)

def test_call_europeen_une_date():
    """N=1 : pas de recursion. Isole le payoff et l'actualisation."""
    one = contract_at(0)
    r = LSMCSwingEngine(one).price(MODEL, 40_000, 40_000, seed=42)
    ref = strip_of_calls(one, MODEL, VOL)
    print(f"\ncall europeen : prix={r.price:.4f} ± {r.stderr:.4f} | ref={ref:.4f} "
          f"| ecart={(r.price - ref) / r.stderr:+.2f} sigma")
    assert ref - 3 * r.stderr <= r.price <= ref + 3 * r.stderr

BERM = contract(0.0, Q_MAX)

def test_berm_un_seul_exercice():
    """Prealable : apres un exercice plein, plus rien n'est possible."""
    for i in range(1, N_DATES):
        _, hi = BERM.admissible_volume(i, np.array([Q_MAX]))
        assert hi[0] == pytest.approx(0.0)
#def test_bermudeenne_vs_arbre():
    # V = q_max*sup_to E[exp(-r*tho)(S_tho - K)+] problème d'arret optimal
    # si la régression est fausse, la polotique est mauvaise et le prix tombe

MODEL_CY = GBM(s0=S0, vol=VOL, drift=RATE - 0.08)
def test_bermudeenne_encadree():
    """Sans arbre, on encadre.

    Borne basse : exercer a une date fixe est une strategie admissible,
    donc la bermudeenne vaut au moins la meilleure d'entre elles.
    Borne haute : le strip de calls donne N droits au lieu d'un, il domine.
    """
    engine = LSMCSwingEngine(BERM)
    r = engine.price(MODEL_CY, 40_000, 40_000, seed=42)

    lower = max(
        strip_of_calls(contract_at(i), MODEL_CY, VOL) for i in range(N_DATES)
    )
    upper = strip_of_calls(FREE, MODEL_CY, VOL)

    print(f"\nbermudeenne : {r.price:.4f} ± {r.stderr:.4f} | "
          f"lower={lower:.4f} (+{(r.price - lower) / r.stderr:.1f} sigma) | "
          f"upper={upper:.4f}")

    assert lower - 3 * r.stderr <= r.price <= upper + 3 * r.stderr


