"""L'ensemble admissible est teste SEUL, avant tout moteur.

C'est la source d'erreur numero 1 du projet : une borne mal calculee produit
un prix plausible mais faux, et rien ne le signale en aval.
"""

import numpy as np
import pytest

from swingpricer.contracts.swing import SwingContract


def make_contract(**kw):
    base = dict(
        exercise_dates=np.array([0.25, 0.5, 0.75, 1.0]),
        strike=20.0,
        q_min=0.0,
        q_max=1.0,
        Q_min=2.0,
        Q_max=3.0,
    )
    base.update(kw)
    return SwingContract(**base)


def test_borne_haute_respecte_le_plafond_global():
    c = make_contract()
    _, q_hi = c.admissible_volume(0, np.array([2.5]))
    assert q_hi[0] == pytest.approx(0.5)


def test_borne_basse_force_le_rattrapage_en_fin_de_contrat():
    # derniere date, volume cumule 1.5, Q_min = 2.0 -> il faut prendre 0.5
    c = make_contract()
    q_lo, _ = c.admissible_volume(3, np.array([1.5]))
    assert q_lo[0] == pytest.approx(0.5)


def test_pas_de_contrainte_forcee_en_debut_de_contrat():
    c = make_contract()
    q_lo, _ = c.admissible_volume(0, np.array([0.0]))
    assert q_lo[0] == pytest.approx(0.0)


def test_penalite_rend_Q_min_souple():
    c = make_contract(penalty=5.0)
    q_lo, _ = c.admissible_volume(3, np.array([0.0]))
    assert q_lo[0] == pytest.approx(0.0)


def test_bornes_toujours_ordonnees():
    c = make_contract()
    for i in range(c.n_dates):
        Q = np.linspace(0.0, c.Q_max, 25)
        q_lo, q_hi = c.admissible_volume(i, Q)
        assert np.all(q_lo <= q_hi + 1e-12)


def test_Q_max_inatteignable_rejete():
    with pytest.raises(ValueError):
        make_contract(Q_max=10.0)
