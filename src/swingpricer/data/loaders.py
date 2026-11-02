"""Chargement des donnees de marche.

Sources publiques retenues :
  - ENTSO-E Transparency Platform : prix day-ahead, charge, production par
    filiere (necessite une cle API gratuite)
  - RTE eCO2mix : production et consommation France
  - EIA : Henry Hub

La courbe forward TTF n'est pas publique : elle est reconstruite de facon
synthetique. Cette limite doit etre ecrite explicitement dans le README et
dans la note de validation, pas dissimulee.

Les fichiers bruts vont dans data/raw/ (non versionnes), les series nettoyees
dans data/processed/.
"""

from __future__ import annotations

from pathlib import Path

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"


def load_spot(source: str = "entsoe"):
    raise NotImplementedError("Jalon 4")
