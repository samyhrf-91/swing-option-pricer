# Pricing et validation d'une option swing sur commodités

Implémentation d'un pricer d'option swing (gaz / électricité) par Least Squares
Monte Carlo, encadré par une borne supérieure duale, et accompagné d'une note
de validation de modèle.

L'objectif n'est pas de produire un prix, mais de produire un prix **encadré et
testé**. Un pricer qui sort un chiffre unique sans intervalle de confiance ni
batterie de cas limites n'est pas exploitable en production.

## Le produit

Une option swing donne le droit de prélever, à chaque date d'exercice
`t_1 ... t_N`, un volume `q_i` compris entre `q_min` et `q_max` au prix `K`,
sous contrainte de volume cumulé `Q_min <= sum q_i <= Q_max`. Un déficit par
rapport à `Q_min` est pénalisé linéairement.

C'est un problème de **contrôle stochastique** à deux variables d'état : le
prix spot `S` et le volume déjà consommé `Q`.

## Méthode

**Borne basse — LSMC.** La valeur de continuation `E[V_{i+1} | F_i]` n'est pas
observable en Monte Carlo. Elle est estimée par régression des cash-flows
futurs sur une base de fonctions de `S_i`, en coupe transversale, avec une
régression par niveau de grille de volume.

Deux règles structurantes :

- La régression sert à **décider**, pas à valoriser. La politique est estimée
  sur un jeu de trajectoires, puis rejouée sur un jeu **indépendant**. Sans
  cette séparation le prix est biaisé vers le haut et n'est plus une borne
  basse.
- Toute politique sous-optimale donne une valeur inférieure à l'optimum, donc
  le prix LSMC est bien une borne **inférieure**.

**Propriété bang-bang.** Le payoff étant linéaire en `q` et sans coût
d'exercice, la fonction valeur est concave par morceaux en `Q` et l'optimum est
atteint en `q_lo` ou `q_hi`, sauf lorsque la contrainte terminale impose un
volume intermédiaire. L'optimisation par nœud se réduit donc à une comparaison
binaire. *(Démonstration à rédiger ici — c'est l'argument qu'un validateur veut
voir écrit.)*

**Borne haute — dualité.** Martingale construite par sous-simulations le long
de la politique LSMC (Andersen-Broadie, version multi-exercice). Le livrable
est l'intervalle `[borne basse, borne haute]` et le *duality gap* relatif.

## Modèles de prix

| Modèle | Rôle |
|---|---|
| GBM | Uniquement les cas limites analytiques. Ce n'est pas un modèle énergie. |
| OU 1 facteur + saisonnalité | Premier palier crédible. Markovien en dim. 1 → benchmark par arbre possible. |
| Schwartz-Smith 2 facteurs | Effet Samuelson, structure par terme de volatilité réaliste. |

## Batterie de validation

| Cas limite | Référence attendue |
|---|---|
| `q_min=0`, `Q_max=N·q_max`, pas de `Q_min` | Dates découplées → strip de calls (Black-76) |
| `Q_min=Q_max=N·q_max` | Exercice forcé → strip de forwards |
| `N=1`, `Q_max=q_max` | Call européen |
| `Q_max=q_max`, `N>1` | Bermudéenne → arbre 1 facteur |

Complétée par : monotonie en `K`, `Q_max` et volatilité ; convergence en nombre
de trajectoires et en finesse de grille de volume, avec barres d'erreur Monte
Carlo ; sensibilité au degré de la base de régression ; stabilité du duality
gap.

## Données

Sources publiques : ENTSO-E Transparency Platform (prix day-ahead, charge,
production par filière), RTE éCO2mix, EIA (Henry Hub).

**Limite assumée :** la courbe forward TTF n'est pas publique et est
reconstruite de façon synthétique. Cette approximation est documentée dans la
note de validation plutôt que dissimulée.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .[dev]
pytest -q
```

## Jalons

1. Contrat + ensemble admissible + GBM + LSMC → premier prix. Cas limites analytiques.
2. Séparation in/out-of-sample, convergence, benchmark arbre 1 facteur.
3. Borne duale et intervalle d'encadrement.
4. Modèle OU puis Schwartz-Smith, calibration sur données réelles.
5. Grecques, couverture delta, rédaction de la note de validation.

## Limites du modèle

*(Section à compléter au fil du projet — c'est elle qui fera la différence
auprès d'un lecteur issu d'une équipe de validation.)*

- Pas de coûts de transaction ni de contrainte de liquidité
- Courbe forward reconstruite
- Pas de risque de modèle quantifié sur la corrélation en multi-facteurs
- Indexation du strike non traitée (strike fixe)

## Références

- Longstaff & Schwartz (2001), *Valuing American Options by Simulation*
- Andersen & Broadie (2004), *A Primal-Dual Simulation Algorithm*
- Schwartz & Smith (2000), *Short-Term Variations and Long-Term Dynamics*
- Boogert & de Jong (2008), *Gas Storage Valuation Using a Monte Carlo Method*
