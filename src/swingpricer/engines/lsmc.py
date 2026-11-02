"""Least Squares Monte Carlo pour swing option (borne BASSE).

Structure en deux temps, a ne jamais fusionner :

  1. fit_policy()      -> estime les coefficients de regression sur un jeu de
                          trajectoires A (in-sample). Produit une POLITIQUE.
  2. evaluate_policy() -> rejoue cette politique sur un jeu de trajectoires B
                          independant. Produit le PRIX.

Si l'on evalue sur le meme jeu que celui de la regression, le prix est biaise
vers le haut et n'est plus une borne basse. C'est le premier point qu'un
validateur souleve.

Etat : (S_i, Q_i) ou Q_i est le volume cumule. Une regression par niveau de
grille de volume, d'ou la boucle imbriquee date x grille.

Propriete bang-bang : le payoff etant lineaire en q et sans cout d'exercice,
la fonction valeur est concave par morceaux en Q et l'optimum est atteint en
q_lo ou q_hi (sauf contrainte terminale). L'optimisation se reduit donc a une
comparaison binaire. Demonstration a ecrire dans le README.
"""
# %%
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
# %%

from ..contracts.swing import SwingContract
from .regression import fit_continuation, polynomial_basis


@dataclass
class LSMCResult:
    price: float
    stderr: float
    policy: list          # coefficients de regression par (date, noeud de volume)
    volume_grid: np.ndarray


class LSMCSwingEngine:
    def __init__(
        self,
        contract: SwingContract,
        n_volume_nodes: int = 41,
        degree: int = 3,
        basis_fn=polynomial_basis,
    ) -> None:
        self.contract = contract
        self.n_volume_nodes = n_volume_nodes
        self.degree = degree
        self.basis_fn = basis_fn
        self.volume_grid = np.linspace(0.0, contract.Q_max, n_volume_nodes)

    # ------------------------------------------------------------------
    def _interp_weights(self, v):
        """v : volume(s) cible(s). Renvoie (idx, w) tels que
        f(v) ≈ (1-w)·f[idx] + w·f[idx+1] sur la grille de volume."""

        grid = self.volume_grid
        dQ = grid[1] - grid[0] #delta_v
        pos = (np.asarray(v, dtype=float) - grid[0]) / dQ #position float
        idx = np.clip(np.floor(pos).astype(int), 0, len(grid) - 2) # clip pour effet de bordure
        w = np.clip(pos - idx, 0.0, 1.0) #poids

        return idx, w
    
    # ------------------------------------------------------------------
    def fit_policy(self, paths: np.ndarray) -> list:
        """Recursion arriere sur (date, noeud de volume).

        A IMPLEMENTER (jalon 1) :
          - initialiser V_N sur la grille de volume avec la penalite terminale
          - pour i = N-1 ... 0 :
              pour chaque noeud de volume Q_j :
                  regresser V_{i+1} interpole en (Q_j + q) sur base(S_i)
                  comparer q_lo et q_hi (bang-bang) -> decision
                  stocker les coefficients
          - l'interpolation en volume entre noeuds de grille est lineaire
        """
        c = self.contract
        n_paths = paths.shape[0]
        grid = self.volume_grid

        # df[i] : escompte de t_{i+1} vers t_i ; 1 a la derniere date
        # (la penalite terminale tombe en t_N)
        df = np.ones(c.n_dates) #discount factors
        df[:-1] = np.exp(-c.rate * np.diff(c.exercise_dates))

        V_next = -np.tile(c.terminal_penalty(grid)[:, None], (1, n_paths)) #ok pour fonction penalité
        policy = [None] * c.n_dates
        rows = np.arange(n_paths)

        for i in range(c.n_dates-1, -1,-1):

            S = paths[:,i] # S_i 
            #régression : toutes les continuations d'abord, les décisions ensuite
            X = self.basis_fn(S, self.degree) #régression, degré 3 par défaut
            beta, *_ = np.linalg.lstsq(X, (df[i] * V_next).T, rcond=None)   #(d+1, n_nodes) discount pour que V_i+1 soit à la date i
            policy[i] = beta

            V_curr = np.empty_like(V_next)

            for j, Q in enumerate(self.volume_grid): 

                #candidats depuis un noeud
                q_lo, q_hi = c.admissible_volume(i,np.array([Q]))
                q_lo, q_hi = float(q_lo[0]), float(q_hi[0])
                reach = grid - grid[j] #volumes atteignables en un pas (delta_v qu'on peut acheter vendre)
                inner = reach[(reach > q_lo) & (reach < q_hi)]
                candidates = np.unique(np.concatenate(([q_lo, q_hi], inner))) #ensembles des delta_v possibles après prise en compte des contraintes

                obj = np.empty((len(candidates), n_paths))

                for k, q in enumerate(candidates):
                    idx, w = self._interp_weights(grid[j] + q) 
                    C_hat = (1 - w) * (X @ beta[:, idx]) + w * (X @ beta[:, idx + 1]) #X^T*Beta, deuxième morceau du payoff
                    obj[k] = q * (S - c.strike) + C_hat #payoff complet

                best = np.argmax(obj, axis=0)  # indices des meilleurs volumes  (n_paths,)
                q_star = candidates[best] #meilleurs volumes 

                idx, w = self._interp_weights(grid[j] + q_star)          # vecteurs ici
                rows = np.arange(n_paths)
                V_future = (1 - w) * V_next[idx, rows] + w * V_next[idx + 1, rows]
                V_curr[j] = q_star * (S - c.strike) + df[i] * V_future

            V_next = V_curr

        return policy
    # ------------------------------------------------------------------
    def evaluate_policy(self, policy: list, paths: np.ndarray, return_decisions = False):
        """Rejoue la politique en marche AVANT sur des trajectoires neuves.

        Renvoie (prix, erreur-type Monte Carlo). L'erreur-type doit toujours
        etre affichee a cote du prix dans les notebooks.
        """
        #raise NotImplementedError("Jalon 1")
        c = self.contract
        n_paths = paths.shape[0]
        grid = self.volume_grid
        K = c.strike
 
        Q = np.zeros(n_paths)  # volume cumule : etat initial du contrat. On est à 0 au départ
        pv = np.zeros(n_paths) # accumulateur de cash-flows actualises

        decisions = np.empty((c.n_dates, n_paths)) if return_decisions else None

        for i in range(c.n_dates):
            S = paths[:,i]
            X = self.basis_fn(S, self.degree) #polynome, degré 3 par défaut
            beta = policy[i]

            q_lo, q_hi = c.admissible_volume(i, Q)        # vectoriel : Q varie par trajectoire
            ok = np.ones(n_paths, dtype=bool)
            candidates = [(q_lo, ok), (q_hi, ok)]

            for grid_Q in grid:                       # atterrir sur un noeud
                q = grid_Q - Q
                valid = (q >= q_lo - 1e-12) & (q <= q_hi + 1e-12)
                if valid.any():
                    # q_lo comme remplissage : la ligne est ecartee par -inf
                    candidates.append((np.where(valid, q, q_lo), valid))
                
            values = []
  
            for q, valid in candidates:
                # Valeur immédiate
                immediate = q * (S - K)

                idx, w = self._interp_weights(Q + q)
                B = (1 - w)[None, :] * beta[:, idx] + w[None, :] * beta[:, idx + 1]
                continuation = np.einsum('pk,kp->p', X, B)
                # Valeur totale
                values.append(np.where(valid, immediate + continuation, -np.inf))


            # Choisir la meilleure quantité pour chaque trajectoire
            values = np.array(values)
            quantities = np.array([q for q, _ in candidates])

            best = np.argmax(values, axis=0) #chaque trajectoire
            q_star = quantities[best, np.arange(n_paths)]
            if return_decisions:
                decisions[i] = q_star
            
            pv += c.discount(c.exercise_dates[i]) * q_star * (S - K)
            Q = Q + q_star
            assert np.all(q_lo <= q_star + 1e-9) and np.all(q_star <= q_hi + 1e-9)
            #assert np.allclose(q_star, q_lo)   # sur le contrat force uniquement
            
        pv -= c.discount(c.exercise_dates[-1]) * c.terminal_penalty(Q)
        out = (float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(n_paths)))
    
        return out + (decisions,) if return_decisions else out
    # ------------------------------------------------------------------
    def price(self, model, n_paths_fit: int, n_paths_eval: int, seed: int = 0) -> LSMCResult:
        dates = self.contract.exercise_dates
        paths_fit = model.simulate(dates, n_paths_fit, seed=seed)
        paths_eval = model.simulate(dates, n_paths_eval, seed=seed + 10_000)
        policy = self.fit_policy(paths_fit)
        price, stderr = self.evaluate_policy(policy, paths_eval)
        return LSMCResult(price, stderr, policy, self.volume_grid)
