# Swing option pricer

A Least Squares Monte Carlo pricer for gas swing options, with an
analytical validation battery and a model validation note.

The point of the project is not to produce a price. It is to produce a
price that is **bounded and tested**: a swing option has no closed-form
value, so the only way to establish that a pricer is correct is to
confront it with contracts whose value is known, and to say explicitly
what the resulting evidence does and does not cover.

## The instrument

At each exercise date $t_1, \dots, t_N$ the holder withdraws a volume
$q_i \in [q_{\min}, q_{\max}]$ at a fixed strike $K$, subject to a
cumulative constraint $Q_{\min} \le \sum_i q_i \le Q_{\max}$. A shortfall
against $Q_{\min}$ is either forbidden or charged at a linear penalty.

This is a stochastic control problem with two state variables: the spot
price $S$, exogenous and simulated, and the cumulative volume $Q$,
controlled and discretised on a grid.

## Method

**Lower bound, by LSMC.** The continuation value is not observable in a
Monte Carlo framework. It is estimated by least-squares regression of
future cash flows on a basis of functions of the current price, one
regression per (date, volume node) pair.

Two rules structure the implementation.

The regression is used to **decide**; the value propagated backwards is
the value **realised** along each path, not the regression prediction.
Propagating the prediction would stack regression errors across the $N$
dates.

The policy is fitted on one set of paths and evaluated on a second,
**independent** set. Without this separation the estimate exceeds the
true value and is no longer a lower bound. This is a deliberate
departure from the original Boogert and de Jong formulation, and the
cost of that formulation is measured: 1.5% to 2.6% on a 1 000-path
fitting sample, below 0.12% at 100 000.

**Shape of the optimum.** With a payoff linear in $q$ and a
piecewise-linear interpolant in volume, the objective is concave and
piecewise affine, with kinks at the grid nodes. The optimum therefore
lies in a finite candidate set: the two admissible bounds plus the
reachable grid nodes. No bang-bang property is assumed; such a property
requires a compatibility condition between local and global constraints
(Bardou, Bouthemy and Pagès) which is not imposed here.

## Validation

Twenty tests. The approach is to degenerate the contract into objects
whose value is known in closed form, each preceded by a test verifying
that the degeneracy actually occurs.

| Limit case | Degeneracy | Reference |
|---|---|---|
| Forced strip | Admissible set reduced to a singleton | Strip of forwards |
| Strip of calls | Dates decoupled | Sum of Black-76 calls |
| European call | Single date, no recursion | Black-76 |
| Bermudan, no convenience yield | Early exercise never optimal | Black-76 at $t_N$ |

Discrepancies are reported in standard errors, not percentages: it is
the only scale that says whether a difference is significant. The strip
of calls comes in at −0.36 σ.

Beyond the limit cases: monotonicity in strike, cap and volatility, with
common random numbers; Monte Carlo convergence at the theoretical
$-1/2$ slope; volume grid refinement; sensitivity to the regression
degree at two sample sizes.

**Exercise policy.** Aggregate prices say whether the engine is right;
the policy says whether it is right for the right reason. Near the
exercise boundary the two decisions are nearly equivalent in value, so a
wrong decision costs little and an erratic policy can coexist with an
accurate price. On the Bermudan contract with a convenience yield, the
boundary decreases monotonically from 28.3 to the strike at maturity,
and the price beats the best fixed-date strategy by **+3.3 σ**. That gap
is the value of adapting the decision to observed information, and it is
the first quantity in the project whose magnitude depends on the quality
of the regression.

## What this does not establish

- **No analytical reference exists in the operating regime.** Every
  limit case degenerates the contract. A real swing, where the global
  cap binds without forcing every decision, is covered only by
  structural properties.
- **The valuation is one-sided.** A lower bound with no companion upper
  bound, so the distance to the true value is unknown.
- **The price model is out of scope of use.** Validation runs on a GBM,
  which is not a gas price model: no mean reversion, no seasonality,
  flat volatility term structure. It is used because it gives access to
  closed-form references. The engine is validated; the valuation is not
  transferable.

Full detail in `docs/model_validation_note.md`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest -q
```

## Roadmap

1. **Trinomial tree.** The only route to a reference in the operating
   regime, by a method independent of Monte Carlo. One-factor models
   only, by construction.
2. **Dual upper bound** (Andersen and Broadie). Closes the bracket and
   makes the primal-dual gap available as an overall quality measure.
3. **Mean-reverting models.** One-factor Ornstein-Uhlenbeck with
   seasonality, then Schwartz-Smith two-factor. A nested test comes free:
   zeroing the short-term volatility in Schwartz-Smith must reproduce the
   GBM exactly.
4. **Greeks and hedging.** Delta by finite differences with common random
   numbers, and the residual P&L of a delta hedge on forwards.

Electricity as an underlying is treated at step 3 as a separate section,
not as a change of dataset: non-storability breaks the spot-forward
arbitrage relation, and price spikes make a log-price specification
unusable.

## References

- Longstaff and Schwartz (2001), *Valuing American Options by Simulation*
- Boogert and de Jong (2008), *Gas Storage Valuation Using a Monte Carlo Method*
- Schwartz and Smith (2000), *Short-Term Variations and Long-Term Dynamics in Commodity Prices*
- Andersen and Broadie (2004), *A Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options*
- Jaillet, Ronn and Tompaidis (2004), *Valuation of Commodity-Based Swing Options*
