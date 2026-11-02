# Validation note — Swing option pricer

*Trame. Format calqué sur une note de validation interne : c'est le livrable
qui distingue ce projet d'un dépôt de code.*

---

## 1. Scope

### 1.1 Instrument
This deliverable aimed at pricing and assess risk of a gas swing option. At each date $t_1 < \dots < t_N$, the holder withdraws volume $q_i \in [q_{\min}, q_{\max}]$ \(this interval is time-dependant\)at a fixed strike $K$, subject to a cumulative constraint $Q_{\min} \le \sum_i q_i \le Q_{\max}$. If at maturity enough volume isn't withdrawn (i.e $ \sum_i q_i < Q_{\min}$ and if permissible), a linear penalty $\alpha$ is charged against the holder.

In this contract, the strike $K$ is constant over time.

### 1.2 Intended use

The model produces two outputs, both essential:

- a **valuation**, as a lower bound on the contract value since the strategy is by definition under-optimal, with its Monte Carlo standard error $\sigma$;
- an **exercise policy**, i.e. a withdrawal decision as a function of the
  observed spot price and the volume already taken.

### 1.3 Price model

A geometric Brownian motion is used throughout this validation. **This is
not a realistic model for gas prices**: it has no mean reversion, no
seasonality, and a flat volatility term structure. The rationale behind this modelling choice is to have access to closed-form references (Black-76, forward
curve), without which the engine could not be validated against anything.

Mean-reverting models (one-factor Ornstein-Uhlenbeck with seasonality,
then Schwartz-Smith two-factor) are the subject of milestone 4 and are
**not validated by this note**

### 1.4 Out of scope

| Item | Reason |
|---|---|
| Electricity as underlying | Non-storability breaks the spot-forward arbitrage relation; price spikes and negative prices make a log-price specification unusable. Treated separately at milestone 4. |
| Multi-factor price models | Validated references in this note are GBM-specific. The tree benchmark (milestone 2) is one-factor only by construction. |
| Transaction costs, liquidity constraints | Not modelled. |
| Strike indexation | Fixed strike only. |
| Upper bound on the contract value | Only the lower bound is produced today. The dual bound (milestone 3) is not covered. |
| Credit and counterparty risk | Out of scope entirely. |

### 1.5 Documents and code

Validation evidence: `notebooks/03_validation.ipynb`.
Engine: `src/swingpricer/engines/lsmc.py`.
Test battery: `tests/`, 20 tests.


## 2. Model Description

### 2.1 The control problem

Let $S_i$ denote the spot price at exercise date $t_i$ and $Q_i$ the
volume already withdrawn. The admissible set at date $i$ is

$$q_i \in \mathcal{A}_i(Q_i) = \big[q^-(i, Q_i),\; q^+(i, Q_i)\big]$$

$$q^+ = \min\big(q_{\max},\, Q_{\max} - Q_i\big), \qquad
q^- = \max\big(q_{\min},\, Q_{\min} - Q_i - (N - i - 1)\,q_{\max}\big)$$

The lower bound carries a catch-up term (i.e. $(N - i - 1)\,q_{\max}$ ): it forces withdrawal once $Q_{\min}$ can no longer be reached within the remaining dates. Under a finite penalty $\alpha$, $Q_{\min}$ becomes a soft threshold and the catch-up term is dropped from the lower bound, reappearing in the terminal condition instead.

The value function solves

$$V_i(S, Q) = \max_{q \in \mathcal{A}_i(Q)}
\Big\{ q\,(S - K) + \mathbb{E}^{\mathbb{Q}}\big[e^{-r\Delta_i}\,
V_{i+1}(S_{i+1}, Q + q) \mid S_i = S\big] \Big\}$$

with $V_{N+1}(S, Q) = -\alpha\,(Q_{\min} - Q)^+$. The contract value is
$V_1(S_0, 0)$.

No closed form exists. There are two state variables (the spot price $S$ and the cumulated volume $Q$), one of them controlled, and the conditional expectation is not observable in a Monte Carlo framework. Both difficulties are addressed numerically below.

### 2.2 Numerical method

**Volume discretisation.** The controlled state is tabulated on a grid of
$M$ nodes spanning $[0, Q_{\max}]$, with linear interpolation between
them. This has a structural consequence: with a payoff linear in $q$ and
a piecewise-linear interpolant, the objective is concave and piecewise
affine in $q$, with kinks exactly at the grid nodes. Its maximum is
therefore attained on the finite set

$$\{q^-, q^+\} \cup \{Q^{(k)} - Q : Q^{(k)} - Q \in [q^-, q^+]\}$$

and the search is restricted to those points (with $Q^{(k)}$ the $k^{th}$ node of the grid ). No bang-bang property is assumed: such a property holds under a compatibility condition between local and global constraints (Bardou, Bouthemy & Pagès) which is not
imposed here. Where the optimum is observed to sit at a bound, this is
recorded as a numerical observation, not as a theorem.

**Regression.** The conditional expectation is approximated by least
squares on a basis of functions of the current price, one regression per
(date, volume node) pair, hence $N \times M$ in total. The basis is
polynomial of degree 3, centred and scaled; without normalisation the
columns are near-collinear and the design matrix is ill-conditioned. The
$M$ regressions of a given date share the same design matrix, since the
basis depends on price only, so a single least-squares call per date
resolves all of them.

**Separation of fitting and evaluation.** The policy is estimated on one
set of paths and evaluated on a second, independent set. This is different from Boogert & de Jong (2008), who evaluate on the fitting sample; the magnitude of the resulting bias is measured in section 5.

**What propagates through the recursion.** The regression is used to
*decide*; the value propagated backwards is the value *realised* along
each path, read from the tabulated values at the next date, not the
regression prediction. Propagating the prediction would stack regression
errors across the $N$ dates and let the price drift.

**Model independence.** The engine depends on the price model only
through a `simulate` interface. The same validation battery can therefore
be replayed on the mean-reverting models of milestone 4 without modifying
the engine.

### 2.3 Why the output is a lower bound

Any suboptimal policy yields a value below the optimum. The policy
produced here is suboptimal, since it rests on an approximation of the
continuation value, so

$$\widehat V_{\text{out}} \le V^\star$$

Out-of-sample evaluation is what preserves this inequality: evaluating on
the fitting sample would let the policy exploit noise it has already
seen, and the estimate would exceed $V^\star$.

Two consequences run through this note. Maximising
$\widehat V_{\text{out}}$ is a legitimate criterion for setting numerical
parameters, since the gap $V^\star - \widehat V_{\text{out}}$ is exactly
the loss caused by an imperfect regression. And the valuation is
one-sided: an upper bound requires the dual construction of milestone 3.

### 2.4 Numerical parameters

| Parameter | Value | Criterion | Evidence |
|---|---|---|---|
| Regression degree | 3 | Out-of-sample price; cost in $O(d^2)$ over $N \times M$ regressions | §5.3 |
| Volume grid nodes | 21 | Decreasing successive differences under refinement | §5.2 |
| Fitting paths | 40 000 | Overfitting bias below measurement noise | §5.3 |
| Evaluation paths | 40 000 | Standard error of order 0.02 on the reference contract | §5.2 |

Values are justified in the sections referenced, not here.

## 3. Hypothèses et limites structurelles

Each assumption below is listed with what motivates it and what goes
wrong if it does not hold. Assumptions that the code does not enforce at
runtime are flagged explicitly: those are the ones a user can violate
without any error being raised.

### 3.1 Price model

| Assumption | Justification | Impact if violated | Enforced? |
|---|---|---|---|
| Risk-neutral measure (drift equal to the contract rate) | The engine discounts at the contract rate; consistency is required for the valuation to be arbitrage-free | Wrong price, with no structural test detecting it. A physical drift of 8% against a 3% rate shifts the forward curve by roughly 5% per year of maturity | **No.** The engine has no access to the drift through the `PriceModel` interface, and a two-factor model has two risk-neutral parameters rather than one. Detection relies on the forced-strip limit case |
| Geometric Brownian motion | Gives access to closed-form references, without which nothing could be validated | Prices are not representative of a gas contract: no mean reversion, no seasonality, flat volatility term structure | Deliberate restriction of scope |
| Deterministic interest rate | Standard at this maturity | Negligible at one year, material on a multi-year contract | No |

The GBM restriction is the most consequential item in this note. **No
result here transfers to a production valuation.** What is validated is
the engine, not the price process.

### 3.2 Contract

| Assumption | Justification | Impact if violated |
|---|---|---|
| Fixed strike | Simplification | Indexed contracts are the common case in practice. The payoff would become path-dependent through the index, adding a state variable |
| No transaction costs, no liquidity constraint | Simplification | The optimal policy would withdraw less often and the value would fall. The direction of the bias is known, its magnitude is not estimated |
| Volume bounds constant across dates | Simplification | Real contracts often carry seasonal daily caps. The admissible set would become date-dependent beyond the catch-up term already modelled |
| Linear penalty on the volume shortfall | Common specification | Stepped or capped penalties would require a different terminal condition |

### 3.3 Numerical method

| Assumption | Justification | Impact if violated |
|---|---|---|
| Linear interpolation of the value function between volume nodes | Keeps the admissible set exact, unlike snapping decisions to nodes, and makes the objective piecewise affine so the optimum lies in a finite candidate set | Introduces a discretisation bias, measured in §5.2 by successive refinement. Nearest-node rounding breaks the convergence test (§6, fault C) |
| Degree-3 polynomial basis | Out-of-sample price is flat between degrees 3 and 8; cost grows as O(d²) over N × M regressions | Degree 2 underfits by about one standard error. Degree 12 overfits once the fitting sample falls below roughly 10 000 paths |
| Continuation value is a function of the spot price alone | The price is the only exogenous state variable under a one-factor model | A two-factor model would require a basis spanning both factors |
| Fitting and evaluation samples are independent | Preserves the lower-bound property | The estimate would exceed the true value. Measured at 1.5% to 2.6% on a 1 000-path fitting sample, below 0.12% at 100 000 |

### 3.4 Assumptions deliberately not made

Two properties are commonly assumed in the literature and are not assumed
here. Their absence shapes the implementation, so both are worth stating.

**Bang-bang exercise.** A linear payoff is often taken to imply that the
optimum sits at a bound of the admissible set. This requires a
compatibility condition between local and global volume constraints
(Bardou, Bouthemy and Pagès) which is not imposed. The search therefore
covers the full finite candidate set, interior grid nodes included. On
the contracts tested the optimum is in fact always observed at a bound,
but this is recorded as a numerical observation rather than relied upon.

**In-sample valuation.** The original Boogert and de Jong formulation
evaluates on the same paths used to fit the regressions. That formulation
is not followed. The cost is one extra simulation per valuation; the
benefit is that the output is a genuine lower bound.

### 3.5 What this note does not establish

**No analytical reference exists in the regime of interest.** The four
limit cases all degenerate the contract: forced exercise, decoupled
dates, single date, single exercise right. None covers a contract where
the global cap binds without forcing every decision, which is precisely
the regime a real swing lives in. The trinomial tree of milestone 2 is
intended to close this gap.

**Only one side of the bracket is produced.** The valuation is a lower
bound with no companion upper bound, so the distance to the true value is
unknown. The dual construction of milestone 3 would supply it, and the
width of the resulting primal-dual gap is the quantity that would measure
the overall quality of the numerical scheme.

**One defect escapes the test battery.** In-sample evaluation (§6, fault
B) passes every check at the nominal sample size, because the bias it
introduces is smaller than three standard errors. Sample separation is
therefore guaranteed by construction, in the signature of `price()`,
rather than by test.

## 4. Tests performed

Twenty tests, in four families. Each is stated with what it isolates, so
that a failure localises the defect rather than merely signalling one.

### 4.1 Component tests

Run before any engine test, on the two components whose errors are
silent elsewhere.

- `admissible_volume`: upper bound against the global cap, catch-up
  behaviour at the last date, soft threshold under a finite penalty,
  ordering of the bounds across the whole grid, rejection of an
  unreachable `Q_max`.
- `_interp_weights`: exactness on an affine function tabulated on the
  grid, behaviour at both ends of the grid. This test catches index
  offsets, which otherwise produce a plausible but wrong price.

### 4.2 Limit cases

Four contract specialisations whose value is known in closed form, each
preceded by a test verifying that the degeneracy actually occurs. Without
that precondition test, a failure of the price test would be attributed
to the engine when it may lie in the contract construction.

| Case | Degeneracy | Reference | Isolates |
|---|---|---|---|
| Forced strip | Admissible set reduced to a singleton | Strip of forwards | `admissible_volume` in the fully constrained regime, multi-date discounting |
| Strip of calls | Dates decoupled | Sum of Black-76 calls | The full backward recursion, regression and volume interpolation |
| European call | Single date, no recursion | Black-76 | Payoff and discounting in isolation |
| Bermudan, no convenience yield | Early exercise never optimal | Black-76 at maturity t_N | A fifth analytical reference, on a contract the others do not cover |

### 4.3 Structural properties

Tested in the intermediate regime, where no analytical reference exists.
All use common random numbers: without a shared seed the difference
between two prices is swamped by Monte Carlo noise.

- Price decreasing in the strike. The payoff q(S − K) decreases in K
  path by path, for any q ≥ 0.
- Price increasing in Q_max. Any policy admissible under the tighter cap
  is admissible under the looser one, so the supremum over a larger set
  dominates. Both caps are kept strictly below N·q_max, otherwise the
  constraint does not bind and the test asserts nothing.
- Price increasing in volatility. The value function is convex in S, and
  an increase in volatility is a mean-preserving spread.

### 4.4 Numerical convergence

- Standard error against the number of paths, log-log, at fixed policy.
- Price against volume grid refinement, on a contract whose cap binds and
  whose Q_max is not a multiple of the grid step.
- Out-of-sample price and in-sample bias against the degree of the basis,
  at two fitting sample sizes.

---

## 5. Results

### 5.1 Limit cases

All within three standard errors of the analytical reference.
Discrepancies are reported in standard errors rather than percentages:
it is the only scale that says whether a difference is significant.

| Case | Reference | LSMC | Std. error | Gap |
|---|---|---|---|---|
| Strip of calls | 10.4230 | 10.3953 | 0.0771 | −0.36 $\sigma$ |
| Bermudan, no convenience yield | 3.4277 | 3.3917 | 0.0281 | −1.28 $\sigma$ |
| Forced strip | 1.4833 |1.4691| 0.0706 | -0.20 $\sigma$ |
| European call | 1.6630 |1.6573 |0.0134 |-0.4 $\sigma$|

### 5.2 Monte Carlo convergence and grid refinement

Standard error against path count, measured at fixed policy so that only
the evaluation sample varies: empirical slope of -0.5 against a
theoretical −1/2. The limit cases establish that the estimator converges
to the right value; this establishes that it converges at the right
rate, which no value-based test would catch. A slope away from −1/2
would indicate that the paths are not independent.

Volume grid: successive differences under refinement are 0.0191
(11 to 21 nodes), 0.0133 (21 to 41) and 0.0043 (41 to 81). The
successive ratios, 0.70 then 0.32, approach the value expected of a
linear interpolant as the grid enters its asymptotic regime; at 11 nodes
the grid is too coarse for that regime to hold.

Forty-one nodes are retained. The residual discretisation bias is then
of the order of 0.005, roughly a quarter of the Monte Carlo standard
error on this contract, so it is not the limiting source of error.
Twenty-one nodes would leave a bias comparable to the standard error
itself.

### 5.3 Degree of the regression basis

Two regimes, depending on the size of the fitting sample.

With 1 000 fitting paths and 100 000 evaluation paths, the optimistic
bias of an in-sample valuation is measurable and widens with the degree:

| Degree | In-sample | Out-of-sample | Bias |
|---|---|---|---|
| 2 | 7.6239 | 7.5106 | +1.51% |
| 3 | 7.6166 | 7.5380 | +1.04% |
| 5 | 7.6014 | 7.5357 | +0.87% |
| 8 | 7.6614 | 7.5342 | +1.69% |
| 12 | 7.7088 | 7.5153 | +2.57% |

With 100 000 fitting paths the effect disappears: the bias stays below
0.12% in absolute value and shows no trend, even at degree 12. The
observations-to-parameters ratio is too favourable for the noise to be
exploitable.

**Reading.** Degrees 3 to 8 are indistinguishable, their out-of-sample
prices differing by less than 0.1 σ. Degree 2 underfits by about one
standard error, degree 12 overfits. Within the plateau, degree 3 is
retained for cost and for robustness at smaller fitting samples.

Maximising the out-of-sample price is a legitimate criterion only
because the evaluation sample is independent: a degree that inflated the
price by exploiting fitting noise would gain nothing out of sample.
Degree 12 illustrates this directly, at 7.7088 in-sample against 7.5153
out of sample.

The 1.5% to 2.6% figures measure the error incurred by the original
Boogert and de Jong formulation. That error is negligible at the sample
size used here, which justifies the fit/evaluate split by measurement
rather than by assertion.

### 5.4 Exercise policy

Aggregate prices say whether the engine is right; the policy says
whether it is right for the right reason. Near the exercise boundary the
two decisions are nearly equivalent in value, so a wrong decision costs
little and a partially erratic policy can coexist with an accurate price.

Two regimes were examined on the Bermudan contract.

Under a drift equal to the rate, no path exercises before t_N. This is
the classical result for a call on a non-dividend-paying asset. The
contract degenerates to the European call of maturity t_N, and the test
asserts nothing about the exercise decision. This is itself a finding:
the case had to be replaced before the regression could be exercised at
all.

Under a convenience yield (drift of r − 8%), early exercise becomes
optimal. The continuation value then decreases with the date, as the set
of remaining exercise opportunities shrinks:

| Date | Continuation | Exercise rate (live paths) | Boundary |
|---|---|---|---|
| t_0 | 2.6858 | 2.6% | 28.3 |
| t_1 | 2.6726 | 9.0% | 26.7 |
| t_2 | 1.8429 | 11.3% | 24.6 |
| t_3 | 0.0000 | 22.9% | 20.0 |

The boundary decreases monotonically and meets the strike at maturity,
where the continuation value vanishes and exercise is optimal whenever
S > K. Continuation is zero at t_3 for paths that have already
exercised, confirming both the terminal condition and the exhaustion of
the exercise right.

Price: 2.6232 ± 0.0217, against a best fixed-date strategy at 2.5515,
a gap of **+3.3 σ**. This gap is the value of adapting the decision to
observed information, and it is the first quantity in this validation
whose magnitude depends on the quality of the regression.

---

## 6. Coverage of the test battery

A battery that passes says nothing about what it would catch. Three
defects were injected, each a single-line change to a correct component,
and the checks of section 4 were re-run against each.

| Defect | Nature | Caught by |
|---|---|---|
| A. Missing Itô correction in the log-price drift | Coding error. Produces E[S_t] = S₀·exp((μ + σ²/2)t) instead of S₀·exp(μt) | [to complete] |
| B. Evaluation on the fitting sample | Methodological. The original Boogert and de Jong formulation | [to complete] |
| C. Volume interpolation replaced by rounding to the node below | Coding error | [to complete] |

The informative outcome is defect B. At the nominal sample size the bias
it introduces is below three standard errors, so no check detects it.
Sample separation cannot therefore be guaranteed by test; it is
guaranteed by construction, in the signature of `price()`, which draws
two independent path sets from different seeds.

Defect A is detected by the cases that compare against an analytical
forward. This is why the forced strip, which carries no optionality and
might appear redundant, is retained: it is the cheapest check on the
consistency between the simulated process and its own forward curve.

---

## 7. Weaknesses identified

**No reference in the operating regime.** All analytical references rest
on degenerate contracts. The regime where the global cap binds without
forcing every decision, which is the regime of any real swing, is
validated only by structural properties: monotonicity, convergence,
shape of the exercise boundary. These constrain the answer without
pinning it down.

**One-sided valuation.** The output is a lower bound with no companion
upper bound, so the distance to the true value is unknown. Nothing in
this note bounds the error of the valuation itself, only its Monte Carlo
dispersion around its own expectation.

**A defect that no test detects.** See section 6, defect B.

**Price model outside the scope of use.** Validation is carried out on a
GBM, which is not a gas price model. The engine is validated; the
valuation is not transferable.

**Empirical parameter choices are contract-specific.** Degree 3 and 21
grid nodes are justified on one reference contract. Nothing establishes
that they remain appropriate for a contract with 365 dates, a binding
q_min, or a non-zero penalty.

---

## 8. Recommendations and conditions of use

### 8.1 Conditions of use as it stands

The engine may be used to value a swing contract under the following
conditions.

- One-factor price model, specified under the risk-neutral measure. The
  consistency between the model drift and the contract rate must be
  checked by the caller: the engine does not check it.
- Fixed strike, constant volume bounds, linear penalty.
- Contract parameters within the range tested: up to a few dozen
  exercise dates, Q_max at most N·q_max.
- The output is to be read as a lower bound, with its standard error
  stated alongside. A valuation reported without its standard error is a
  misuse.

### 8.2 Model reserve

No reserve can be quantified at this stage, since the distance to the
true value is unknown. Pending the dual bound, a conservative treatment
would be to report the lower bound as is and to treat the absence of an
upper bound as an explicit gap rather than as a tolerance.

### 8.3 Monitoring

- Standard error as a proportion of the price, at each valuation. A rise
  signals either an under-sized path count or a contract whose payoff
  dispersion exceeds what was tested.
- Shape of the exercise boundary. A non-monotone or erratic boundary
  indicates an unreliable continuation estimate, and it is visible when
  prices are not.
- Share of paths exercising at each date. An all-zero profile at the
  early dates means the contract has degenerated and the valuation is
  not exercising the regression.

### 8.4 Next steps, in order of priority

1. **Trinomial tree** (milestone 2). The only way to obtain a reference
   in the operating regime, by a method independent of Monte Carlo.
   Restricted to one-factor models by construction.
2. **Dual upper bound** (milestone 3). Closes the bracket and makes the
   primal-dual gap available as an overall quality measure, replacing
   the one-sided criterion used in section 5.3.
3. **Mean-reverting models** (milestone 4). One-factor Ornstein-Uhlenbeck
   with seasonality, then Schwartz-Smith. A nested test is available at
   no cost: setting the short-term volatility to zero in Schwartz-Smith
   must reproduce the GBM exactly, and setting the long-term volatility
   to zero must reproduce the one-factor OU.
4. **Greeks and hedging** (milestone 5). Delta by finite differences with
   common random numbers, and the residual profit and loss of a delta
   hedge on forwards. This is what would turn the valuation into a
   statement about realisable value rather than expected value.

### 8.5 Revalidation triggers

Beyond the scheduled review, revalidation is required on any of the
following: a change of price model, a move to electricity as underlying,
a contract with a non-zero penalty or a binding q_min, a contract with
more than a few hundred exercise dates, or any change to the volume
interpolation scheme.
