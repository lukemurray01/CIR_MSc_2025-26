# Weak-error benchmark with analytical control variates

Methodology note and audit brief for `experiments/run_weak_error.py`,
`src/metrics/control_variate.py`, and the Kaggle runner
`notebooks/kaggle/kaggle_weak_error_control_variate.ipynb`.

Status: validated at small scale, **not yet run at production scale**.

Revision 2 (post-audit). An external audit of revision 1 found one blocking
defect and several real errors; §12 records what changed. The headline
corrections: the rate fit has been replaced (the previous one was not what
this document claimed it was), the adaptive `g2` lookup was quadratic in
storage and would not have finished, and the `g4` and `g2` open questions
from §10 are now closed by validation tests rather than left open.

---

## 1. What the experiment measures

The CIR process

    dX_t = kappa (theta - X_t) dt + sigma sqrt(X_t) dW_t,   X_0 = x_0,

with `kappa = 2`, `theta = 0.02`, `x_0 = 0.02`, `T = 1`, and `sigma` setting
the regime through the dimensionless `delta = 4 kappa theta / sigma^2`:

| Regime | sigma    | delta | Position                    |
|--------|----------|-------|-----------------------------|
| A      | 0.10     | 16    | Feller well satisfied       |
| B      | 0.20     | 4     | Feller satisfied            |
| C      | 0.282843 | 2     | Feller boundary (critical)  |
| D      | 0.50     | 0.64  | Boundary-accessible         |
| E      | 0.80     | 0.25  | Strong Feller violation     |

For each scheme, each functional `g_j`, and each step size `h`, the quantity
measured is the **signed weak bias**

    b_j(h) = E[ g_j(X^h_T) ] - E[ g_j(X_T) ],

where the second term is the *exact* value under the CIR law, not a
fine-grid reference. The experiment therefore carries **no
reference-solution error** — the dominant caveat on the strong-error results
elsewhere in this project. Its only error terms are Monte Carlo noise,
floating point, and (for `g2`) quadrature of the reference value.

### Functionals and their exact references

| | Functional | Exact reference |
|---|---|---|
| `g1` | `x` | CIR mean, `theta + (x_0 - theta) e^{-kappa T}` |
| `g2` | `(x - K)_+^2`, `K = 0.02` | quadrature against the noncentral chi-square density (`scipy.integrate.quad`, `epsabs=1e-12`, `epsrel=1e-10`) |
| `g3` | `exp(-x)` | CIR Laplace transform at `u = 1` |
| `g4` | `exp(-int_0^T X_s ds)` | affine zero-coupon bond price |

`g4` is path-dependent; the numerical path integral uses the **trapezoidal
rule on the scheme's own grid**, so the measured `g4` bias contains a
functional-discretisation term as well as the scheme's. See §9.

Transition law as implemented (`src/samplers/exact.py`): `X_{t+dt} | X_t = x`
is `Z / c` with `Z ~ ncx2(df, nc)`, `c = 4 kappa / (sigma^2 (1 - e^{-kappa dt}))`,
`df = 4 kappa theta / sigma^2`, `nc = c x e^{-kappa dt}`.

---

## 2. The problem this change solves

The direct estimator's standard error is `sd(g_j(X^h_T)) / sqrt(M)`, which is
**essentially independent of `h`** — refining the mesh shrinks the bias but
not the noise. Concretely, regime E, `g1`:

- payoff standard deviation ~ `0.056` against a mean of `0.020`;
- at `M = 2x10^5`, s.e. ~ `1.25x10^-4`;
- FTE bias ~ `2.7x10^-3` at `h = 2^-3`, falling to ~ `2.3x10^-5` at `2^-9`.

So from roughly `h = 2^-6` down, the direct estimator measures noise. In a
standalone pilot at `M = 10^5` the raw regime-E FTE bias sign-flips across
consecutive levels (`-2.8e-4, +3.0e-4, -2.9e-4, -6.3e-5, +7.8e-6, ...`), all
within one s.e. of zero.

This is visible in the currently published results: in
`results/weak_error_orders.csv` at `main`, the regime-E FTE orders for
`g1`–`g3` (0.62, 0.62, 0.62) are each fitted to **exactly 3 levels**, the fit
policy's stated minimum, and the regime-D FTE entries are refused at 2 and 1
usable levels. Reaching order-one behaviour at `2^-12` with the direct
estimator would need `~10^9` paths to detect and `~10^10` for a rate.

---

## 3. The control variate

For the value function `u_j(t, x) = E[ g_j(X_T) | X_t = x ]`, martingale
representation for the *exact* CIR process gives

    g_j(X_T) = E[g_j(X_T)] + int_0^T sigma sqrt(X_s) d_x u_j(s, X_s) dW_s.

The discrete analogue evaluated along the *scheme's own* trajectory is

    C_h = sum_n sigma sqrt(X_n^+) d_x u_j(t_n, X_n^+) dW_n,          (*)

and the production estimator is

    b_hat_j(h) = (1/M) sum_m [ P_h,j^(m) - C_h,j^(m) ] - E[g_j(X_T)].

Two properties carry the whole change.

**(i) `E[C_h] = 0` exactly.** Each summand is `F_{t_n}`-measurable times an
increment that is conditionally centred given `F_{t_n}`. This holds for any
adapted scheme whose step size is fixed *before* the increment is drawn.
Consequently the control **cannot bias the estimate**, whatever `d_x u_j` is.
Errors in `d_x u_j` cost variance-reduction efficiency only. This is what
makes the tabulated `g2` derivative (§5) safe, and it is the single most
important claim for an auditor to check.

**(ii) The residual is small.** Since (*) would be an exact hedge for the
true diffusion, `P_h - C_h` retains only the discrepancy between the scheme
and the exact dynamics. The residual standard deviation is *observed* to scale like `O(h)` for the
order-one schemes (this is measured, not established in general), so
`|b| / s.e.` is empirically **level-independent** and the deep ladder becomes reachable at the existing
path count.

**Choice of `beta`.** A coefficient `beta` multiplying `C_h` leaves the
estimator unbiased for *any* fixed value, so `beta = 1` is used — the
martingale-representation value, and the `h -> 0` limit of the variance-
optimal coefficient. Measured optimal `beta_hat` for FTE regime E: `1.131,
1.068, 1.036, 1.019, 1.010, 1.005, 1.003, 1.001` for `k = 3..10`. Using a
fixed `beta` avoids the same-sample coefficient-estimation problem entirely
and keeps the confidence intervals exact.

---

## 4. Where the control is computed

The five path-based schemes (FTE, HH, ProjEuler, uniform KL, IF, BLT) return
full path arrays, and the samplers return `max(X, 0)` — precisely the state
whose square root each scheme used in its own diffusion coefficient. The
control is therefore computed **post hoc** from the returned paths and the
increment array, with **no change to any sampler**.

The two free-running adaptive schemes (KLM backstop, adaptive soft-zero KL)
return terminal values only, so the control is accumulated inside their step
loops. Both choose the step from the state before drawing the increment:

- KLM: `h = h_max * min(1, |Y_n|)`, floored at `h_min = h_max / rho`
  (`rho = 64`), in the Lamperti coordinate `Y = sqrt(X)` so `sqrt(X) = |Y|`.
- adaptive KL: soft-zero region takes a deterministic ODE step and
  contributes nothing to the control; splitting steps fix `dt` from the state
  before drawing `dW`.

**Backstop retakes.** All three backstop maps (`implicit`, `projected`,
`blt`) reuse the *same* Brownian increment that made the explicit step fail.
The `blt` map additionally draws a bridge running infimum, but conditionally
on that increment. `E[C_h] = 0` is therefore preserved in every case. This
was checked in source rather than assumed.

---

## 5. The derivatives

With `tau = T - t` the remaining time:

- **`g1`**: `d_x u_1 = e^{-kappa tau}`.
- **`g3`**: `u_3 = D^{-delta/2} exp(-x e^{-kappa tau} / D)` with
  `D = 1 + sigma^2 (1 - e^{-kappa tau}) / (2 kappa)`, so
  `d_x u_3 = -(e^{-kappa tau} / D) u_3`.
- **`g4`**: `u_4(t, x, A_t) = e^{-A_t} P(tau, x)` where `A_t` is the running
  trapezoidal integral and `P = A(tau) e^{-B(tau) x}` is the affine bond
  price; `d_x u_4 = -B(tau) P(tau, x) e^{-A_t}`.
- **`g2`**: no closed form. Derived via the noncentral chi-square
  Poisson-mixture identity

      dG/dlambda = (1/2) [ G(delta + 2, lambda) - G(delta, lambda) ],

  where `G(delta, lambda; z) = E[(Z - z)_+^2]` for `Z ~ ncx2(delta, lambda)`,
  combined with `dlambda/dx = c e^{-kappa tau}`, giving

      d_x u_2 = (e^{-kappa tau} / (2c)) [ G(delta+2, lambda) - G(delta, lambda) ].

  `G` itself uses truncated-moment identities obtained by expanding the
  Poisson mixture over central chi-squares and re-indexing:

      P(Z > z)       = sf(z; d, l)
      E[Z 1_{Z>z}]   = d sf(z; d+2, l) + l sf(z; d+4, l)
      E[Z^2 1_{Z>z}] = d(d+2) sf(z; d+4, l) + 2 l (d+2) sf(z; d+6, l)
                       + l^2 sf(z; d+8, l)

  Note the middle coefficient is `2 l (d+2)`, not `2 l (d+4)`; the latter is
  an easy slip and was caught by the quadrature test.

**Tabulation.** Evaluating `ncx2.sf` per path per step is far too slow, so
`d_x u_2` is tabulated once per regime on a `96 x 384` grid, log-spaced in
`tau` and graded in `x` (three quarters of the nodes in `[0, 4K]`, where the
`tau -> 0` limit `2(x-K)_+` has its kink), then bilinearly interpolated.
Measured interpolation error is `< 5x10^-3` relative. **This costs efficiency
only, never bias**, by property (i).

---

## 6. Validation performed

34 new tests in `tests/test_control_variate.py`, all passing; full suite 228
passing.

- `G` from the moment identities vs direct quadrature — regimes A/C/E,
  `tau` in {1, 0.25, 0.01}, `x` in {0.005, 0.02, 0.05}, rel 1e-7.
- `d_x u_2` vs central finite difference of a quadrature-computed `u_2`,
  same grid, rel 2e-4.
- `d_x u_1`, `d_x u_3`, `d_x u_4` vs finite differences of the shipped closed
  forms, rel 1e-5/1e-6.
- `cir_bond_AB` reproduces the shipped `affine_cir_bond_price`, rel 1e-12.
- Running trapezoid endpoint matches the shipped `trapezoidal_integral`.
- `E[C_h] = 0` within 4 s.e. by simulation, all four functionals.
- **Direct vs control-variate agreement**: across 24 smoke rows, every
  difference between the two estimators sits within `1.2` s.e. of the direct
  estimator — the empirical form of property (i).
- **Resume fidelity**: an interrupted-then-resumed run is bit-identical to an
  uninterrupted one on every statistical column.

---

## 7. Statistical methodology

### What changed and why

The previous policy fitted `log|b|` against `log h` over only those levels
satisfying `|b| >= 2 s.e.`, refusing the fit below three such levels.
Selecting points by significance and then fitting **selects on the very noise
it is trying to exclude**, biasing the fitted slope; and fitting absolute
values discards sign information that distinguishes a real bias from a noise
excursion.

The production fit is **signed generalised least squares**: minimise

    sum_k ( b_k - C h_k^alpha )^2 / s_k^2

directly on the signed biases, with `C` profiled out analytically at each
`alpha` (the model is linear in `C`), leaving a one-dimensional search. No
logarithm, no absolute value, no data-dependent deletion of levels. The
interval on `alpha` is the 95% profile region
`{alpha : RSS(alpha) <= RSS_min + 3.841}`.

An intermediate version fitted `log|b|` after choosing a dominant sign from
the data and discarding minority-sign levels, weighting by `(|b|/s.e.)^2`.
That was still a data-dependent selection, and near the noise floor `log|b|`
is badly non-normal while the weights correlate with the response they
weight — biasing the slope in precisely the regime the control variate
exists to reach. A signed fit needs none of that machinery: a level whose
bias is consistent with zero simply carries little weight and pulls the
curve toward zero, which is the correct inference.

The old policy is retained as `fitted_weak_order_legacy` so the rerun stays
directly comparable with the numbers currently in the results chapter.

### Reporting gate

An order is **refused** when fewer than three levels carry `|b| >= 2 s.e.`
This gates *reporting*, not point selection — the fit still uses every level,
weighted. A refusal keeps its original meaning (the bias sits below the noise
floor, which is itself a finding) without reintroducing selection bias. This
was added after the runner reported "order 1.071 +/- 1.429" for adaptive KL
in regime E on `g3` with **0 of 5** levels resolved.

### Drift diagnostic

`order_drifting` is set when the signed GLS fits over the **coarse half** and
the **fine half** of the ladder have disjoint 95% profile intervals. Because
levels are independently seeded, fits on disjoint level sets are independent
and their intervals may be compared directly.

An earlier version regressed consecutive local slopes against level index.
That is invalid: adjacent local slopes share an observation and are
correlated, so the trend's standard error was understated, and omitting
unresolved pairs silently compressed the spacing. Local slopes are still
reported (with `noise` and `sign-change` markers) as a descriptive aid, but
no longer carry the diagnostic.

`fitted_order_fine_window` gives the same fit over a contiguous block at the
fine end, declared by position rather than chosen after seeing the fit.

### Plotting

Levels whose interval crosses zero are drawn as a **95% upper bound** with an
open marker, not joined into the curve — a point estimate there is a draw
from the noise, and connecting it manufactures apparent oscillation that is
not measured convergence behaviour.

---

## 8. Experimental design for the production run

| Item | Setting |
|---|---|
| Paths | 200,000 (from `configs/experiments.yaml`) |
| Fixed-step ladder | `h = 2^-3 .. 2^-12` (10 levels) |
| Adaptive ladder | `h_max = 2^-3 .. 2^-10` |
| Regimes | A–E |
| Schemes | FTE, HH, ProjEuler, KL, IF, KLM, BLT |
| Precision | float64 throughout |
| Seeding | per level, `master_seed + 7919 * level` |
| Estimator | exact-reference, `beta = 1` martingale control variate |

**Levels are independently seeded**, hence independent across `k`. There is
deliberately **no shared fine Brownian hierarchy across levels**. Rationale:
the adaptive schemes' coupling quantises a proposed step to `m = max(floor(h
/ h_ref), 1)` fine steps. Since `h <= h_max`, a shared grid at `2^-12` forces
`m = 1` at every step for `k = 12`, and with typical `|Y| ~ sqrt(0.02) ~
0.14` the degeneracy begins around `k = 9` — silently collapsing the adaptive
scheme to a uniform mesh and misattributing the result. Preserving adaptivity
would require a shared grid near `2^-17`, at ~32x the finest-level cost. The
adaptive schemes therefore run free, as in the published version.

**Adaptive cost.** KLM's accepted steps per path in regime E, measured:
3,679 at `h_max = 2^-7` rising to 31,592 at `2^-10` against a nominal 1,024 —
a ~31x adaptivity tax, with the backstop firing on 58.6% to 64.7% of steps.
`steps_per_path` and `backstop_fraction` are now recorded per row. This is
why the adaptive ladder stops at `2^-10`.

**Memory.** Batch size is scaled as `min(requested, 25e6 / n_steps)` to bound
the path array; peak observed ~1.6 GB.

**Runtime.** ~3.1 h measured for the fixed-step portion plus ~1 h adaptive on
a desktop CPU; expect 5–8 h on Kaggle against a 12 h cap. Checkpointed per
(regime, level) with an optional wall-clock budget, resumable.

---

## 9. Preliminary results

**Variance reduction** (FTE, `M = 10^5`, ratio of variances):

| Regime | `g1` k=6 | `g1` k=9 | `g2` k=9 | `g3` k=9 | `g4` k=9 |
|---|---|---|---|---|---|
| A | 1,641 | 103,421 | 472 | 101,931 | 867,303 |
| C | 1,612 | 103,152 | 560 | 87,561 | 692,893 |
| E | 410 | 25,070 | 651 | 10,627 | 97,586 |

In regime E this moves `|b| / s.e.` for `g1` from `0.2` (unresolvable) to
`20`, and holds it near `20` at every level from `k = 3` to `k = 13` —
the level-independence predicted in §3(ii).

**Two findings that would change the results chapter.** Both are from a
*standalone* pilot implementation (not the repo samplers) at `M = 10^5`,
`k = 3..13`, and need confirmation by the production run:

1. **FTE, regime E, `g1`**: local order settles at `1.07–1.11` for
   `k >= 9`, against the published `0.62`. Repo smoke runs at `k = 3..7` give
   `1.23` (20k paths) and `1.26` (5k paths) where the legacy policy refuses
   the fit entirely. If this holds, the "strong and weak rankings invert
   below the boundary" claim gets *stronger*, but the sentences framing
   sub-noise results as findings become obsolete.

2. **ProjEuler, regime E, `g1`**: local slope drifts monotonically downward —
   `0.40, 0.36, 0.32, 0.29, 0.27, 0.25, 0.23, 0.22, 0.20, 0.19` for
   consecutive pairs through `k = 13` — with no sign of settling. If that
   holds, the published `0.32` is a property of the measurement window, not
   of the scheme, and running deeper returns a *smaller* number rather than
   a more accurate one.

   An earlier draft connected this to the `delta/2 = 0.125` Hefter–Jentzen
   ceiling. **That connection is withdrawn**: the HJ lower bound constrains
   *strong* (pathwise `L^p`) approximation on uniform meshes and does not by
   itself bound the weak convergence of `E[g_1]`. The observation is
   empirically consistent with a boundary-driven fractional bias; it does
   not have a theoretical target attached.

---

## 10. Points to audit hardest

Ordered by how much a defect would cost.

1. **`E[C_h] = 0`.** The whole design rests on it. Adaptivity chooses the
   step from the state before the increment is drawn; backstops reuse the
   same increment. Is there any path through any scheme where the increment
   influences the step size or is redrawn?

2. **RESOLVED — `g4`'s trapezoidal term is `O(h^2)`.** Evaluated
   deterministically by backward affine recursion on an exact CIR skeleton
   (no Monte Carlo): local order `2.000` in every regime, and in regime E
   the term falls from `1.93x10^-6` at `h = 2^-3` to `7.28x10^-12` at
   `2^-12` — orders of magnitude below any bias the experiment reports. The
   near-order-one `g4` results are therefore not a quadrature artefact.
   Locked in by `test_g4_trapezoid_error_is_second_order`.

3. **RESOLVED — `g2` reference is now closed-form.** `E[g2]` uses the
   truncated-moment expression `c^-2 G(delta, lambda; cK)` rather than
   semi-infinite quadrature; the two agree to `<1e-15` in every regime and
   the quadrature is retained as a test oracle. The tolerance question is
   moot.

4. **Pre-asymptotic curvature vs true fractional order.** FTE's local slopes
   are non-monotone at coarse `h` (`1.81, 1.29, 0.85, 0.91, 0.92, 1.11, ...`).
   Is a single fitted order meaningful over `k = 3..12`, or should the
   headline number come from a fine-end window only? The stability columns
   (`order_stability_spread`, leave-one-end-out) are meant to expose this.

5. **Window choice.** The fine window is the finest 5 levels and the drift
   test splits the ladder in half — both declared by position, but still
   arbitrary. Does the conclusion survive other splits?

6. **`beta = 1` for schemes far from exact dynamics.** ProjEuler gets only
   3–17x variance reduction at coarse `h`, presumably because its floor makes
   the exact-CIR hedge a poor fit. It does not need the reduction (its bias
   is large), but is `beta = 1` leaving anything on the table for BLT or KLM?

7. **`g2` control effectiveness.** 60–650x versus `10^4`–`10^5` for the
   others. Is the limit the tabulation resolution or the tail nature of the
   functional? If the former, a finer table is cheap.

8. **Configuration discrepancy, unresolved.** `configs/experiments.yaml` sets
   `weak_error.n_paths: 200000` and an `n_steps` list that **is never read**
   (the runner takes its ladder from the CLI default). The results chapter
   says `10^5` paths and six levels. The production run uses 200,000. Which
   is correct for the published numbers needs settling for the
   reproducibility ledger.

---

## 11. Files changed

| File | Change |
|---|---|
| `src/metrics/control_variate.py` | new — derivatives, `g2` table, control assembly |
| `tests/test_control_variate.py` | new — 34 validation tests |
| `experiments/run_weak_error.py` | control variate, signed weighted fit, drift diagnostic, upper-bound plotting, checkpoint/resume, `--out-dir`, adaptive step-count recording |
| `src/samplers/klm_backstop.py` | optional in-place control accumulation |
| `src/samplers/kelly_lord_adaptive.py` | optional in-place control accumulation |
| `notebooks/kaggle/kaggle_weak_error_control_variate.ipynb` | new — clone-and-run Kaggle driver |
| `notebooks/kaggle/README.md` | documents the deviation from the standalone convention |

Output columns added to `results/weak_error.csv`: `estimator`,
`signed_error_cv`, `mc_standard_error_cv`, `error_to_se_cv`,
`variance_reduction`, `batch_standard_error_cv`, `steps_per_path`,
`backstop_fraction`. The direct-estimator columns are unchanged and still
written, so the rerun remains comparable with the published results.


---

## 12. Revision 2: what the audit changed

An external audit of revision 1 returned the following. Items marked
**fixed** are in the code; items marked **noted** are recorded judgements.

**Blocking defect (fixed).** The adaptive `g2` derivative lookup built an
`(n_active, 384)` array and Python-looped `np.interp` over active paths on
every adaptive round — measured at 3.0 s and 613 MB per round at 200,000
paths. KLM in regime E at `k = 10` runs tens of thousands of rounds, so the
production run would not have finished. Replaced with vectorised bilinear
interpolation from four indexed entries per path: 50 ms per round, 60x
faster, `O(n_active)` storage.

**The rate fit was not what revision 1 described (fixed).** See §7.

**Correlated drift statistic (fixed).** See §7.

**Plot joined across unresolved gaps (fixed).** All resolved levels were
drawn as a single line, so a segment was drawn straight across any
unresolved gap — reintroducing exactly the spurious convergence the
upper-bound policy exists to prevent. Now each contiguous run of resolved
levels is its own line, with 95% error bars and per-scheme noise floors
(which differ by orders of magnitude between schemes once the control
variate is applied, so a single pooled floor was misleading). Order-1 and
order-1/2 guides are both drawn.

**Resume could merge incompatible partials (fixed).** A configuration hash
over paths, ladder, schemes, control setting, seed and `T` is written
alongside the partial CSV and checked before any resume; a mismatch aborts
with a diff rather than silently combining runs.

**Seed stream depended on scheme selection (fixed).** The free-running
adaptive schemes drew from one shared per-level generator in sequence, so
adaptive KL's stream depended on whether KLM had run first — meaning
`--schemes KL` and `--schemes KLM KL` gave different KL numbers from the
same seed. Streams are now keyed by `(regime, scheme, level)`.

**Runtime estimate was wrong (corrected).** Revision 1 said 5–8 h. That
probe was run with `--max-adaptive-steps 0`, which disabled the adaptive
schemes entirely, and the adaptive cost was then extrapolated from a
control-free timing. Measured properly: ~3 h fixed-step plus ~4 h adaptive,
so ~7 h on a desktop CPU and plausibly beyond Kaggle's 12 h cap on slower
hardware. The run should be split by regime across two sessions.

**`delta/2` ceiling claim withdrawn (corrected).** See §9.

**Overstated "exact" intervals (corrected).** See §3.

**Noted — level coupling.** The audit suggests coupling the fixed-step
levels via nested Brownian increments while leaving the adaptive schemes
free, observing correctly that the adaptive argument does not force
independence on the fixed-step schemes. Not adopted, for two reasons.
First, independent levels are what make the checkpoint/resume bit-identical
to an uninterrupted run, which is what allows the run to be split across
Kaggle sessions at all. Second, they make the coarse-half/fine-half drift
test statistically clean, since disjoint level sets are then independent.
With the control variate delivering `|b| / s.e. ~ 20` at every level, the
marginal value of coupling for slope precision is small. This is a genuine
trade rather than an oversight, and it is reversible.

**Noted — figure layout.** The audit reports that the code produces one
four-panel figure per regime "not the requested four figures organised by
`g1..g4`". No such layout was ever specified; the existing thesis figure
(`figures/results/weak_error_regime_E.pdf`) is one 2x2 panel per regime and
that convention is retained.

**Noted — the P100.** The audit lists "P100 will accelerate this
implementation" as a claim to be rejected. It is not a claim made here: the
notebook sets the accelerator to None and states that no GPU is needed. The
question of whether a JAX port would be worth it is open, but it is not
required for this run.
