# Thesis Alignment Changes

Status date: 1 July 2026

This checklist records the changes needed before the repository can be treated
as fully aligned with the rebased thesis scope:

- CIR benchmark as the mathematical anchor.
- GPU/JAX implementation as the reproducible computational contribution.
- CEV as a cautious controlled extension, not a completed general theory.

## Required Before Thesis-Ready

1. Settle the strong-error reference policy. [Done]
   The main CIR benchmark now uses HH as the fine-grid reference throughout.
   Thesis text, README, figure captions, and CSV columns should describe this
   same convention.

2. Stop treating HH as both reference and ordinary benchmark scheme. [Done]
   HH is `reference_only` in config and is removed from the ranked strong-error
   benchmark curves.

3. Canonicalise method names. [Done]
   `ProjEuler` is the public identifier across config, scripts, styles, README,
   CSV outputs, and tests.

4. Make the method registry honest. [Done]
   Every method listed in config should import, have a plotting style/label,
   be exercised by at least one test or script-level smoke check, and have its
   role documented as core, reference-only, terminal-only, or exploratory.

5. Update shared plotting styles. [Done]
   Add `KLM` to `METHOD_COLOURS`, `METHOD_LINESTYLES`, and `METHOD_LABELS`, and
   fix `DEFAULT_LINEWDITH` to `DEFAULT_LINEWIDTH`.

6. Add a reproducibility manifest. [Done]
   Record the exact commands, seeds, parameter regimes, output paths, and status
   for each figure/table intended for the thesis.

7. Decide the generated-output policy. [Done]
   Either ignore all generated outputs consistently or intentionally track a
   complete artifact bundle. Avoid tracking only `config.json` and PDFs while
   hiding the corresponding CSVs through global `*.csv` ignore rules.

8. Validate the main CIR benchmark command. [Smoke validated]
   A small HH-reference smoke run has passed. Before final thesis figures, run
   the full thesis-selected strong-error command, record the test command, and
   keep the generated figure/CSV paths in the reproducibility manifest.

9. Align GPU/JAX claims with evidence. [Done]
   Keep CPU/JAX parity as the minimum claim. Only describe GPU speedup or
   scaling as a result if timings are measured on the target environment and
   recorded with hardware/software details.

10. Keep CEV scoped as an extension. [Done]
    Preserve the beta `1/2` CIR consistency test and one beta `>1/2`
    experiment, but label the CEV results as exploratory unless a separate CEV
    backstop convergence proof is added.

11. Separate proved, cited, and observed claims.
    The repository outputs should support thesis wording that distinguishes
    KLM proof assumptions, cited convergence results, and numerical behaviour
    observed in the local experiments.

12. Sync the thesis branch to the public/mainline repository.
    The thesis-aligned state should not live only on a local feature branch.
    Push or merge the reviewed branch once the alignment checklist is complete.

## Fixed After Review (5 July 2026)

13. Rescale the projected Euler floor. [Done]
    The ProjEuler default floor was `y_floor = dt` in Lamperti coordinates.
    For alpha < 0 (regimes D/E) the drift kick at that floor is O(1) against
    O(sqrt(dt)) noise, so escaping the floor is a Gaussian tail event
    N > 2|alpha|/(sigma*sqrt(dt)): paths absorb at X = dt^2 as dt -> 0
    (96.6% of paths ended at the floor in regime E at dt = 2^-9) and the
    strong error grows instead of converging, which produced the negative
    fitted orders in `fig_order_vs_delta_summary.csv`.  The default is now
    `y_floor = 0.5*sigma*sqrt(dt)` (`default_y_floor` in
    `src/samplers/projected_euler.py`), which is the Hefter--Herzwurm
    truncation level `sigma^2*dt/4` in X-coordinates and restores monotone
    order ~1/2 convergence in D and E.  Applied consistently to:
    - `src/samplers/projected_euler.py` (all path/terminal variants),
    - `src/samplers/cev.py` (`sigma*sqrt(dt)`, twice the CIR floor, which
      preserves the exact beta = 1/2 reduction),
    - the standalone Kaggle notebooks (`kaggle_cir_benchmark_suite.ipynb`,
      `kaggle_cir_benchmark_suite_JAX.ipynb`,
      `kaggle_reference_sensitivity.ipynb`,
      `kaggle_four_scheme_regime_rates_JAX.ipynb`).
    The paper-reproduction floor `dt**0.25` in
    `experiments/kl_adaptive_splitting_paper*.py` is intentionally unchanged
    (it reproduces the published choice and is not subject to the trap since
    dt^(1/4) decays slower than sqrt(dt)).
    `src/metrics/strong_error.py` now also uses the HH reference instead of
    the drift-implicit scheme, which is undefined (negative discriminant,
    silent NaN) for alpha < 0.
    Regenerated: `results/strong_error_regime_{A..E}.csv`, the strong-error
    figures, `results/fig_order_vs_delta_summary.csv`, and
    `results/distributional_diagnostics.csv`.  Any Kaggle-generated outputs
    produced before this date used the old floor and must be re-run before
    appearing in the thesis.

## Fixed After Review (6 July 2026)

14. Apply the 48-hour review plan. [Done]
    - Regression tests pin the floor fix: `tests/test_projected_euler_floor.py`
      (`test_projected_euler_floor_trap_regression` reproduces the old
      dt-scaled-floor trap in regime E and asserts the default
      `0.5*sigma*sqrt(dt)` floor restores monotone convergence;
      `test_default_y_floor_scales_like_sqrt_dt` pins the scaling and its
      equality with the HH truncation level `sigma^2*dt/4`).
    - `klm_backstop_terminal_from_fine_dW` now emits a `RuntimeWarning` when
      `dt_fine > h_max/rho`, i.e. when the backstop floor is not representable
      on the fine reference grid (guarded by
      `test_klm_coupled_warns_when_fine_grid_coarser_than_h_min`).  The guard
      immediately exposed an inconsistent grid in
      `test_klm_coupled_error_decreases_with_h_max`, fixed by using rho=32
      there.  Note: the reference-sensitivity gate legitimately triggers this
      warning for its 4096/16384 reference rows at the finest coarse levels —
      that distortion is part of what the gate measures.
    - `experiments/fig_order_summary.py` now reports full-fit AND tail-fit
      orders (L1 and L2) plus the reference-sensitivity min/max per
      regime/scheme, draws the sensitivity span as bars on the figure, and
      carries a footnote that uniform-mesh points above the delta/2 band are
      reference-limited coupled diagnostics.  CSV columns renamed:
      `order_l1`/`order_l2` -> `order_l{1,2}_full`/`order_l{1,2}_tail` +
      `sens_l2_min`/`sens_l2_max`.
    - `experiments/run_strong_error.py` marks ProjEuler/KLM legend entries
      with a dagger in alpha < 0 regimes and adds the not-reference-converged
      footnote to those figures.
    - README, `docs/reproducibility_manifest.md`, and the
      `configs/experiments.yaml` comment now all state the production
      reference `reference_n_steps = 32768` and the D/E quoting rule
      (sensitivity range attached, coupled diagnostics not true rates), with
      the Hefter--Herzwurm min(1,delta)/2 and Hefter--Jentzen delta/2 anchors.
    - Reference-sensitivity gate re-run against the current code state to
      archive the evidence backing every quoted slope
      (`outputs/reference_sensitivity/strong_reference_sensitivity*.csv`).

## Added After Supervisor Meeting (6 July 2026)

15. BLT splitting scheme, BLT backstop, and CPU/GPU timing harness. [Done]
    - `src/samplers/blt_splitting.py`: exact BESQ(1) + linear-ODE Lie-Trotter
      composition in the paper's normal form (a = delta, b = kappa), with the
      modified positive-part variant for a < 1.  Noise machinery in
      `src/utils/rng.py` (joint (dW, infimum) sampling, Asmussen--Glynn--
      Pitman bridge minimum) and `src/utils/brownian.py` (exact coarse-grid
      aggregation: sum increments, min-compose infima).
    - `src/samplers/klm_backstop.py`: third backstop kind "blt" — retakes
      the failed explicit step with the SAME increment plus one auxiliary
      exponential; output floored at gamma*sqrt(h) (needed only for a < 1).
    - `klm_jax/fixed_step.py`: JAX kernels (FTE, ProjEuler, KL, BLT) +
      device-side joint noise; parity tests in
      `tests/test_jax_fixed_step_parity.py` (note: FTE parity in regime E is
      asserted at 1e-4 because sqrt at the truncation kink amplifies 1-ulp
      XLA reassociation differences).
    - Experiments: `run_blt_strong_error.py` (dual BLT-self/HH reference),
      `run_klm_backstop_comparison.py` (implicit/projected/BLT vs exact law),
      `run_cpu_gpu_timing.py` (NumPy vs JAX, backend recorded per row).
    - Registry: BLT added to configs methods.core, style maps, and the
      registry tests.  Suite: 191 tests passing, warning-clean.
    - First results: BLT fitted L1 orders 1.01/1.01/0.97 in A/B/C (theorems:
      1, 1, 1-eps) — implementation validated against the paper.  Modified
      BLT in D/E: terminal zero atom 7-23% (D) and 21-42% (E) of paths,
      terminal-mean bias up to +52% in E — same taxonomy class as FTE.
      Backstop comparison: BLT backstop admissible and competitive with
      implicit in C; in E it does NOT remove the projected backstop's bias
      (+32% vs +29% at h=1/256), locating the D/E bias in the adaptive-rule
      + positivity-floor framework rather than in the backstop map.

## Added After Supervisor Meeting (6 July 2026, part 2)

16. Weak-error benchmark g1-g4 and a KL-adaptive stall fix. [Done]
    - `experiments/run_weak_error.py`: weak errors against exact functionals
      for g1 (mean), g2 (squared call, K=0.02), g3 (Laplace) and the
      path-dependent bond discount g4 (trapezoidal integral vs the affine
      bond formula), matching the thesis background-chapter definitions.
      Schemes follow the thesis method registry: FTE, HH, ProjEuler, KL
      (uniform A-C / adaptive soft-zero D-E), IF (alpha > 0 only), KLM, BLT.
      HH and IF are ranked as ordinary schemes here because weak error has
      no reference-path conflict.  Per the methodology chapter, every point
      records the MC standard error; order fits use only points with
      |error| >= 2 s.e. and are refused otherwise; consecutive local slopes
      are reported alongside global fits; noise-dominated points are drawn
      hollow in the figures.  Outputs: `results/weak_error.csv`,
      `results/weak_error_orders.csv`, `figures/weak_error_regime_{R}.pdf`.
    - Latent bug fixed in `src/samplers/kelly_lord_adaptive.py`
      (`kl_adaptive_terminal`, free-running): the exact soft-zero exit flow
      could land a few ulp below X_zero, re-entering the region with a
      vanishing step and stalling until max_rounds (exposed by regime E at
      dt_max = 1/8).  Completed exits are now snapped onto the boundary;
      regression test
      `test_kl_adaptive_terminal_does_not_stall_at_soft_zero_boundary`.
      The Brownian-coupled variant always advances one fine step and was
      never affected.
    - NOTE for the thesis text: chapter5.tex still documents the ProjEuler
      floor as Y_floor = h (pre-fix state); the code default has been
      0.5*sigma*sqrt(h) since item 13.  Reconcile the prose (keep the old
      floor as the documented failure and present the rescaled floor as the
      remedy, or update the paragraph).

## Useful But Secondary

1. Add a quick benchmark smoke command.
   Provide a low-cost command that regenerates representative CIR outputs with
   small `n_paths` for supervisor or examiner reproducibility checks.

2. Add script-level tests for figure entry points.
   Keep full experiments out of the unit suite, but test that main scripts load
   config, validate method names, and can run with tiny path counts.

3. Add a short limitations note.
   Summarise KLM behaviour outside the design regime, the role of projected
   fallback in alpha-negative regimes, and why exact transition sampling is
   used for terminal-law checks rather than Brownian-coupled strong error.

4. Archive stale outputs separately.
   Move notebook-era or exploratory artifacts out of the current thesis output
   path unless they are regenerated by current scripts.

## Definition Of Done

The repository is thesis-aligned when:

- the main CIR benchmark can be regenerated from a clean command;
- the strong-error reference convention is consistent across code and prose;
- method registries, styles, tests, and configs agree;
- CPU/JAX parity supports the implementation claim being made;
- CEV is clearly presented as a controlled extension;
- every thesis figure/table has a current source command and output path;
- the aligned branch is pushed or merged so the public repo matches the thesis
  evidence.
