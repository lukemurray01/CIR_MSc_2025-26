# CIR MSc Thesis Code (2025–26)

Numerical benchmark code for the MSc thesis *Backstopped adaptive schemes for
CIR and CEV processes: proof anatomy, GPU implementation, and numerical
benchmarks* (Luke Murray).

The repository implements the CIR benchmark schemes used in the thesis, an HH
fine-grid strong-error reference, the exact
noncentral chi-squared transition sampler, strong/weak/distributional error
metrics, a mean-reverting CEV extension, and a JAX implementation of the
Kelly–Lord–Maulana (KLM) backstopped adaptive scheme validated against the
NumPy reference.

## Layout

```
configs/          regimes.yaml (CIR parameter regimes A–E), experiments.yaml
src/samplers/     schemes: FTE, projected Euler, Kelly–Lord splitting
                  (uniform + adaptive/soft-zero), KLM backstopped adaptive,
                  HH Milstein reference, drift-implicit (IF), exact ncx2, CEV
src/metrics/      strong error, weak error (g1–g4), distributional (KS, W1),
                  boundary diagnostics
src/utils/        RNG, Brownian increments/aggregation/bridge, params, IO
klm_jax/          JAX (FP64) KLM kernel + Fig.-3 sweep
experiments/      figure- and table-producing scripts (see below)
tests/            pytest suite, including CPU/JAX parity tests
```

## Setup

With [uv](https://docs.astral.sh/uv/) (preferred; `uv.lock` is committed):

```
uv sync
uv run pytest -q
```

Or plain pip: `pip install numpy scipy matplotlib pyyaml pytest jax` then
`python -m pytest -q`. JAX-dependent tests are skipped automatically if JAX
is not installed.

## Full-scale KLM reproductions (Kaggle / GPU)

Two paper-scale JAX experiments are integrated verbatim from the July 2026
Kaggle notebooks (streaming Brownian chunks with bridge coupling, so the
h_ref = 2^-25 fine reference never has to be stored):

- `experiments/klm_fig2a_streaming.py` — KLM Fig. 2(a): RMSE vs h_mean for
  EA/EF_y/EF_x/IF/SIA/FT. Knobs: `FIG2A_M`, `FIG2A_REFERENCE_POWER`,
  `FIG2A_SEED`, `FIG2A_CHUNK_STEPS`.
- `experiments/klm_fig3_full_diagnostic.py` — KLM Fig. 3: fitted rate vs the
  Feller ratio a for kappa in {2, 0.2}. Knobs: `FIG3_FULL_DIAG_*`.

Defaults are the full paper-scale configuration (seed 2022); reduced-scale
runs of the integrated scripts are bit-identical to the original notebook
code (verified at integration; guarded by
`tests/test_klm_streaming_scripts.py`). Run them on Kaggle's free P100 via
the standalone single-cell notebooks in
[`notebooks/kaggle/`](notebooks/kaggle/README.md): `kaggle_klm_fig2a.ipynb`,
`kaggle_klm_fig2a_JAX.ipynb`, `kaggle_klm_fig3_full.ipynb`, and
`kaggle_klm_fig3_full_JAX.ipynb`. For KLM Fig. 2(a), the base notebook is a
NumPy/CPU bridge-streaming port and the `_JAX` notebook is the GPU paper-scale
route. The KLM Fig. 3 full notebooks remain JAX-native because the paper-scale
h_ref = 2^-25 diagnostic is maintained as a streaming GPU implementation.
Treat GPU speedup or scaling as unreported until timings are measured and
recorded with hardware/software details.

The full repo benchmark suite also has standalone Kaggle notebooks:
`kaggle_cir_benchmark_suite.ipynb` is a monolithic CPU/NumPy cell, while
`kaggle_cir_benchmark_suite_JAX.ipynb` is a monolithic GPU/JAX cell. Neither
notebook imports the local repository.

The Kelly-Lord adaptive-splitting paper figures are handled separately by
`experiments/kl_adaptive_splitting_paper.py`. That script reproduces the
paper's Fig. 2/3 setup with HH/Milstein reference `dt_ref = 1e-5`, `M = 1000`,
20 batches of 50, the paper `dtmax` grid, projected floor `N^(-1/4)`, and
Brownian-bridge coupling for adaptive splitting. The standalone single-cell
Kaggle notebook is
`notebooks/kaggle/kaggle_kl_adaptive_splitting_paper.ipynb`. A JAX/GPU port of
the same reproduction is available at
`experiments/kl_adaptive_splitting_paper_jax.py` and
`notebooks/kaggle/kaggle_kl_adaptive_splitting_paper_JAX.ipynb`; treat the
NumPy version as the paper-faithful oracle and the JAX version as the GPU
implementation layer.

## Reproducing the thesis figures

All experiments are seeded from `configs/experiments.yaml`
(`master_seed: 120339106`); regimes A–E are defined in `configs/regimes.yaml`
(kappa = 2, theta = 0.02, X0 = 0.02, sigma varies).

| Output | Command |
|---|---|
| Strong-error benchmark, 4 schemes vs HH reference × 5 regimes (`figures/strong_error_regime_{R}.pdf`, `results/strong_error_regime_{R}.csv`) | `uv run python experiments/run_strong_error.py` |
| Reference-sensitivity gate for production slopes (`outputs/reference_sensitivity/*.csv`) | `uv run python experiments/run_reference_sensitivity.py`; Kaggle route: `notebooks/kaggle/kaggle_reference_sensitivity.ipynb` |
| KLM backstop-usage diagnostic (`figures/klm_backstop_diagnostic.pdf`) | `uv run python experiments/run_klm_diagnostic.py` |
| KL adaptive-splitting paper Fig. 2/3 smoke (`figures/kl_adaptive_splitting_smoke_combined.pdf`) | `uv run python experiments/kl_adaptive_splitting_paper.py --mode smoke` |
| KL adaptive-splitting paper Fig. 2/3 JAX smoke (`figures/kl_adaptive_splitting_smoke_jax_combined.pdf`) | `uv run python experiments/kl_adaptive_splitting_paper_jax.py --mode smoke` |
| BLT splitting strong error, dual reference (`results/blt_strong_error.csv`, `figures/blt_strong_error.pdf`) | `uv run python experiments/run_blt_strong_error.py` |
| KLM backstop-candidate comparison, implicit/projected/BLT (`results/klm_backstop_comparison.csv`, `figures/klm_backstop_comparison.pdf`) | `uv run python experiments/run_klm_backstop_comparison.py` |
| CPU vs JAX scheme timing (`results/cpu_gpu_timing.csv`, `figures/cpu_gpu_timing.pdf`; run on Kaggle P100 for GPU numbers) | `uv run python experiments/run_cpu_gpu_timing.py` |
| Weak error g1-g4 vs exact functionals, 7 schemes x A-E (`results/weak_error.csv`, `results/weak_error_orders.csv`, `figures/weak_error_regime_{R}.pdf`) | `uv run python experiments/run_weak_error.py` |
| Terminal-law KS/W1 diagnostics (`figures/distributional_diagnostics.pdf`) | `uv run python experiments/run_distributional.py` |
| CEV convergence, beta = 0.5 and 0.75 (`figures/cev_convergence.pdf`) | `uv run python experiments/run_cev_experiment.py` |
| KLM Fig.-3-style order sweep, JAX (`outputs/klm_fig3_jax/fig3_orders.pdf`) | `uv run python experiments/klm_fig3_jax.py --quick` (drop `--quick` for the full sweep) |
| Chapter-2 background figures | `uv run python experiments/fig2_*.py` |

Add `--n-paths 2000` (or similar) to any `run_*` script for a fast smoke run.
See `docs/reproducibility_manifest.md` for the current thesis command list,
output policy, and claim boundaries.

Production strong-error figures use `reference_n_steps = 32768` by default.
Before quoting fitted slopes, run the reference-sensitivity gate comparing
`4096`, `16384`, and `32768` strong-reference steps and Fig. 3 reference
powers `14`, `16`, and `18`. In the alpha < 0 regimes D and E the ProjEuler
and KLM fitted slopes are not reference-converged even at `32768` (they move
monotonically with reference resolution); report them as coupled-diagnostic
slopes with the sensitivity range, not as measured true rates. This is
consistent with theory: every scheme built on equidistant Brownian increments
has strong order at most delta/2 for delta < 2 (Hefter--Jentzen), and the HH
reference itself has proven L1 rate only min(1, delta)/2
(Hefter--Herzwurm 2018).

## Scheme conventions and known limitations

- **FTE** follows Lord–Koekkoek–van Dijk: the possibly negative auxiliary
  variable is carried forward and only the read-off value is truncated.
- **HH Milstein** is the fine-grid reference for the main Brownian-coupled
  strong-error benchmark. It is reference infrastructure, not a ranked
  benchmark curve in `experiments/run_strong_error.py`.
- **KL splitting** is uniform only when alpha >= 0 (delta >= 1). In regimes
  D and E the strong-error scripts and benchmark-suite Kaggle notebooks plot
  the Brownian-coupled adaptive soft-zero Kelly-Lord variant instead, recorded
  as `scheme_variant=adaptive-soft-zero`. Treat those D/E KL rows as
  exploratory variant comparisons, not as the uniform KL theorem-covered
  scheme used in regimes A-C.
- **KLM backstopped adaptive** uses the drift-implicit backstop when
  alpha > 0 and a projected fallback otherwise. Outside its design regime
  (alpha < 0: regimes D, E) the backstop becomes the typical mode and the
  scheme carries a documented positive terminal-mean bias — see
  `notes/diagnostics.md` (01.07.2026). Report those runs as exploratory.
- **BLT splitting** (Brehier–Cohen–Herzwurm–Kelly–Neuenkirch, in preparation
  2026) composes the exact squared-Bessel(1) flow with the exact linear-ODE
  flow; it consumes joint (increment, running-infimum) noise, which places it
  outside the Hefter–Jentzen equidistant-increment information class. Proven
  L1 order 1 (up to logs) for delta > 2 and delta/2 - eps for delta in (1, 2];
  regimes D/E use the paper's modified scheme (positive part), which has a
  zero atom and no theory yet — report as exploratory. A BLT step is also
  available as a KLM backstop (`backstop="blt"`): it retakes the failed
  explicit step with the SAME Brownian increment plus one auxiliary
  exponential for the bridge infimum.
- **Strong-error coupling**: adaptive KLM steps and adaptive KL soft-zero steps
  are quantised down to whole multiples of the fine reference grid, so schemes
  and reference share one Brownian path exactly.
- **Precision**: all strong-error work is FP64; JAX is configured with
  `jax_enable_x64` (see thesis background chapter for why FP32 is
  insufficient at the 1e-5 error scale).

## Tests

`pytest -q` runs ~120 tests: sampler shape/positivity/moment checks against
the exact CIR law, Brownian-coupling checks, config consistency, KS/W1
sanity bounds, CEV beta = 1/2 exact reduction to CIR, and bit-level CPU/JAX
parity of the KLM scheme (terminal values and backstop-trigger counts).
