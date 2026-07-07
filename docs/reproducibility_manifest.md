# Reproducibility Manifest

Status date: 3 July 2026

All thesis-facing runs use the shared seed in `configs/experiments.yaml`
(`master_seed: 120339106`) unless a script documents a more specific seed.

Generated figures, CSVs, and large experiment artifacts are working outputs.
They should be regenerated from the commands below rather than committed by
default. The repository ignores `figures/*.pdf`, `figures/*.png`, `results/`,
`outputs/`, and CSV files.

## Current Thesis Commands

| Purpose | Command | Main outputs | Status |
|---|---|---|---|
| CIR strong-error benchmark against HH fine reference | `python experiments/run_strong_error.py` | `results/strong_error_regime_{R}.csv`, `figures/strong_error_regime_{R}.pdf` | Thesis benchmark target |
| Reference-sensitivity gate | `python experiments/run_reference_sensitivity.py`; full Kaggle route: `notebooks/kaggle/kaggle_reference_sensitivity.ipynb` | `outputs/reference_sensitivity/strong_reference_sensitivity*.csv`, `outputs/reference_sensitivity/fig3_reference_sensitivity*.csv`; Kaggle writes `/kaggle/working/cir_reference_sensitivity_outputs` | Required before quoting production fitted slopes |
| Fast CIR strong-error smoke | `python experiments/run_strong_error.py --n-paths 2000 --regimes A C E` | Same output names for selected regimes | Supervisor/examiner smoke run |
| Cross-regime order summary | `python experiments/fig_order_summary.py` | `results/fig_order_vs_delta_summary.csv`, `figures/fig_order_vs_delta_summary.pdf` | Requires strong-error CSVs first |
| KLM backstop diagnostic | `python experiments/run_klm_diagnostic.py` | `results/klm_backstop_diagnostic.csv`, `figures/klm_backstop_diagnostic.pdf` | Supports KLM limitation wording |
| KL adaptive-splitting paper Fig. 2/3 smoke | `python experiments/kl_adaptive_splitting_paper.py --mode smoke` | `results/kl_adaptive_splitting_smoke_errors.csv`, `results/kl_adaptive_splitting_smoke_rates.csv`, `figures/kl_adaptive_splitting_smoke_combined.pdf` | Paper-reproduction smoke; full run via `notebooks/kaggle/kaggle_kl_adaptive_splitting_paper.ipynb` |
| KL adaptive-splitting paper Fig. 2/3 JAX smoke | `python experiments/kl_adaptive_splitting_paper_jax.py --mode smoke` | `results/kl_adaptive_splitting_smoke_jax_errors.csv`, `results/kl_adaptive_splitting_smoke_jax_rates.csv`, `figures/kl_adaptive_splitting_smoke_jax_combined.pdf` | GPU/JAX implementation smoke; full run via `notebooks/kaggle/kaggle_kl_adaptive_splitting_paper_JAX.ipynb` |
| Terminal-law diagnostics | `python experiments/run_distributional.py` | `results/distributional_diagnostics.csv`, `figures/distributional_diagnostics.pdf` | Uses exact law as terminal comparator |
| Weak error g1-g4 benchmark | `python experiments/run_weak_error.py` | `results/weak_error.csv`, `results/weak_error_orders.csv`, `figures/weak_error_regime_{R}.pdf` | Exact-comparator weak errors; fits refused below the 2x MC-s.e. floor; g4 not defined for adaptive schemes |
| BLT splitting strong error (dual reference) | `python experiments/run_blt_strong_error.py` | `results/blt_strong_error.csv`, `figures/blt_strong_error.pdf` | Validates BLT implementation against the paper's rates; D/E rows are the modified scheme, exploratory |
| KLM backstop-candidate comparison | `python experiments/run_klm_backstop_comparison.py` | `results/klm_backstop_comparison.csv`, `figures/klm_backstop_comparison.pdf` | implicit vs projected vs BLT backstops against the exact terminal law |
| CPU vs JAX scheme timing | `python experiments/run_cpu_gpu_timing.py` | `results/cpu_gpu_timing.csv`, `figures/cpu_gpu_timing.pdf` | Backend recorded per row; GPU claims only from Kaggle P100 runs of this script |
| CEV minimum extension | `python experiments/run_cev_experiment.py` | `results/cev_convergence.csv`, `figures/cev_convergence.pdf` | Exploratory CEV extension |
| JAX KLM Fig. 3 quick run | `python experiments/klm_fig3_jax.py --quick` | `outputs/klm_fig3_jax/results.csv`, `outputs/klm_fig3_jax/fitted_orders.csv`, `outputs/klm_fig3_jax/fig3_orders.pdf` | CPU/JAX methodology artifact |

## Standalone Kaggle Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `notebooks/kaggle/kaggle_cir_benchmark_suite.ipynb` | Single-cell CPU/NumPy monolithic benchmark suite | Standalone; no repo imports |
| `notebooks/kaggle/kaggle_cir_benchmark_suite_JAX.ipynb` | Single-cell GPU/JAX monolithic benchmark suite | Standalone; no repo imports |
| `notebooks/kaggle/kaggle_reference_sensitivity.ipynb` | Single-cell GPU/JAX reference-sensitivity gate | Standalone; use before production fitted-order claims |
| `notebooks/kaggle/kaggle_klm_fig2a.ipynb` | KLM Fig. 2(a) NumPy/CPU bridge-streaming port | Standalone; smoke by default |
| `notebooks/kaggle/kaggle_klm_fig2a_JAX.ipynb` | KLM Fig. 2(a) JAX/GPU bridge-streaming reproduction | Standalone paper-scale route |
| `notebooks/kaggle/kaggle_klm_fig3_full.ipynb` | KLM Fig. 3 full diagnostic, h_ref = 2^-25 | Standalone JAX/GPU paper-scale run |
| `notebooks/kaggle/kaggle_klm_fig3_full_JAX.ipynb` | Explicit `_JAX` copy of the KLM Fig. 3 notebook | Same embedded JAX source as base notebook |
| `notebooks/kaggle/kaggle_kl_adaptive_splitting_paper.ipynb` | KL adaptive-splitting paper Fig. 2/3 CPU/NumPy reproduction | Standalone; NumPy oracle |
| `notebooks/kaggle/kaggle_kl_adaptive_splitting_paper_JAX.ipynb` | KL adaptive-splitting paper Fig. 2/3 GPU/JAX port | Standalone JAX counterpart |

## Validation Commands

| Purpose | Command | Status |
|---|---|---|
| Full test suite | `python -m pytest tests -q -p no:cacheprovider` | Required before using outputs in the thesis |
| Focused registry/style checks | `python -m pytest tests/test_experiments_config.py tests/test_style.py -q -p no:cacheprovider` | Required after method-list edits |
| CPU/JAX parity | `python -m pytest tests/test_klm_parity.py tests/test_klm_jax_fig3.py -q -p no:cacheprovider` | Minimum support for JAX implementation claims |
| KL paper reproduction helpers | `python -m pytest tests/test_kl_adaptive_splitting_paper.py -q -p no:cacheprovider` | Checks paper thresholds, projected floor, and coupled smoke simulation |
| KL paper JAX helpers | `python -m pytest tests/test_kl_adaptive_splitting_paper_jax.py -q -p no:cacheprovider` | Checks JAX fixed/adaptive kernels and standalone notebook packaging |
| Standalone Kaggle notebook packaging | `python -m pytest tests/test_standalone_kaggle_notebooks.py -q -p no:cacheprovider` | Checks single-cell, syntax-valid, no-clone/no-repo-import notebook packaging |

## Claim Boundaries

- Strong-error figures compare benchmark schemes against the HH fine-grid
  reference on a shared Brownian path. The production default is
  `reference_n_steps = 32768`, and fitted slopes should be treated as
  provisional until the reference-sensitivity gate has compared `4096`,
  `16384`, and `32768`. The gate run of 2026-07-05 (post floor fix) shows the
  ProjEuler and KLM slopes in regimes D/E are NOT reference-converged at
  `32768`; quote them only with the sensitivity range attached.
- The compact JAX Fig. 3 fitted-order sweep defaults to `reference_power = 16`.
  Values above order one should not be quoted as final thesis evidence unless
  they survive the `14`, `16`, `18` reference-power sensitivity rerun.
- Exact transition sampling is used for terminal-law diagnostics, not as a
  Brownian-coupled strong-error reference.
- CPU/JAX parity supports implementation correctness. GPU speedup or scaling
  should only be claimed after measured timings are recorded with hardware and
  software details.
- CEV results are exploratory unless a separate CEV backstop convergence proof
  is added.
- The KL adaptive-splitting paper reproduction uses the paper's HH/Milstein
  reference step `dt_ref = 1e-5`. This is separate from the KLM paper's
  `2^-25` reference scale.
- In the CIR strong-error benchmark, KL rows for regimes D and E are the
  Brownian-coupled adaptive soft-zero Kelly-Lord variant, recorded as
  `scheme_variant=adaptive-soft-zero`. Uniform KL remains the A-C variant.
- For the KL adaptive-splitting reproduction, the NumPy notebook remains the
  paper-fidelity oracle; the `*_JAX.ipynb` notebook is the GPU implementation
  counterpart.
- The KLM Fig. 2(a) base notebook is the NumPy/CPU port and the `_JAX`
  notebook is the GPU paper-scale route. The KLM Fig. 3 full notebooks remain
  JAX-native because the paper-scale h_ref = 2^-25 diagnostic is maintained as
  a streaming GPU implementation.
