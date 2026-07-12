# Kaggle notebooks

Single-cell standalone notebooks for the heavy thesis experiments. Each
notebook contains the numerical kernels it needs directly in the cell, so it
can be pasted into Kaggle without a git clone, a pushed branch, or imports from
the local repository.

## Before use

1. Import the `.ipynb` into Kaggle, or paste the single code cell into a new
   notebook.
2. For JAX notebooks, set Accelerator = GPU P100.
3. Internet can stay off unless Kaggle is missing a standard package from the
   base image.
4. Run the single cell. Outputs are archived under `/kaggle/working`.

Most notebooks default to the intended full-scale run. For a quick integrity
check, set the appropriate environment variable before running the cell:

- CIR suite notebooks: `CIR_SUITE_RUN_MODE=smoke` (A-E; every regime and all
  benchmark schemes are included)
- Reference-sensitivity gate: `REFERENCE_SENSITIVITY_RUN_MODE=smoke`
- KLM Fig. 2(a): `FIG2A_RUN_MODE=smoke`
- KLM Fig. 3 full diagnostic: `FIG3_FULL_DIAG_RUN_MODE=smoke`

For production JAX CIR benchmark-suite runs, the full-mode A-E strong
benchmark now uses a streamed HH reference with `CIR_SUITE_REFERENCE_POWER=22`
(`h_ref = 2^-22`) by default, with `CIR_SUITE_N_PATHS=20000` (the thesis
methodology-chapter path count; use the checkpoint/resume support to split the
run across Kaggle sessions). The run is batched with
`CIR_SUITE_PATH_BATCH_SIZE` and `CIR_SUITE_CHUNK_STEPS`, so it should be run on
Kaggle GPU rather than locally. The compact JAX Fig. 3 sweep remains separate
and defaults to `CIR_SUITE_FIG3_REFERENCE_POWER=16`.

The reference-ladder notebook is the deep sensitivity gate: it measures every
scheme against HH references at all rungs of `CIR_SUITE_REFERENCE_GRID`
(default `4096,16384,32768`) built from one shared fine Brownian path. For the
KLM-paper-depth D/E evidence run it with
`CIR_SUITE_REFERENCE_GRID="32768,1048576,33554432"` (2^15, 2^20, 2^25),
`CIR_SUITE_ADAPTIVE_GRID_STEPS=131072`. Memory is bounded by
`CIR_SUITE_MEM_BUDGET_GB` (default 8, sized for the 16 GB P100) via automatic
path batching; a streamed-vs-materialised HH parity check runs at startup.

## Contents

| Notebook | Lane | Scale / runtime |
|---|---|---|
| `kaggle_cir_benchmark_suite.ipynb` | CPU/NumPy monolithic cell | Strong error, KLM diagnostics, terminal-law diagnostics, CEV extension, and summary figures; CPU-bound |
| `kaggle_cir_benchmark_suite_JAX.ipynb` | GPU/JAX monolithic cell | JAX strong benchmark and JAX Fig. 3 order sweep |
| `kaggle_reference_sensitivity.ipynb` | GPU/JAX monolithic cell | Reference-sensitivity gate for strong benchmark and compact Fig. 3 fitted orders |
| `kaggle_reference_ladder_JAX.ipynb` | GPU/JAX monolithic cell | Deep reference ladder (streamed HH rungs to `h_ref = 2^-25` on one shared Brownian path): sensitivity CSV + order-vs-delta summary figure with bars |
| `kaggle_four_scheme_regime_rates_JAX.ipynb` | GPU/JAX monolithic cell | Four-scheme fitted-rate sweep over thesis regimes A-E, with side-by-side kappa panels |
| `kaggle_klm_fig2a.ipynb` | CPU/NumPy paper-logic port | KLM Fig. 2(a) bridge-streaming logic; full h_ref = 2^-25 by default, smoke available |
| `kaggle_klm_fig2a_JAX.ipynb` | JAX/GPU paper reproduction | KLM Fig. 2(a), h_ref = 2^-25, M = 1000; paper-scale route |
| `kaggle_klm_fig3_full.ipynb` | JAX/GPU paper reproduction | KLM Fig. 3 full diagnostic, 41 a-values x 2 kappas at h_ref = 2^-25 |
| `kaggle_klm_fig3_full_JAX.ipynb` | JAX/GPU paper reproduction | Explicit `_JAX` copy of the same Fig. 3 notebook for naming consistency |
| `kaggle_kl_adaptive_splitting_paper.ipynb` | CPU/NumPy standalone script | KL adaptive-splitting paper Fig. 2/3: HH/Milstein dt_ref = 1e-5, M = 1000, 20 batches |
| `kaggle_kl_adaptive_splitting_paper_JAX.ipynb` | JAX/GPU standalone script | JAX/GPU port of the KL adaptive-splitting paper Fig. 2/3 reproduction |

## Reproducibility notes

- `kaggle_four_scheme_regime_rates_JAX.ipynb` is a Fig. 3-style thesis extension: it keeps side-by-side `kappa=2` and `kappa=0.2` panels, uses the A-E regime ladder via `a = sigma^2/(2 kappa theta)`, and fits rates for FTE, ProjEuler, KL, and KLM against a streamed HH reference with `h_ref = 2^-22` by default.
- The KLM Fig. 2(a) notebooks are intentionally split: the base notebook is a
  NumPy/CPU bridge-streaming port, while `_JAX` is the GPU paper-scale route.
  Both default to full mode with `h_ref = 2^-25`; set
  `FIG2A_RUN_MODE=smoke` only for a quick integrity check.
- The KLM Fig. 3 full notebooks remain JAX-native because the paper-scale
  h_ref = 2^-25 diagnostic is a streaming GPU implementation.
- The CIR benchmark suite has both a CPU/NumPy notebook and a JAX/GPU
  notebook. Both are monolithic cells, not repo import wrappers. The JAX/GPU
  notebook is the production route for A-E strong-error figures and streams an
  HH reference grid with `h_ref = 2^-22`; the CPU notebook remains a compact
  CPU-bound diagnostic route.
- The reference-sensitivity notebook is the Kaggle route for the thesis slope
  gate. It compares strong-reference grids `4096,16384,32768` and compact
  Fig. 3 powers `14,16,18`, writing the four sensitivity CSVs and two summary
  PDFs under `/kaggle/working/cir_reference_sensitivity_outputs`.
- The reference-ladder notebook extends the strong-error gate to
  KLM-paper-depth references (streamed, so `2^-25` fits on the P100). All
  rungs share one fine Brownian path, so fitted-order drift across rungs is
  pure reference effect. Its `strong_reference_sensitivity_orders.csv` matches
  the schema `experiments/fig_order_summary.py` reads from
  `outputs/reference_sensitivity/`, so the local summary figure can draw its
  sensitivity bars directly from the Kaggle output.
- In the CIR benchmark notebooks, KL is the uniform Kelly-Lord scheme in
  regimes A-C. For regimes D-E the KL row is the Brownian-coupled adaptive
  soft-zero variant and is recorded as `scheme_variant=adaptive-soft-zero`.
- The KL adaptive-splitting paper notebooks use `dt_ref = 1e-5`, matching the
  KL paper figure. The KLM notebooks use `h_ref = 2^-25`, matching the KLM
  paper-scale reference.
- Every standalone notebook is guarded by
  `tests/test_standalone_kaggle_notebooks.py`, which checks that it is a
  single code cell, parses as Python, and does not depend on `git clone` or
  local repository imports.


Checkpointing note: the JAX CIR benchmark and four-scheme regime-rate notebooks now write batch contribution CSVs, partial diagnostics, `progress_manifest.json`, and `backups/latest_partial.zip` during execution. Re-running with the same configuration resumes completed batches using the recorded `run_config_hash`.
