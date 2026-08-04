# Runbook — matched reference-sensitivity ladder (regimes D and E)

Produces reference-sensitivity evidence commensurable with the canonical
strong-error benchmark, so `fig_order_summary.py` will draw the bars and
§res-ref-sensitivity can state convergence rather than caveat it.

## Why only D and E

The drift is a boundary-regime phenomenon: ProjEuler in E falls
0.47 → 0.38 → 0.34 with reference refinement, KLM rises 0.21 → 0.27 → 0.35.
A and C are stable to two or three decimals in the existing ladder (FTE moves
< 1e-3). It is **unlikely** that shifting the level window changes that, but
it is **untested on the canonical window** — say so rather than claiming the
A–C conclusion is established there.

Cost also decides it: all five regimes at three rungs is ~23 h. D alone and
E alone are ~7 h each, comfortably inside Kaggle's 12 h cap.

## Configuration

Notebook: `notebooks/kaggle/kaggle_reference_ladder_JAX.ipynb`
Accelerator: **GPU P100**. Internet: not required (standalone; no clone).

Add one cell **above** the main cell:

```python
import os
os.environ["CIR_SUITE_RUN_MODE"]       = "full"
os.environ["CIR_SUITE_REGIMES"]        = "D"          # then "E" in session 2
os.environ["CIR_SUITE_REFERENCE_GRID"] = "262144,1048576,4194304"
```

Everything else takes the full-mode defaults, which are the canonical ones:

| | Value | |
|---|---|---|
| `N_PATHS` | 20,000 | matches the canonical benchmark |
| `COARSE_N_STEPS` | 8,16,32,64,128,256 | = h 2^-3…2^-8, matches Table 6.1 |
| reference rungs | 2^18, 2^20, 2^22 | finest rung **is** the canonical reference |
| `ADAPTIVE_GRID_STEPS` | 131072 (2^17) | |
| `CHUNK_STEPS` | 32768 (2^15) | |
| path batches | ~3,050 (8 GB budget) | ~7 batches, so a checkpoint roughly hourly |

The 2^22 rung is regenerated inside each run rather than reused from
`f1a1377ad11a430f` / `0ceb47d539211b6e`. Those came from a different notebook
with a different batch and chunk decomposition, and increments are drawn from
`batch_chunk_key(batch_index, chunk_index)` — so their 2^22 is not paired with
this run's 2^18 and 2^20, which is the whole point of the shared-path design.

## Resuming after a session kill

`/kaggle/working` does not survive a hard termination. To resume:

1. Save a version of the interrupted run so its output persists.
2. In the new session, **Add data → Notebook Output** and attach that run.
3. Start it with the *identical* env cell.

The notebook finds `ladder_checkpoint.json` under `/kaggle/input`, copies it
in, and continues from the last completed batch. It refuses any checkpoint
whose experiment signature differs — including a different path-batch or
chunk setting, since rebatching changes the Brownian draws.

Resumed totals are statistically and numerically equivalent to an
uninterrupted run. They are not bit-identical: summing per-batch
contributions in a different order changes the floating-point reduction
order.

## After both runs

1. Merge `jax_strong_error_all_references.csv` from both into
   `results/reference_sensitivity/strong_reference_sensitivity_orders.csv`.
2. Re-run `python experiments/fig_order_summary.py`. It prints
   `matched ladder for ['D', 'E']` and draws bars **for those regimes only**;
   A/B/C stay suppressed and that is correct.
3. Copy `figures/fig_order_vs_delta_summary.pdf` into the thesis.
4. Rewrite §res-ref-sensitivity: the D/E incommensurability caveat becomes a
   convergence result. **Scope the removal to D/E** — A–C remains on the
   2^-4…2^-9 window and its caveat stands.
