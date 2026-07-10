# From NumPy loops to JAX kernels: fixed-step and adaptive SDE schemes on a GPU

*A pedagogical companion to the thesis's computational chapter. Every example
is taken from (or simplified from) this repository, so you can open the real
file next to each section.*

---

## 0. The mental model: what a GPU actually parallelises

A GPU is not "a thousand independent little CPUs". It is closer to **one CPU
with enormously wide registers**: thousands of lanes that all execute *the
same instruction at the same moment*, each on its own piece of data. This
execution model is called SIMT (single instruction, multiple threads).

Monte Carlo SDE simulation fits this model with one decision:

> **Parallelise over paths. Stay sequential over time.**

Time stepping is *inherently* sequential — `X_{n+1}` needs `X_n`, so no
machine can compute step 100 before step 99. But at any given step, the
20,000 paths of the ensemble need identical arithmetic on different numbers.
So the unit of GPU work is the **ensemble step**: "advance every path by one
step" is a single vectorised operation, and the simulation is a sequential
loop of such operations.

Everything in this document — fixed-step, adaptive, mutable, immutable — is
a variation on that one sentence.

---

## 1. A fixed-step scheme, twice

### 1.1 The NumPy version (what we call *reference logic*)

Here is the heart of full truncation Euler as implemented in
`src/samplers/full_truncation_euler.py`:

```python
def fte_terminal_from_dW(X0, kappa, theta, sigma, dt, dW):
    n_paths, n_steps = dW.shape
    x_aux = np.full(n_paths, X0)          # state: one value per path

    for n in range(n_steps):              # SEQUENTIAL over time
        x_pos = np.maximum(x_aux, 0.0)    # VECTORISED over paths
        x_aux = x_aux + kappa*(theta - x_pos)*dt + sigma*np.sqrt(x_pos)*dW[:, n]

    return np.maximum(x_aux, 0.0)
```

Notice the structure: a plain Python `for` over steps, and inside it, array
operations that touch all paths at once. NumPy is already "parallel over
paths" in spirit — each line is a loop over 20,000 paths written in C. What
NumPy cannot do is (a) run those array operations on a GPU, or (b) fuse the
five operations of the loop body into one pass over memory. Each NumPy line
allocates a temporary and makes a full round trip through memory.

### 1.2 The JAX version

```python
import jax, jax.numpy as jnp
from jax import lax

@jax.jit
def fte_terminal_from_dW_jax(X0, kappa, theta, sigma, dt, dW):
    n_paths, n_steps = dW.shape

    def one_step(x_aux, dW_col):                       # (carry, x_n) -> (carry', y_n)
        x_pos = jnp.maximum(x_aux, 0.0)
        x_new = x_aux + kappa*(theta - x_pos)*dt + sigma*jnp.sqrt(x_pos)*dW_col
        return x_new, None                             # new carry, nothing stacked

    x_final, _ = lax.scan(one_step, jnp.full(n_paths, X0), dW.T)
    return jnp.maximum(x_final, 0.0)
```

Three things changed, and only three:

1. **The `for` loop became `lax.scan`.** `scan` is "run this step function
   `n_steps` times, threading a *carry* through" — a compiled loop instead
   of a Python loop. The carry **is** the mathematical state `x_n`; the
   per-iteration input is the `n`-th column of increments.
2. **`np` became `jnp`.** Same API, but operations build a computation graph
   that XLA (the compiler behind JAX) fuses and compiles for the device.
   The five operations of the loop body become roughly one GPU kernel.
3. **Nothing is assigned in place.** `one_step` takes a state and *returns a
   new one*. Which brings us to your second question.

A real example of exactly this pattern in the repo:
`if_terminal_from_fine_dW_jax` in `src/jax_schemes.py` is the drift-implicit
reference scheme written as a `lax.scan` — twelve lines.

---

## 2. Immutability: why it is not a restriction for us

### 2.1 Our schemes never actually mutate anything

Look at the mathematics: every scheme in this thesis is a **recurrence**

```
x_{n+1} = F(x_n, ΔW_{n+1})
```

There is no "modify x" in that statement. There is an old state and a new
state. When the NumPy code writes

```python
X[:, n+1] = fte_step(X[:, n], ...)      # paths version: writes into a matrix
x[bad] = backstop_values                # adaptive version: scatter into a slice
```

the *mutation is an implementation convenience for memory reuse*, not part
of the algorithm. NumPy lets you overwrite because it is cheap; the
algorithm never required it. JAX simply forces you to write the recurrence
as it actually is:

| NumPy idiom (mutation) | JAX idiom (new value) |
|---|---|
| `for n in range(N): x = step(x)` | `x, _ = lax.scan(step, x0, xs)` |
| `while np.any(active): ...` | `lax.while_loop(cond, body, state)` |
| `x[mask] = v[mask]` | `x = jnp.where(mask, v, x)` |
| `X[:, n+1] = x_new` | stacked `ys` output of `scan`, or `X = X.at[:, n+1].set(x_new)` |
| `if x.min() < 0: raise` | `x = jnp.where(x < 0, fallback, x)` (no data-dependent Python control flow) |
| `rng.standard_normal(k)` (stateful) | `jax.random.normal(key, (k,))` (explicit key) |

Every left-hand idiom appears in our NumPy samplers; every right-hand one in
`src/jax_schemes.py`. The translation is mechanical once you see that mutation
was never load-bearing.

### 2.2 Why JAX insists — and why you win

`jax.jit` works by **tracing**: it runs your function once with abstract
arrays, records every operation, and hands the whole graph to XLA to
reorder, fuse, and compile. That is only sound if the function is
*referentially transparent* — same inputs, same outputs, no hidden effects.
In-place mutation creates hidden effects: if two names alias one buffer, the
compiler can no longer prove that reordering two operations is safe.
Immutability is not an aesthetic; it is the license that lets the compiler
generate aggressive GPU code.

### 2.3 "But doesn't copying every array kill performance?"

No — and this is the part people miss. Immutability is a property of the
**language**, not of the **machine code**. After XLA compiles the graph, it
performs liveness analysis: if `x_old` is never used after `x_new =
x_old + ...` is computed, XLA writes `x_new` **into `x_old`'s buffer**. The
update is functional in your source and in-place in the executable. In other
words: NumPy makes *you* do the memory reuse (by mutating); JAX has the
*compiler* do it (after proving it safe). The same holds for
`x.at[i].set(v)`: it reads as "return a copy with slot `i` changed", and it
compiles to an in-place scatter whenever the old array is dead.

### 2.4 The RNG bonus you get for free

NumPy's generator is a mutable object: every draw silently changes hidden
state, so "give me chunk 517 of the Brownian path again" is impossible
without having saved the state. JAX's randomness is **counter-based**: a key
is an ordinary immutable value, and `fold_in(key, k)` deterministically
derives the key for chunk `k`. This is exactly what makes the
`h_ref = 2^-25` streaming experiments possible
(`experiments/klm_fig2a_streaming.py`): pass 2 *regenerates* the identical
33 million Brownian increments chunk by chunk, storing none of them —

```python
def make_brownian_increment_chunk(chunk_number):
    chunk_key = jax.random.fold_in(PATH_KEY, chunk_number)   # pure function of k
    return jax.random.normal(chunk_key, shape) * jnp.sqrt(REFERENCE_STEP)
```

Immutable RNG state is not a workaround; for replay-based algorithms it is
strictly more powerful than the mutable kind.

---

## 3. Adaptive schemes: why per-path step sizes do not break SIMT

This is the crux, so let us set up the worry precisely.

> *The adaptive rule gives every path its own step size
> `h = h_max · min(1, |Y_n|)`. Path 7 might need 300 steps and path 8,
> which visits the boundary, might need 3,000. Doesn't "same instruction on
> every lane" fail the moment two lanes disagree about what to do next?*

It does not, for three reasons, and each one is a specific implementation
device.

### 3.1 The step size is **data**, not code

Look at the explicit update:

```python
y_new = y + (alpha/y + beta*y) * h + gamma * dW
```

This is *one* arithmetic expression. It does not matter that lane 7 holds
`h = 0.031` and lane 8 holds `h = 0.0005`: both lanes execute
multiply–add–multiply–add on their own operands. Adaptivity changes the
**values** flowing through the pipeline, never the **instructions**. The
same holds for the increment: lane `i` needs `ΔW ~ N(0, h_i)` — a
per-lane variance is just another array.

The subtle mental shift: after `k` rounds of the loop, the lanes are at
*different physical times* `t_i`. That is fine — each lane carries its own
clock `t_i` (or grid position) **as part of the state vector**. Lockstep
means "same round index", not "same simulated time".

### 3.2 Branches become masks: compute both, select

The backstop logic reads like control flow:

```python
if h_proposed < h_min:        # trigger (a)
    take a backstop step at h_min
else:
    y_star = explicit_step(...)
    if y_star <= 0:           # trigger (b)
        retake with the backstop, same dW
```

On a GPU you do not branch — you **evaluate both branches for every lane and
select pointwise**. From `src/jax_schemes.py`, lightly abridged:

```python
y_explicit = y + (alpha/y + beta*y)*h + gamma*dW        # everyone computes this
y_backstop = backstop_fn(y, h, dW, alpha, beta, gamma)  # ...and this

neg_retake   = (~min_triggered) & (y_explicit <= 0.0)   # boolean masks: data again
use_backstop = min_triggered | neg_retake

y_next = jnp.where(use_backstop, y_backstop, y_explicit)  # pointwise select
```

Cost: every lane computes one update formula it will discard (~2x arithmetic
on the update, negligible next to memory traffic). Benefit: there is no
divergence at all — the instruction stream is identical on every lane, every
round. The `if` has been *reified into a boolean array*.

This is also why the two implementations can agree bit-for-bit: the NumPy
version applies the backstop to the sub-array `y[bad]` (mutation on a
boolean slice), the JAX version computes both everywhere and `where`-selects
— **the same values land in the same places** (verified per-path, including
trigger counts, in `tests/test_klm_parity.py`).

### 3.3 Rounds, ragged finishing, and no-op padding

Different paths finish after different numbers of rounds. The ensemble loop
is a `lax.while_loop` whose condition is "does *any* lane still have time
left?":

```
round:      1    2    3    4    5    6    7    8    9   ...
lane 1:     h    h    h    h    DONE .    .    .    .        (interior path)
lane 2:     h    h    h/4  h/16 h/16 h/4  h    h    DONE     (boundary visit)
lane 3:     h    h    h    DONE .    .    .    .    .
                              ^^^ lanes 1,3 idle: zero-length no-op steps
```

A finished lane takes **zero-length steps**: `h = 0`, `ΔW = 0`, and every
update map in the scheme satisfies `F(y, 0, 0) = y` — the state is a fixed
point. (Check it against each formula: explicit, drift-implicit, projected —
all return `y` exactly at `h = 0`.) The lane still occupies hardware and
still executes instructions; it just computes the identity.

So the honest statement is:

> **Ragged finishing costs utilisation, never correctness.**

The waste is quantifiable: with `n_i` steps on path `i` and `N` lanes, the
loop runs `max_i n_i` rounds and pays `N · max_i n_i` lane-rounds for
`Σ_i n_i` of useful work. The utilisation ratio measured for the KLM scheme
on this repo's regime grid (`h_max = 2^-6`, `ρ = 2^6`, 4,000 paths):

| Regime | mean steps | max steps | utilisation `Σnᵢ/(N·max nᵢ)` |
|---|---|---|---|
| A (δ=16)  | 470  | 745  | 63% |
| C (δ=2)   | 686  | 2086 | **33%** |
| E (δ=0.25)| 1760 | 3178 | 55% |

Worst case ≈ 3x overhead — a tax, not a defeat, against thousands of lanes.
(Note the shape: the *worst* regime is the Feller boundary C, not the most
singular regime E. Cost is driven by **heterogeneity** — in C a minority of
paths visit the boundary and take 3x the median steps; in E almost *every*
path is boundary-dominated, so the workload re-homogenises.)

### 3.4 Why none of this changes the statistics

Three invariants, all checkable, make lockstep execution a pure performance
matter:

1. **No-op exactness.** Finished lanes are fixed points of the loop body
   (Section 3.3), so extra rounds change nothing.
2. **Randomness independence.** No lane's noise may depend on another
   lane's trajectory. In the coupled experiments the increments are
   pre-generated (or bridge-sampled from a shared fine path), so this holds
   trivially. Caveat worth knowing: the *free-running* NumPy sampler draws
   `rng.standard_normal(active.size)` per round, so *which* draw lands on
   *which* path depends on the active set — the joint law is unchanged
   (i.i.d. draws, each used once), but bit-level output depends on batch
   layout. A reproducibility caveat, not a bias.
3. **Reduction hygiene.** Terminal statistics average path values only;
   there are no padded phantom entries to accidentally include (every lane
   is a real path run to T).

Under 1–3, the computation performed *for each path* is identical to what a
serial, one-path-at-a-time run would perform. And that is the deep reason
the advisor's remark — *"the sample space is unaffected"* — is true: the
adaptive rule `h_{n+1} = g(Y_n)` is a function of the current state, hence
(by induction) a functional of the *same Brownian path* the scheme is
driven by; it is `F_{t_n}`-measurable, never peeking at future noise. The
scheme is therefore one fixed deterministic map from (Brownian path, seed)
to output. Serial NumPy, vectorised NumPy, and lockstep JAX are three
*execution schedules* for the same map — and a deterministic map's output
does not depend on the schedule used to evaluate it. The parity test is the
constructive proof: same increments in, same terminal values and same
trigger counts out, to 1e-9.

---

## 4. The full adaptive kernel, annotated

The complete state of the coupled KLM kernel
(`src/jax_schemes.py::klm_backstop_terminal_from_fine_dW_jax`) is three
arrays:

```python
y        # (n_paths,)  current Lamperti state, one per lane
pos      # (n_paths,)  each lane's position on the fine reference grid = its clock
counters # (3,)        total steps, min-triggers, negativity-retakes
```

The loop:

```python
def cond(state):
    y, pos, counters = state
    return jnp.any(pos < n_fine)              # any lane not yet at T?

def body(state):
    y, pos, counters = state
    active = pos < n_fine

    # 1. every lane proposes its own step (DATA, not control flow)
    h_prop = h_max * jnp.minimum(1.0, jnp.abs(y))
    m = jnp.floor(h_prop / dt_fine).astype(jnp.int64)     # quantise to fine grid

    # 2. trigger (a) as a mask; clamp to the remaining horizon
    min_triggered = m < m_min
    m = jnp.where(min_triggered, m_min, jnp.minimum(m, m_max))
    m = jnp.minimum(m, n_fine - pos)          # finished lanes get m = 0 -> no-op

    # 3. Brownian increment = partial sum of the SHARED fine path (coupling)
    h  = m * dt_fine
    dW = W[rows, pos + m] - W[rows, pos]      # gather; W is the cumulative path

    # 4. compute BOTH updates, select with masks (Section 3.2)
    y_explicit = jnp.where(h > 0.0, y + (alpha/y + beta*y)*h + gamma*dW, y)
    y_backstop = jnp.where(h > 0.0, backstop_fn(y, h, dW, ...), y)
    neg_retake   = (~min_triggered) & (y_explicit <= 0.0)
    use_backstop = min_triggered | neg_retake
    y_next = jnp.where(use_backstop, y_backstop, y_explicit)

    # 5. masked bookkeeping (only active lanes count)
    counters += jnp.array([jnp.sum(active),
                           jnp.sum(active & min_triggered),
                           jnp.sum(active & neg_retake)])
    return y_next, pos + m, counters

y, pos, counters = lax.while_loop(cond, body, (y0, pos0, counters0))
```

Read it against Section 3: step size as data (1), triggers as masks (2, 4),
per-lane clocks (`pos`), no-op padding for finished lanes (`m = 0` in 2–3),
branchless selection (4), reduction hygiene (5). There is not a single
`if` on a traced value in the whole kernel.

One JAX-specific detail deserves a comment: the NumPy version processes only
the active sub-ensemble (`idx = np.flatnonzero(active)`, then works on
`y[idx]`). JAX **cannot** do this inside `jit` — `y[active]` would have a
data-dependent *shape*, and XLA compiles fixed-shape programs only. This is
not a loss: on a GPU, computing the no-op lanes costs (almost) nothing extra
because the lanes are there whether you use them or not. Static shapes are
the price of compilation; masks are how you pay it.

---

## 5. Fixed vs adaptive on the GPU: the summary picture

| | fixed-step | adaptive (lockstep) |
|---|---|---|
| loop construct | `lax.scan` (trip count known) | `lax.while_loop` (data-dependent) |
| rounds needed | `n_steps`, same for all lanes | `max_i n_i`, set by the slowest path |
| per-round work | one update | both updates + selects (~2x arithmetic) |
| utilisation | ~100% | `Σnᵢ/(N·max nᵢ)` — measured 33–63% here |
| statistics | exact | exact (invariants 1–3 of §3.4) |
| increments | column slice of a pre-generated array | per-lane variance `N(0, hᵢ)`, or gather/bridge from a shared fine path |

The conclusion the thesis draws (Chapter 6/7): adaptivity converts the
boundary difficulty into a *load-imbalance overhead*, bounded here by a
factor ~3 — while remaining exactly, provably, the same random variable the
serial algorithm defines.

## 6. Practical gotchas (each one has bitten this project)

- **Enable float64 first.** `jax.config.update("jax_enable_x64", True)`
  *before* array creation; JAX defaults to float32, which cannot resolve
  strong errors at the 1e-5 scale (thesis Ch. 2). All repo kernels set it.
- **Time correctly.** Dispatch is asynchronous: without
  `result.block_until_ready()` you time the *launch*, not the work. First
  call includes compilation — measure it separately, then time steady state.
- **Recompilation.** `jit` specialises on shapes and dtypes; changing
  `n_paths` triggers a fresh compile. Keep benchmark shapes fixed.
- **Keys are not seeds.** Never reuse a key for two draws; `split`/`fold_in`
  and thread keys explicitly (see any `src/jax_schemes.py` function).
- **Cross-backend RNG differs.** NumPy (PCG64) and JAX (threefry) produce
  different streams from "the same seed" — compare implementations only on
  pre-generated increments (as the parity tests do), or at distribution level.

## 7. Where to look in this repository

| Concept | File |
|---|---|
| NumPy fixed-step reference | `src/samplers/full_truncation_euler.py`, `lamperti_implicit.py` |
| JAX fixed-step (`scan`) | `src/jax_schemes.py::if_terminal_from_fine_dW_jax` |
| NumPy adaptive (masked, mutable) | `src/samplers/klm_backstop.py` |
| JAX adaptive (`while_loop`, immutable) | `src/jax_schemes.py::klm_backstop_terminal_from_fine_dW_jax` |
| Bit-level parity between the two | `tests/test_klm_parity.py` |
| Counter-based RNG replay at 2^-25 | `experiments/klm_fig2a_streaming.py` (`make_brownian_increment_chunk`) |
| Standalone CIR benchmark notebooks | `notebooks/kaggle/kaggle_cir_benchmark_suite.ipynb` (NumPy), `notebooks/kaggle/kaggle_cir_benchmark_suite_JAX.ipynb` (JAX) |
| Standalone KLM paper notebooks | `notebooks/kaggle/kaggle_klm_fig2a.ipynb` (NumPy), `notebooks/kaggle/kaggle_klm_fig2a_JAX.ipynb` (JAX), `notebooks/kaggle/kaggle_klm_fig3_full.ipynb`, `notebooks/kaggle/kaggle_klm_fig3_full_JAX.ipynb` |
| KL adaptive-splitting JAX/GPU reproduction | `experiments/kl_adaptive_splitting_paper_jax.py`, `notebooks/kaggle/kaggle_kl_adaptive_splitting_paper_JAX.ipynb` |
| Utilisation measurements | thesis Ch. 7, Table res-utilisation |
