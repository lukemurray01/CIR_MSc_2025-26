# Integrated from the Kaggle notebook
# 'full-diagnostic-jax-reconstruction-of-klm-fig-3.ipynb' (single code
# cell, verbatim). Only change: non-Kaggle output dir points at the repo
# results/ directory instead of the CWD. All experiment knobs were already
# environment variables (FIG3_FULL_DIAG_*); defaults reproduce the
# notebook outputs exactly (same PRNG keys, chunk fold-in, arithmetic).
"""
Full diagnostic JAX reconstruction of KLM Fig. 3.

Fig. 3 asks a different question from Fig. 2.  In Fig. 2 we fixed the model
parameters and estimated the strong error as a function of the mean step size.
In Fig. 3 we repeat that convergence experiment for many values of

    a = sigma^2 / (2 kappa lambda),

and plot the fitted strong convergence order against a.

The paper uses

    lambda = 0.05,
    Y0 = 0.02,
    kappa = 2.0 in panel (a),
    kappa = 0.2 in panel (b),
    40 uniformly spaced a values in [0.04, 1.6].

For a given kappa and a, sigma is therefore

    sigma = sqrt(2 kappa lambda a).

Important note about EF
-----------------------
In the numerical-method definition section of the paper, EF is the explicit
Higham-Mao discretisation of the original CIR variable X,

    X_{n+1} = X_n + h kappa(lambda - X_n)
              + sigma sqrt(|X_n|) Delta W_n.

That is the EF curve used here.  The Lamperti explicit fixed method EF_y can be
included as an extra diagnostic by setting PLOT_EF_Y_DIAGNOSTIC = True, but it is
not part of the paper's Fig. 3.

Why this script is chunked
--------------------------
The fine reference mesh h_ref = 2^-25 has 33,554,432 intervals.  We cannot store
the full Brownian path for M=1000 paths.  Instead, each chunk stores a short
piece of the Brownian path, scans through the fine intervals in that chunk,
updates the fine implicit reference, updates any coarse/adaptive scheme times
that land inside each fine interval by Brownian bridge, then discards the chunk.

This is intentionally the same fine-interval landing logic as the trusted
Fig. 2(a) bridge script.  It is less aggressive than the earlier chunk-level
landing optimisation, but it removes a possible source of fitted-rate bias.

This diagnostic run is designed to separate the paper-grid reproduction from
the noiseless-limit check.

The defaults are

    kappa = 2 and kappa = 0.2,
    40 paper-grid a values in [0.04, 1.6],
    one extra diagnostic point at a = 0,
    M = 1000,
    h_ref = 2^-25,
    FIG3_FULL_DIAG_FIXED_STEP_SOURCE = hmax,
    FIG3_FULL_DIAG_FIT_X_SOURCE = hmax.

The a = 0 point is not part of the paper-grid reproduction.  It is included
only to make the noiseless limit explicit: when sigma = 0, EF and FT should
recover deterministic order one.  The script therefore saves both

    * a paper-grid figure that excludes a = 0, and
    * an a = 0 diagnostic figure that includes it.

Extra diagnostics are saved:

    * RMSE for every method at every a value and every level,
    * h_mean, hmax, and h_mean / hmax,
    * counts/proportions of negative EF and FT candidates during fixed updates.

The code saves

    /kaggle/working/fig3_full_diagnostic_jax_h25_fixed-hmax_fit-hmax_paper_grid.png
    /kaggle/working/fig3_full_diagnostic_jax_h25_fixed-hmax_fit-hmax_with_a0.png
    /kaggle/working/fig3_full_diagnostic_jax_h25_fixed-hmax_fit-hmax.csv
    /kaggle/working/fig3_full_diagnostic_jax_h25_fixed-hmax_fit-hmax_diagnostics.csv
    /kaggle/working/fig3_full_diagnostic_jax_h25_fixed-hmax_fit-hmax.npz

Diagnostic step-size switches
-----------------------------
The paper text says that the fixed-step schemes EF, IF, and FT use the EA
average step h_mean.  To test the possibility that the printed Fig. 3 was made
with h_max instead, set

    FIG3_FULL_DIAG_FIXED_STEP_SOURCE=hmax

The fitted slope is also allowed to use either h_mean or h_max:

    FIG3_FULL_DIAG_FIT_X_SOURCE=hmean   # paper-text convention
    FIG3_FULL_DIAG_FIT_X_SOURCE=hmax    # full-diagnostic default
"""

from __future__ import annotations

import csv
import os
import time

# These must be set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
# If Kaggle memory is fragmented, uncomment this before importing jax:
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 1. Experiment configuration
# =============================================================================

LAMBDA = 0.05
FINAL_TIME = 1.0

# The paper gives Y0, and Y = sqrt(X), so X0 = Y0^2.
INITIAL_Y = 0.02
INITIAL_X = INITIAL_Y**2

# Full diagnostic defaults.  Environment variables can still override these values.
NUMBER_OF_PATHS = int(os.environ.get("FIG3_FULL_DIAG_M", "1000"))
REFERENCE_POWER = int(os.environ.get("FIG3_FULL_DIAG_REFERENCE_POWER", "25"))
NUMBER_OF_FINE_STEPS = 2**REFERENCE_POWER
REFERENCE_STEP = 2.0 ** (-REFERENCE_POWER)

NUMBER_OF_PAPER_A_VALUES = int(os.environ.get("FIG3_FULL_DIAG_N_A", "40"))
PAPER_A_VALUES_AS_NUMPY = np.linspace(0.04, 1.6, NUMBER_OF_PAPER_A_VALUES, dtype=np.float64)
INCLUDE_A0_DIAGNOSTIC = os.environ.get("FIG3_FULL_DIAG_INCLUDE_A0", "1") == "1"
A_VALUES_AS_NUMPY = (
    np.concatenate([np.array([0.0], dtype=np.float64), PAPER_A_VALUES_AS_NUMPY])
    if INCLUDE_A0_DIAGNOSTIC
    else PAPER_A_VALUES_AS_NUMPY
)
NUMBER_OF_A_VALUES = len(A_VALUES_AS_NUMPY)

if NUMBER_OF_PAPER_A_VALUES == 40:
    assert np.any(np.isclose(PAPER_A_VALUES_AS_NUMPY, 0.20)), "paper grid must include a=0.20"
    assert np.any(np.isclose(PAPER_A_VALUES_AS_NUMPY, 1.00)), "paper grid must include a=1.00"
if INCLUDE_A0_DIAGNOSTIC:
    assert A_VALUES_AS_NUMPY[0] == 0.0, "a=0 diagnostic point must be first"

KAPPAS_AS_PYTHON_LIST = [2.0, 0.2]

# The adaptive schemes use hmax = 2^-i for i=4,...,9.
LEVELS_AS_PYTHON_LIST = [4, 5, 6, 7, 8, 9]
LEVELS = jnp.array(LEVELS_AS_PYTHON_LIST, dtype=jnp.int32)
NUMBER_OF_LEVELS = len(LEVELS_AS_PYTHON_LIST)
HMAX_BY_LEVEL = 2.0 ** (-LEVELS.astype(jnp.float64))

# The paper takes rho = 2^6, so hmin = hmax / rho.
RHO = 2**6

# Full run uses chunks of 2^14 fine steps.  For smoke tests with much smaller
# REFERENCE_POWER, set FIG3_FULL_DIAG_CHUNK_STEPS manually so each chunk remains short.
FINE_STEPS_PER_CHUNK = int(os.environ.get("FIG3_FULL_DIAG_CHUNK_STEPS", str(2**14)))
FINE_STEPS_PER_CHUNK = min(FINE_STEPS_PER_CHUNK, NUMBER_OF_FINE_STEPS)
NUMBER_OF_CHUNKS = NUMBER_OF_FINE_STEPS // FINE_STEPS_PER_CHUNK
assert NUMBER_OF_FINE_STEPS % FINE_STEPS_PER_CHUNK == 0

# Process this many a values at once.  The default batches all 40 values for
# each kappa panel.  If Kaggle runs out of memory, try FIG3_FULL_DIAG_A_BATCH_SIZE=10.
A_BATCH_SIZE = int(os.environ.get("FIG3_FULL_DIAG_A_BATCH_SIZE", str(NUMBER_OF_A_VALUES)))

# Optional diagnostic.  The paper Fig. 3 does not plot EF_y.
PLOT_EF_Y_DIAGNOSTIC = os.environ.get("FIG3_FULL_DIAG_PLOT_EF_Y", "0") == "1"

FIXED_STEP_SOURCE = os.environ.get("FIG3_FULL_DIAG_FIXED_STEP_SOURCE", "hmax").strip().lower()
FIT_X_SOURCE = os.environ.get("FIG3_FULL_DIAG_FIT_X_SOURCE", "hmax").strip().lower()
VALID_STEP_SOURCES = {"hmean", "hmax"}
if FIXED_STEP_SOURCE not in VALID_STEP_SOURCES:
    raise ValueError("FIG3_FULL_DIAG_FIXED_STEP_SOURCE must be either 'hmean' or 'hmax'")
if FIT_X_SOURCE not in VALID_STEP_SOURCES:
    raise ValueError("FIG3_FULL_DIAG_FIT_X_SOURCE must be either 'hmean' or 'hmax'")

SEED = int(os.environ.get("FIG3_FULL_DIAG_SEED", "2022"))
PATH_KEY = jax.random.PRNGKey(SEED)
BRIDGE_KEY_EA = jax.random.PRNGKey(SEED + 100_000)
BRIDGE_KEY_SIA = jax.random.PRNGKey(SEED + 200_000)
BRIDGE_KEY_FIXED = jax.random.PRNGKey(SEED + 300_000)

if os.path.isdir("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working"
else:
    import pathlib
    OUTPUT_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_STEM = f"fig3_full_diagnostic_jax_h{REFERENCE_POWER}"
if FIXED_STEP_SOURCE != "hmean" or FIT_X_SOURCE != "hmean":
    OUTPUT_STEM += f"_fixed-{FIXED_STEP_SOURCE}_fit-{FIT_X_SOURCE}"


# =============================================================================
# 2. Coefficients and small mathematical building blocks
# =============================================================================

def make_coefficients(kappa, a_values):
    """Return sigma, alpha, beta, gamma for a batch of a values.

    We use

        a = sigma^2 / (2 kappa lambda),

    hence

        sigma = sqrt(2 kappa lambda a).

    The Lamperti coefficients are

        alpha = (4 kappa lambda - sigma^2) / 8,
        beta  = -kappa / 2,
        gamma = sigma / 2.
    """
    a = jnp.asarray(a_values, dtype=jnp.float64)
    kappa_jnp = jnp.asarray(kappa, dtype=jnp.float64)
    sigma = jnp.sqrt(2.0 * kappa_jnp * LAMBDA * a)
    alpha = (4.0 * kappa_jnp * LAMBDA - sigma * sigma) / 8.0
    beta = -kappa_jnp / 2.0
    gamma = sigma / 2.0
    return a, sigma, alpha, beta, gamma


def implicit_lamperti_step(y_old, brownian_increment, step_size, alpha, beta, gamma):
    """Positive-root drift-implicit Lamperti step.

    It solves

        y_new = y_old + h(alpha/y_new + beta y_new) + gamma Delta W.

    If u = y_old + gamma Delta W and d = 1 - beta h, then

        y_new = u/(2d) + sqrt(u^2/(4d^2) + alpha h/d).
    """
    u = y_old + gamma * brownian_increment
    denominator = 1.0 - beta * step_size
    return (
        u / (2.0 * denominator)
        + jnp.sqrt(u * u / (4.0 * denominator * denominator) + alpha * step_size / denominator)
    )


def choose_adaptive_step_size(y_values, current_times):
    """Adaptive step rule used by EA and SIA.

        h_prop = hmax min(1, |Y_n|),
        h_min  = hmax / rho,
        h_n    = max(h_min, h_prop),
        h_n    = min(h_n, T - t_n).
    """
    proposed = HMAX_BY_LEVEL[None, :, None] * jnp.minimum(1.0, jnp.abs(y_values))
    minimum = (HMAX_BY_LEVEL / RHO)[None, :, None]
    used_minimum_step = proposed <= minimum
    chosen = jnp.where(used_minimum_step, minimum, proposed)
    remaining_time = FINAL_TIME - current_times
    return jnp.minimum(chosen, remaining_time), used_minimum_step


def make_brownian_increment_chunk(chunk_number, path_key):
    """Generate one deterministic chunk of Brownian increments.

        Delta W_i = sqrt(h_ref) Z_i,    Z_i ~ Normal(0,1).
    """
    chunk_key = jax.random.fold_in(path_key, chunk_number)
    shape = (FINE_STEPS_PER_CHUNK, NUMBER_OF_PATHS)
    return jax.random.normal(chunk_key, shape, dtype=jnp.float64) * jnp.sqrt(REFERENCE_STEP)


def sample_brownian_bridge_value(w_left, w_right, t_left, t_right, target_time, key):
    """Sample W(target_time) inside one fine reference interval.

    For target_time tau in [t_left, t_right], the Brownian bridge law is

        mean = W_left + theta(W_right - W_left),
        var  = (tau - t_left)(t_right - tau) / h_ref,

    where theta = (tau - t_left) / h_ref.

    This is the same fine-interval bridge coupling used in the Fig. 2(a)
    reconstruction.  The target_time array has shape (a, levels, paths).
    """
    tau = jnp.clip(target_time, t_left, t_right)
    theta = (tau - t_left) / REFERENCE_STEP
    conditional_mean = w_left[None, None, :] + theta * (w_right - w_left)[None, None, :]
    conditional_variance = (tau - t_left) * (t_right - tau) / REFERENCE_STEP
    standard_normals = jax.random.normal(key, target_time.shape, dtype=jnp.float64)
    return conditional_mean + jnp.sqrt(jnp.maximum(conditional_variance, 0.0)) * standard_normals


def root_mean_square_error(scheme_x_values, reference_x_values):
    """RMSE for each a value and each level.

        RMSE(a, level)
        = sqrt(mean_m |X_scheme(a, level, m) - X_ref(a, m)|^2).
    """
    path_errors = scheme_x_values - reference_x_values[:, None, :]
    return jnp.sqrt(jnp.mean(path_errors * path_errors, axis=2))


def fit_orders(h_mean, errors):
    """Fit log(error) = c + p log(h_mean), returning p for each a value."""
    log_h = jnp.log(h_mean)
    log_e = jnp.log(errors)
    centered_h = log_h - jnp.mean(log_h, axis=1, keepdims=True)
    centered_e = log_e - jnp.mean(log_e, axis=1, keepdims=True)
    numerator = jnp.sum(centered_h * centered_e, axis=1)
    denominator = jnp.sum(centered_h * centered_h, axis=1)
    return numerator / denominator


def hmax_matrix_like(h_mean):
    """Return the level-wise hmax values with the same shape as h_mean."""
    return jnp.broadcast_to(HMAX_BY_LEVEL[None, :], h_mean.shape)


def choose_step_matrix(source, h_mean):
    """Choose either EA-derived h_mean or level-wise hmax for diagnostics."""
    if source == "hmean":
        return h_mean
    if source == "hmax":
        return hmax_matrix_like(h_mean)
    raise ValueError(f"unknown step source: {source}")


# =============================================================================
# 3. Pass 1: reference IF plus adaptive EA/SIA
# =============================================================================

def create_pass1_state(a_count, ea_key, sia_key):
    """Create the first-pass state for one batch of a values."""
    reference_shape = (a_count, NUMBER_OF_PATHS)
    scheme_shape = (a_count, NUMBER_OF_LEVELS, NUMBER_OF_PATHS)

    initial_scheme_y = jnp.full(scheme_shape, INITIAL_Y, dtype=jnp.float64)
    initial_step, initial_used_minimum = choose_adaptive_step_size(initial_scheme_y, 0.0)

    return {
        "reference_y": jnp.full(reference_shape, INITIAL_Y, dtype=jnp.float64),
        "brownian_w": jnp.zeros((NUMBER_OF_PATHS,), dtype=jnp.float64),

        "ea_y": initial_scheme_y,
        "ea_last_w": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "ea_last_t": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "ea_target_t": initial_step,
        "ea_used_minimum_step": initial_used_minimum,
        "ea_number_of_steps": jnp.zeros(scheme_shape, dtype=jnp.int32),
        "ea_sum_of_steps": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "ea_bridge_key": ea_key,

        "sia_y": initial_scheme_y,
        "sia_last_w": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "sia_last_t": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "sia_target_t": initial_step,
        "sia_used_minimum_step": initial_used_minimum,
        "sia_number_of_steps": jnp.zeros(scheme_shape, dtype=jnp.int32),
        "sia_sum_of_steps": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "sia_bridge_key": sia_key,
    }


def update_adaptive_scheme_when_it_lands(
    *,
    y_old,
    last_w,
    last_t,
    target_t,
    used_minimum_step,
    number_of_steps,
    sum_of_steps,
    w_left,
    w_right,
    t_left,
    t_right,
    bridge_key,
    alpha,
    beta,
    gamma,
    use_sia_formula,
):
    """Update EA or SIA if its next target time is in this fine interval.

    This is the faithful Fig. 2-style landing rule:

        1. detect target_t <= t_right;
        2. sample W(target_t) from the Brownian bridge on [t_left, t_right];
        3. use Delta W = W(target_t) - W(last_t);
        4. apply EA or SIA, with the implicit backstop if needed.
    """
    target_is_inside_this_fine_interval = target_t <= t_right
    alpha3 = alpha[:, None, None]
    gamma3 = gamma[:, None, None]

    def land_at_target(_):
        next_key, normal_key = jax.random.split(bridge_key)

        target_w_raw = sample_brownian_bridge_value(
            w_left, w_right, t_left, t_right, target_t, normal_key
        )
        target_w = jnp.where(target_is_inside_this_fine_interval, target_w_raw, last_w)

        step_size = target_t - last_t
        scheme_dW = target_w - last_w

        if use_sia_formula:
            # SIA:
            #   Y_new = (Y + h alpha/Y + gamma dW) / (1 - beta h).
            explicit_candidate = (
                y_old + step_size * alpha3 / y_old + gamma3 * scheme_dW
            ) / (1.0 - beta * step_size)
        else:
            # EA:
            #   Y_new = Y + h(alpha/Y + beta Y) + gamma dW.
            explicit_candidate = (
                y_old
                + step_size * (alpha3 / y_old + beta * y_old)
                + gamma3 * scheme_dW
            )

        backstop_value = implicit_lamperti_step(y_old, scheme_dW, step_size, alpha3, beta, gamma3)
        should_use_backstop = used_minimum_step | (explicit_candidate <= 0.0)
        y_after_step = jnp.where(should_use_backstop, backstop_value, explicit_candidate)

        y_new = jnp.where(target_is_inside_this_fine_interval, y_after_step, y_old)
        last_w_new = jnp.where(target_is_inside_this_fine_interval, target_w, last_w)
        last_t_new = jnp.where(target_is_inside_this_fine_interval, target_t, last_t)

        number_of_steps_new = number_of_steps + target_is_inside_this_fine_interval.astype(jnp.int32)
        sum_of_steps_new = sum_of_steps + jnp.where(
            target_is_inside_this_fine_interval, step_size, 0.0
        )

        next_step_size, next_used_minimum_step = choose_adaptive_step_size(y_new, last_t_new)
        target_t_new = jnp.where(
            target_is_inside_this_fine_interval,
            last_t_new + next_step_size,
            target_t,
        )
        used_minimum_step_new = jnp.where(
            target_is_inside_this_fine_interval,
            next_used_minimum_step,
            used_minimum_step,
        )

        return (
            y_new,
            last_w_new,
            last_t_new,
            target_t_new,
            used_minimum_step_new,
            number_of_steps_new,
            sum_of_steps_new,
            next_key,
        )

    def do_not_land(_):
        return (
            y_old,
            last_w,
            last_t,
            target_t,
            used_minimum_step,
            number_of_steps,
            sum_of_steps,
            bridge_key,
        )

    return lax.cond(
        jnp.any(target_is_inside_this_fine_interval),
        land_at_target,
        do_not_land,
        operand=None,
    )


@jax.jit
def run_pass1_chunk(
    state,
    brownian_increment_chunk,
    first_fine_index_in_chunk,
    alpha,
    beta,
    gamma,
):
    """Run one chunk by scanning its fine reference intervals."""
    alpha_ref = alpha[:, None]
    gamma_ref = gamma[:, None]

    fine_indices_after_step = first_fine_index_in_chunk + jnp.arange(
        1, brownian_increment_chunk.shape[0] + 1, dtype=jnp.int32
    )

    def pass1_one_fine_step(state_c, scan_input):
        fine_brownian_increment, fine_index_after_step = scan_input
        t_right = fine_index_after_step.astype(jnp.float64) * REFERENCE_STEP
        t_left = t_right - REFERENCE_STEP

        w_left = state_c["brownian_w"]
        w_right = w_left + fine_brownian_increment

        reference_y_new = implicit_lamperti_step(
            state_c["reference_y"],
            fine_brownian_increment[None, :],
            REFERENCE_STEP,
            alpha_ref,
            beta,
            gamma_ref,
        )

        (
            ea_y_new,
            ea_last_w_new,
            ea_last_t_new,
            ea_target_t_new,
            ea_used_minimum_step_new,
            ea_number_of_steps_new,
            ea_sum_of_steps_new,
            ea_bridge_key_new,
        ) = update_adaptive_scheme_when_it_lands(
            y_old=state_c["ea_y"],
            last_w=state_c["ea_last_w"],
            last_t=state_c["ea_last_t"],
            target_t=state_c["ea_target_t"],
            used_minimum_step=state_c["ea_used_minimum_step"],
            number_of_steps=state_c["ea_number_of_steps"],
            sum_of_steps=state_c["ea_sum_of_steps"],
            w_left=w_left,
            w_right=w_right,
            t_left=t_left,
            t_right=t_right,
            bridge_key=state_c["ea_bridge_key"],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            use_sia_formula=False,
        )

        (
            sia_y_new,
            sia_last_w_new,
            sia_last_t_new,
            sia_target_t_new,
            sia_used_minimum_step_new,
            sia_number_of_steps_new,
            sia_sum_of_steps_new,
            sia_bridge_key_new,
        ) = update_adaptive_scheme_when_it_lands(
            y_old=state_c["sia_y"],
            last_w=state_c["sia_last_w"],
            last_t=state_c["sia_last_t"],
            target_t=state_c["sia_target_t"],
            used_minimum_step=state_c["sia_used_minimum_step"],
            number_of_steps=state_c["sia_number_of_steps"],
            sum_of_steps=state_c["sia_sum_of_steps"],
            w_left=w_left,
            w_right=w_right,
            t_left=t_left,
            t_right=t_right,
            bridge_key=state_c["sia_bridge_key"],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            use_sia_formula=True,
        )

        new_state = {
            "reference_y": reference_y_new,
            "brownian_w": w_right,
            "ea_y": ea_y_new,
            "ea_last_w": ea_last_w_new,
            "ea_last_t": ea_last_t_new,
            "ea_target_t": ea_target_t_new,
            "ea_used_minimum_step": ea_used_minimum_step_new,
            "ea_number_of_steps": ea_number_of_steps_new,
            "ea_sum_of_steps": ea_sum_of_steps_new,
            "ea_bridge_key": ea_bridge_key_new,
            "sia_y": sia_y_new,
            "sia_last_w": sia_last_w_new,
            "sia_last_t": sia_last_t_new,
            "sia_target_t": sia_target_t_new,
            "sia_used_minimum_step": sia_used_minimum_step_new,
            "sia_number_of_steps": sia_number_of_steps_new,
            "sia_sum_of_steps": sia_sum_of_steps_new,
            "sia_bridge_key": sia_bridge_key_new,
        }
        return new_state, None

    new_state, _ = lax.scan(
        pass1_one_fine_step,
        state,
        (brownian_increment_chunk, fine_indices_after_step),
    )
    return new_state


def run_pass1_for_batch(a_count, panel_index, batch_index, path_key, alpha, beta, gamma):
    """Run pass 1 for a batch of a values."""
    ea_key = jax.random.fold_in(BRIDGE_KEY_EA, panel_index * 10_000 + batch_index)
    sia_key = jax.random.fold_in(BRIDGE_KEY_SIA, panel_index * 10_000 + batch_index)
    state = create_pass1_state(a_count, ea_key, sia_key)
    start = time.perf_counter()

    for chunk_number in range(NUMBER_OF_CHUNKS):
        chunk = make_brownian_increment_chunk(chunk_number, path_key)
        first_index = jnp.array(chunk_number * FINE_STEPS_PER_CHUNK, dtype=jnp.int32)
        state = run_pass1_chunk(state, chunk, first_index, alpha, beta, gamma)

        if (chunk_number + 1) % 128 == 0 or (chunk_number + 1) == NUMBER_OF_CHUNKS:
            state["reference_y"].block_until_ready()
            elapsed = time.perf_counter() - start
            print(
                f"  pass 1 batch {batch_index}: "
                f"{chunk_number + 1}/{NUMBER_OF_CHUNKS} chunks, {elapsed:.1f}s"
            )

    return state


# =============================================================================
# 4. Pass 2: fixed EF, IF, FT
# =============================================================================

def create_pass2_state(a_count, fixed_step_by_a_and_level, fixed_key):
    """Create pass-2 state for fixed EF, IF, FT, and optional EF_y."""
    scheme_shape = (a_count, NUMBER_OF_LEVELS, NUMBER_OF_PATHS)
    first_target_time = jnp.broadcast_to(
        jnp.minimum(fixed_step_by_a_and_level[:, :, None], FINAL_TIME),
        scheme_shape,
    )

    return {
        "brownian_w": jnp.zeros((NUMBER_OF_PATHS,), dtype=jnp.float64),
        "ef_x_value": jnp.full(scheme_shape, INITIAL_X, dtype=jnp.float64),
        "ef_y_value": jnp.full(scheme_shape, INITIAL_Y, dtype=jnp.float64),
        "ft_raw_x": jnp.full(scheme_shape, INITIAL_X, dtype=jnp.float64),
        "fixed_if_y": jnp.full(scheme_shape, INITIAL_Y, dtype=jnp.float64),
        "fixed_landing_count": jnp.zeros((a_count, NUMBER_OF_LEVELS), dtype=jnp.int64),
        "ef_negative_before_count": jnp.zeros((a_count, NUMBER_OF_LEVELS), dtype=jnp.int64),
        "ef_negative_after_count": jnp.zeros((a_count, NUMBER_OF_LEVELS), dtype=jnp.int64),
        "ft_negative_before_count": jnp.zeros((a_count, NUMBER_OF_LEVELS), dtype=jnp.int64),
        "ft_negative_after_count": jnp.zeros((a_count, NUMBER_OF_LEVELS), dtype=jnp.int64),
        "last_w": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "last_t": jnp.zeros(scheme_shape, dtype=jnp.float64),
        "target_t": first_target_time,
        "fixed_step": fixed_step_by_a_and_level.astype(jnp.float64),
        "bridge_key": fixed_key,
    }


def update_fixed_schemes_when_they_land(
    *,
    state,
    w_left,
    w_right,
    t_left,
    t_right,
    kappa,
    sigma,
    alpha,
    beta,
    gamma,
):
    """Update fixed schemes if their next target time is in this fine interval."""
    target_is_inside_this_fine_interval = state["target_t"] <= t_right

    sigma3 = sigma[:, None, None]
    alpha3 = alpha[:, None, None]
    gamma3 = gamma[:, None, None]
    kappa_jnp = jnp.asarray(kappa, dtype=jnp.float64)

    def land_at_target(_):
        next_key, normal_key = jax.random.split(state["bridge_key"])

        target_w_raw = sample_brownian_bridge_value(
            w_left, w_right, t_left, t_right, state["target_t"], normal_key
        )
        target_w = jnp.where(target_is_inside_this_fine_interval, target_w_raw, state["last_w"])

        step_size = state["target_t"] - state["last_t"]
        scheme_dW = target_w - state["last_w"]

        # Paper EF: explicit Higham-Mao discretisation of X.
        ef_x_candidate = (
            state["ef_x_value"]
            + step_size * kappa_jnp * (LAMBDA - state["ef_x_value"])
            + sigma3 * jnp.sqrt(jnp.abs(state["ef_x_value"])) * scheme_dW
        )
        ef_x_new = jnp.where(
            target_is_inside_this_fine_interval,
            ef_x_candidate,
            state["ef_x_value"],
        )

        # Optional diagnostic EF_y: explicit Euler in the Lamperti variable.
        ef_y_candidate = (
            state["ef_y_value"]
            + step_size * (alpha3 / state["ef_y_value"] + beta * state["ef_y_value"])
            + gamma3 * scheme_dW
        )
        ef_y_new = jnp.where(
            target_is_inside_this_fine_interval,
            ef_y_candidate,
            state["ef_y_value"],
        )

        # FT: full truncation in X.
        positive_part = jnp.maximum(state["ft_raw_x"], 0.0)
        ft_candidate = (
            state["ft_raw_x"]
            + step_size * kappa_jnp * (LAMBDA - positive_part)
            + sigma3 * jnp.sqrt(positive_part) * scheme_dW
        )
        ft_new = jnp.where(
            target_is_inside_this_fine_interval,
            ft_candidate,
            state["ft_raw_x"],
        )

        landing_count_add = jnp.sum(
            target_is_inside_this_fine_interval.astype(jnp.int64),
            axis=2,
        )
        ef_negative_before_add = jnp.sum(
            (target_is_inside_this_fine_interval & (state["ef_x_value"] <= 0.0)).astype(jnp.int64),
            axis=2,
        )
        ef_negative_after_add = jnp.sum(
            (target_is_inside_this_fine_interval & (ef_x_candidate <= 0.0)).astype(jnp.int64),
            axis=2,
        )
        ft_negative_before_add = jnp.sum(
            (target_is_inside_this_fine_interval & (state["ft_raw_x"] <= 0.0)).astype(jnp.int64),
            axis=2,
        )
        ft_negative_after_add = jnp.sum(
            (target_is_inside_this_fine_interval & (ft_candidate <= 0.0)).astype(jnp.int64),
            axis=2,
        )

        # Fixed IF: implicit Lamperti on the selected fixed grid.
        fixed_if_candidate = implicit_lamperti_step(
            state["fixed_if_y"],
            scheme_dW,
            step_size,
            alpha3,
            beta,
            gamma3,
        )
        fixed_if_new = jnp.where(
            target_is_inside_this_fine_interval,
            fixed_if_candidate,
            state["fixed_if_y"],
        )

        last_w_new = jnp.where(target_is_inside_this_fine_interval, target_w, state["last_w"])
        last_t_new = jnp.where(
            target_is_inside_this_fine_interval,
            state["target_t"],
            state["last_t"],
        )

        next_target = jnp.minimum(last_t_new + state["fixed_step"][:, :, None], FINAL_TIME)
        target_t_new = jnp.where(
            target_is_inside_this_fine_interval,
            next_target,
            state["target_t"],
        )

        return {
            "brownian_w": w_right,
            "ef_x_value": ef_x_new,
            "ef_y_value": ef_y_new,
            "ft_raw_x": ft_new,
            "fixed_if_y": fixed_if_new,
            "fixed_landing_count": state["fixed_landing_count"] + landing_count_add,
            "ef_negative_before_count": state["ef_negative_before_count"] + ef_negative_before_add,
            "ef_negative_after_count": state["ef_negative_after_count"] + ef_negative_after_add,
            "ft_negative_before_count": state["ft_negative_before_count"] + ft_negative_before_add,
            "ft_negative_after_count": state["ft_negative_after_count"] + ft_negative_after_add,
            "last_w": last_w_new,
            "last_t": last_t_new,
            "target_t": target_t_new,
            "fixed_step": state["fixed_step"],
            "bridge_key": next_key,
        }

    def do_not_land(_):
        return {
            "brownian_w": w_right,
            "ef_x_value": state["ef_x_value"],
            "ef_y_value": state["ef_y_value"],
            "ft_raw_x": state["ft_raw_x"],
            "fixed_if_y": state["fixed_if_y"],
            "fixed_landing_count": state["fixed_landing_count"],
            "ef_negative_before_count": state["ef_negative_before_count"],
            "ef_negative_after_count": state["ef_negative_after_count"],
            "ft_negative_before_count": state["ft_negative_before_count"],
            "ft_negative_after_count": state["ft_negative_after_count"],
            "last_w": state["last_w"],
            "last_t": state["last_t"],
            "target_t": state["target_t"],
            "fixed_step": state["fixed_step"],
            "bridge_key": state["bridge_key"],
        }

    return lax.cond(
        jnp.any(target_is_inside_this_fine_interval),
        land_at_target,
        do_not_land,
        operand=None,
    )


@jax.jit
def run_pass2_chunk(
    state,
    brownian_increment_chunk,
    first_fine_index_in_chunk,
    kappa,
    sigma,
    alpha,
    beta,
    gamma,
):
    """Run one chunk by scanning its fine reference intervals."""
    fine_indices_after_step = first_fine_index_in_chunk + jnp.arange(
        1, brownian_increment_chunk.shape[0] + 1, dtype=jnp.int32
    )

    def pass2_one_fine_step(state_c, scan_input):
        fine_brownian_increment, fine_index_after_step = scan_input
        t_right = fine_index_after_step.astype(jnp.float64) * REFERENCE_STEP
        t_left = t_right - REFERENCE_STEP

        w_left = state_c["brownian_w"]
        w_right = w_left + fine_brownian_increment

        new_state = update_fixed_schemes_when_they_land(
            state=state_c,
            w_left=w_left,
            w_right=w_right,
            t_left=t_left,
            t_right=t_right,
            kappa=kappa,
            sigma=sigma,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        return new_state, None

    new_state, _ = lax.scan(
        pass2_one_fine_step,
        state,
        (brownian_increment_chunk, fine_indices_after_step),
    )
    return new_state


def run_pass2_for_batch(
    a_count,
    panel_index,
    batch_index,
    path_key,
    fixed_step,
    kappa,
    sigma,
    alpha,
    beta,
    gamma,
):
    """Run pass 2 for a batch of a values."""
    fixed_key = jax.random.fold_in(BRIDGE_KEY_FIXED, panel_index * 10_000 + batch_index)
    state = create_pass2_state(a_count, fixed_step, fixed_key)
    start = time.perf_counter()

    kappa_jnp = jnp.asarray(kappa, dtype=jnp.float64)
    for chunk_number in range(NUMBER_OF_CHUNKS):
        chunk = make_brownian_increment_chunk(chunk_number, path_key)
        first_index = jnp.array(chunk_number * FINE_STEPS_PER_CHUNK, dtype=jnp.int32)
        state = run_pass2_chunk(
            state,
            chunk,
            first_index,
            kappa_jnp,
            sigma,
            alpha,
            beta,
            gamma,
        )

        if (chunk_number + 1) % 128 == 0 or (chunk_number + 1) == NUMBER_OF_CHUNKS:
            state["ef_x_value"].block_until_ready()
            elapsed = time.perf_counter() - start
            print(
                f"  pass 2 batch {batch_index}: "
                f"{chunk_number + 1}/{NUMBER_OF_CHUNKS} chunks, {elapsed:.1f}s"
            )

    return state


# =============================================================================
# 5. One kappa panel and plotting
# =============================================================================

def run_one_kappa_panel(kappa, panel_index):
    """Run all a values for one Fig. 3 panel."""
    print(f"\n=== Panel {panel_index + 1}: kappa={kappa} ===")
    panel_path_key = jax.random.fold_in(PATH_KEY, panel_index)

    panel_results = []
    for batch_start in range(0, NUMBER_OF_A_VALUES, A_BATCH_SIZE):
        batch_stop = min(batch_start + A_BATCH_SIZE, NUMBER_OF_A_VALUES)
        a_batch_np = A_VALUES_AS_NUMPY[batch_start:batch_stop]
        batch_index = batch_start // A_BATCH_SIZE
        a_count = len(a_batch_np)

        print(
            f"batch {batch_index}: a[{batch_start}:{batch_stop}] "
            f"from {a_batch_np[0]:.4f} to {a_batch_np[-1]:.4f}"
        )

        a, sigma, alpha, beta, gamma = make_coefficients(kappa, a_batch_np)

        pass1_state = run_pass1_for_batch(
            a_count,
            panel_index,
            batch_index,
            panel_path_key,
            alpha,
            beta,
            gamma,
        )

        reference_x = pass1_state["reference_y"] ** 2
        ea_x = pass1_state["ea_y"] ** 2
        sia_x = pass1_state["sia_y"] ** 2

        h_mean = jnp.mean(
            pass1_state["ea_sum_of_steps"] / pass1_state["ea_number_of_steps"],
            axis=2,
        )
        h_mean.block_until_ready()
        fixed_step_for_pass2 = choose_step_matrix(FIXED_STEP_SOURCE, h_mean)
        fit_x = choose_step_matrix(FIT_X_SOURCE, h_mean)

        pass2_state = run_pass2_for_batch(
            a_count,
            panel_index,
            batch_index,
            panel_path_key,
            fixed_step_for_pass2,
            kappa,
            sigma,
            alpha,
            beta,
            gamma,
        )

        ef_x = pass2_state["ef_x_value"]
        ef_y_x = pass2_state["ef_y_value"] ** 2
        fixed_if_x = pass2_state["fixed_if_y"] ** 2
        ft_x = jnp.maximum(pass2_state["ft_raw_x"], 0.0)

        errors = {
            "EA": root_mean_square_error(ea_x, reference_x),
            "EF": root_mean_square_error(ef_x, reference_x),
            "IF": root_mean_square_error(fixed_if_x, reference_x),
            "SIA": root_mean_square_error(sia_x, reference_x),
            "FT": root_mean_square_error(ft_x, reference_x),
        }
        if PLOT_EF_Y_DIAGNOSTIC:
            errors["EF_y"] = root_mean_square_error(ef_y_x, reference_x)

        orders = {name: fit_orders(fit_x, value) for name, value in errors.items()}
        for value in orders.values():
            value.block_until_ready()

        landing_count = pass2_state["fixed_landing_count"]
        denominator = jnp.maximum(landing_count, 1)
        fixed_diagnostics = {
            "fixed_landing_count": landing_count,
            "ef_negative_before_count": pass2_state["ef_negative_before_count"],
            "ef_negative_after_count": pass2_state["ef_negative_after_count"],
            "ft_negative_before_count": pass2_state["ft_negative_before_count"],
            "ft_negative_after_count": pass2_state["ft_negative_after_count"],
            "ef_negative_before_proportion": pass2_state["ef_negative_before_count"] / denominator,
            "ef_negative_after_proportion": pass2_state["ef_negative_after_count"] / denominator,
            "ft_negative_before_proportion": pass2_state["ft_negative_before_count"] / denominator,
            "ft_negative_after_proportion": pass2_state["ft_negative_after_count"] / denominator,
        }

        hmax_by_a_and_level = hmax_matrix_like(h_mean)

        batch_a = np.asarray(a, dtype=np.float64)
        is_a0_diagnostic = batch_a == 0.0

        batch_result = {
            "kappa": np.full(a_count, kappa, dtype=np.float64),
            "a": batch_a,
            "is_a0_diagnostic": is_a0_diagnostic,
            "sigma": np.asarray(sigma, dtype=np.float64),
            "h_mean": np.asarray(h_mean, dtype=np.float64),
            "hmax": np.asarray(hmax_by_a_and_level, dtype=np.float64),
            "h_mean_over_hmax": np.asarray(h_mean / hmax_by_a_and_level, dtype=np.float64),
            "fixed_step": np.asarray(fixed_step_for_pass2, dtype=np.float64),
            "fit_x": np.asarray(fit_x, dtype=np.float64),
            "errors": {name: np.asarray(value, dtype=np.float64) for name, value in errors.items()},
            "fixed_diagnostics": {
                name: np.asarray(value)
                for name, value in fixed_diagnostics.items()
            },
            "orders": {name: np.asarray(value, dtype=np.float64) for name, value in orders.items()},
        }
        panel_results.append(batch_result)

        print("  fitted orders at batch endpoints:")
        for local_index in (0, a_count - 1):
            pieces = [
                f"{name}={batch_result['orders'][name][local_index]:.3f}"
                for name in ("EA", "EF", "IF", "SIA", "FT")
            ]
            print(f"    a={batch_result['a'][local_index]:.4f}: " + ", ".join(pieces))

    return concatenate_panel_results(panel_results)


def concatenate_panel_results(panel_results):
    """Concatenate batch dictionaries for one kappa panel."""
    scheme_names = list(panel_results[0]["orders"].keys())
    diagnostic_names = list(panel_results[0]["fixed_diagnostics"].keys())
    return {
        "kappa": np.concatenate([item["kappa"] for item in panel_results]),
        "a": np.concatenate([item["a"] for item in panel_results]),
        "is_a0_diagnostic": np.concatenate(
            [item["is_a0_diagnostic"] for item in panel_results]
        ),
        "sigma": np.concatenate([item["sigma"] for item in panel_results]),
        "h_mean": np.concatenate([item["h_mean"] for item in panel_results], axis=0),
        "hmax": np.concatenate([item["hmax"] for item in panel_results], axis=0),
        "h_mean_over_hmax": np.concatenate(
            [item["h_mean_over_hmax"] for item in panel_results],
            axis=0,
        ),
        "fixed_step": np.concatenate([item["fixed_step"] for item in panel_results], axis=0),
        "fit_x": np.concatenate([item["fit_x"] for item in panel_results], axis=0),
        "errors": {
            name: np.concatenate([item["errors"][name] for item in panel_results], axis=0)
            for name in scheme_names
        },
        "fixed_diagnostics": {
            name: np.concatenate(
                [item["fixed_diagnostics"][name] for item in panel_results],
                axis=0,
            )
            for name in diagnostic_names
        },
        "orders": {
            name: np.concatenate([item["orders"][name] for item in panel_results])
            for name in scheme_names
        },
    }


def save_results_csv(all_panel_results):
    """Save a simple CSV table of fitted orders."""
    scheme_names = list(all_panel_results[0]["orders"].keys())
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_STEM + ".csv")

    header = ["kappa", "a", "grid_role", "sigma"] + scheme_names
    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        for panel in all_panel_results:
            for index in range(len(panel["a"])):
                grid_role = "a0_diagnostic" if panel["is_a0_diagnostic"][index] else "paper_grid"
                row = [
                    panel["kappa"][index],
                    panel["a"][index],
                    grid_role,
                    panel["sigma"][index],
                ]
                row.extend(panel["orders"][name][index] for name in scheme_names)
                writer.writerow(row)
    print("saved", csv_path)


def save_diagnostics_csv(all_panel_results):
    """Save one row per a value and level with RMSE and negativity diagnostics."""
    scheme_names = list(all_panel_results[0]["errors"].keys())
    diagnostic_names = list(all_panel_results[0]["fixed_diagnostics"].keys())
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_STEM + "_diagnostics.csv")

    header = [
        "kappa",
        "a",
        "grid_role",
        "sigma",
        "level",
        "hmax",
        "h_mean",
        "h_mean_over_hmax",
        "fixed_step",
        "fit_x",
    ]
    header.extend(f"rmse_{name}" for name in scheme_names)
    header.extend(diagnostic_names)

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        for panel in all_panel_results:
            for a_index in range(len(panel["a"])):
                grid_role = "a0_diagnostic" if panel["is_a0_diagnostic"][a_index] else "paper_grid"
                for level_index, level in enumerate(LEVELS_AS_PYTHON_LIST):
                    row = [
                        panel["kappa"][a_index],
                        panel["a"][a_index],
                        grid_role,
                        panel["sigma"][a_index],
                        level,
                        panel["hmax"][a_index, level_index],
                        panel["h_mean"][a_index, level_index],
                        panel["h_mean_over_hmax"][a_index, level_index],
                        panel["fixed_step"][a_index, level_index],
                        panel["fit_x"][a_index, level_index],
                    ]
                    row.extend(
                        panel["errors"][name][a_index, level_index]
                        for name in scheme_names
                    )
                    row.extend(
                        panel["fixed_diagnostics"][name][a_index, level_index]
                        for name in diagnostic_names
                    )
                    writer.writerow(row)

    print("saved", csv_path)


def save_results_npz(all_panel_results):
    """Save a richer NPZ file containing h_mean and all order arrays."""
    npz_path = os.path.join(OUTPUT_DIR, OUTPUT_STEM + ".npz")
    arrays = {}
    for panel_index, panel in enumerate(all_panel_results):
        prefix = f"panel{panel_index + 1}"
        arrays[f"{prefix}_kappa"] = panel["kappa"]
        arrays[f"{prefix}_a"] = panel["a"]
        arrays[f"{prefix}_is_a0_diagnostic"] = panel["is_a0_diagnostic"]
        arrays[f"{prefix}_sigma"] = panel["sigma"]
        arrays[f"{prefix}_h_mean"] = panel["h_mean"]
        arrays[f"{prefix}_hmax"] = panel["hmax"]
        arrays[f"{prefix}_h_mean_over_hmax"] = panel["h_mean_over_hmax"]
        arrays[f"{prefix}_fixed_step"] = panel["fixed_step"]
        arrays[f"{prefix}_fit_x"] = panel["fit_x"]
        for name, values in panel["errors"].items():
            arrays[f"{prefix}_rmse_{name}"] = values
        for name, values in panel["fixed_diagnostics"].items():
            arrays[f"{prefix}_{name}"] = values
        for name, values in panel["orders"].items():
            arrays[f"{prefix}_order_{name}"] = values
    np.savez(npz_path, **arrays)
    print("saved", npz_path)


def plot_fig3(all_panel_results, *, include_a0, output_suffix, title_label):
    """Plot fitted rates, optionally including the a=0 diagnostic point."""
    style = {
        "EA": dict(color="blue", marker="o", ls="-", mfc="none"),
        "EF": dict(color="black", marker="+", ls="-"),
        "IF": dict(color="magenta", marker="o", ls="-.", mfc="none"),
        "SIA": dict(color="cyan", marker="+", ls="--"),
        "FT": dict(color="red", marker="x", ls=":"),
        "EF_y": dict(color="0.45", marker="s", ls="--", mfc="none"),
    }

    scheme_names = ["EA", "EF", "IF", "SIA", "FT"]
    if PLOT_EF_Y_DIAGNOSTIC:
        scheme_names.append("EF_y")

    if len(all_panel_results) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
        axes = [ax]
        panel_labels = ["(a)"]
    else:
        fig, axes_array = plt.subplots(1, len(all_panel_results), figsize=(9.2, 3.7), sharey=True)
        axes = list(np.atleast_1d(axes_array))
        panel_labels = ["(a)", "(b)"][: len(all_panel_results)]

    plotted_a_values = []
    for panel in all_panel_results:
        mask = np.ones_like(panel["a"], dtype=bool)
        if not include_a0:
            mask = ~panel["is_a0_diagnostic"]
        plotted_a_values.append(panel["a"][mask])
    plotted_a_values = np.concatenate(plotted_a_values)

    x_min = float(np.min(plotted_a_values))
    x_max = float(np.max(plotted_a_values))
    x_padding = 0.03 * (x_max - x_min)

    for ax, panel, panel_label in zip(axes, all_panel_results, panel_labels):
        mask = np.ones_like(panel["a"], dtype=bool)
        if not include_a0:
            mask = ~panel["is_a0_diagnostic"]

        for name in scheme_names:
            ax.plot(
                panel["a"][mask],
                panel["orders"][name][mask],
                label=name,
                **style[name],
            )

        if x_min <= 0.25 <= x_max:
            ax.axvline(0.25, color="0.55", lw=1, ls="--")
        if x_min <= 1.0 <= x_max:
            ax.axvline(1.0, color="0.55", lw=1, ls="--")
        ax.set_xlim(max(0.0, x_min - x_padding), min(1.6, x_max + x_padding))
        ax.set_ylim(0.3, 1.1)
        ax.set_xlabel("a")
        ax.set_title(f"{panel_label}  kappa = {panel['kappa'][0]:g}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Rate")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        rf"{title_label}, "
        rf"$h_{{\mathrm{{ref}}}}=2^{{-{REFERENCE_POWER}}}$"
    )
    fig.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_STEM + output_suffix + ".png")
    fig.savefig(output_path, dpi=220)
    print("saved", output_path)
    plt.show()


# =============================================================================
# 6. Main program
# =============================================================================

def main():
    print("devices:", jax.devices())
    print(
        f"M={NUMBER_OF_PATHS}, "
        f"h_ref=2^-{REFERENCE_POWER}, "
        f"N_REF={NUMBER_OF_FINE_STEPS:,}, "
        f"chunks={NUMBER_OF_CHUNKS}, "
        f"chunk_steps={FINE_STEPS_PER_CHUNK}, "
        f"landing_mode=fine-interval bridge"
    )
    print(
        f"paper-grid a values: {NUMBER_OF_PAPER_A_VALUES} uniformly spaced in "
        f"[{PAPER_A_VALUES_AS_NUMPY[0]}, {PAPER_A_VALUES_AS_NUMPY[-1]}]; "
        f"include a=0 diagnostic: {INCLUDE_A0_DIAGNOSTIC}; "
        f"total simulated a values: {NUMBER_OF_A_VALUES}; "
        f"A_BATCH_SIZE={A_BATCH_SIZE}"
    )
    print(
        f"fixed-step source for EF/IF/FT: {FIXED_STEP_SOURCE}; "
        f"polyfit x-axis source: {FIT_X_SOURCE}"
    )
    print("first simulated a values:", A_VALUES_AS_NUMPY[:5])
    print("output stem:", OUTPUT_STEM)
    print("Paper-style EF is direct-CIR EF_x. Set FIG3_FULL_DIAG_PLOT_EF_Y=1 to add EF_y.")

    all_panel_results = []
    for panel_index, kappa in enumerate(KAPPAS_AS_PYTHON_LIST):
        panel_result = run_one_kappa_panel(kappa, panel_index)
        all_panel_results.append(panel_result)

    save_results_csv(all_panel_results)
    save_diagnostics_csv(all_panel_results)
    save_results_npz(all_panel_results)
    plot_fig3(
        all_panel_results,
        include_a0=False,
        output_suffix="_paper_grid",
        title_label="KLM Fig. 3 full diagnostic paper grid",
    )
    if INCLUDE_A0_DIAGNOSTIC:
        plot_fig3(
            all_panel_results,
            include_a0=True,
            output_suffix="_with_a0",
            title_label="KLM Fig. 3 full diagnostic with a=0",
        )


if __name__ == "__main__":
    main()
