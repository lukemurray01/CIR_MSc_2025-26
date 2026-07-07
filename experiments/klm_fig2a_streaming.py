# Integrated from the Kaggle notebook 'klm-fig-2-a-reconstruction.ipynb'
# (single code cell, verbatim) with only these RNG-neutral changes:
#   * FIG2A_M / FIG2A_REFERENCE_POWER / FIG2A_SEED / FIG2A_CHUNK_STEPS env
#     overrides (defaults identical to the notebook: M=1000, power 25,
#     seed 2022, chunk 2^14),
#   * output paths: /kaggle/working when on Kaggle, else repo figures/ and
#     results/; adds PDF + CSV export next to the original PNG.
# Full-scale defaults therefore reproduce the notebook outputs exactly
# (same PRNG keys, same chunk fold-in, same arithmetic order).
# Verified: reduced-scale run (power 14) is bit-identical to the original
# notebook code; see tests/test_klm_streaming_scripts.py and the commit
# message.
"""
Readable Kaggle/JAX reconstruction of KLM Fig. 2(a).

Goal
----
We wish to reproduce the convergence plot for the CIR model using

    * full fine reference step h_ref = 2^-25,
    * Brownian bridge values for off-grid scheme times,
    * the Lamperti explicit EF_y curve,
    * the direct-CIR Higham-Mao EF_x curve as a diagnostic,
    * streaming chunks so we do not store a giant Brownian matrix.

Why streaming?
--------------
The fine reference has 2^25 = 33,554,432 time steps. With M=1000 paths,
storing every Brownian increment would require hundreds of GB. We therefore
generate a small chunk of Brownian increments, update all schemes, discard the
chunk, and then move to the next chunk.

Why two passes?
---------------
The paper defines the fixed step h_mean by first running EA and averaging its
adaptive time steps. Therefore we proceed in two passes:

    Pass 1 computes the fine IF reference, EA, SIA, and h_mean.
    Pass 2 replays the same Brownian chunks and computes fixed EF_y, EF_x,
    IF, and FT using that h_mean.

The two passes see the same Brownian path because each chunk is regenerated
from a deterministic key based on the chunk number.
"""

from __future__ import annotations

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


# =============================================================================
# 1. Experiment configuration
# =============================================================================

# We begin by entering the CIR parameters from Fig. 2(a).
KAPPA = 2.0
LAMBDA = 0.05
SIGMA = 0.2
FINAL_TIME = 1.0

# Notice that the paper gives Y0, not X0. Since Y = sqrt(X), we use X0 = Y0^2.
INITIAL_Y = 0.02
INITIAL_X = INITIAL_Y**2

NUMBER_OF_PATHS = int(os.environ.get("FIG2A_M", "1000"))

# Use REFERENCE_POWER=18 for a quick smoke test. Use 25 for the paper reference.
REFERENCE_POWER = int(os.environ.get("FIG2A_REFERENCE_POWER", "25"))
REFERENCE_STEP = 2.0 ** (-REFERENCE_POWER)
NUMBER_OF_FINE_STEPS = 2**REFERENCE_POWER

# The adaptive schemes use hmax = 2^-i for i=4,...,9.
LEVELS_AS_PYTHON_LIST = [4, 5, 6, 7, 8, 9]
LEVELS = jnp.array(LEVELS_AS_PYTHON_LIST, dtype=jnp.int32)
NUMBER_OF_LEVELS = len(LEVELS_AS_PYTHON_LIST)

# The paper takes rho = 2^6, so hmin = hmax / rho.
RHO = 2**6

# Random seeds. We keep separate keys for the Brownian path and bridge samples.
SEED = int(os.environ.get("FIG2A_SEED", "2022"))
PATH_KEY = jax.random.PRNGKey(SEED)
BRIDGE_KEY_EA = jax.random.PRNGKey(SEED + 100_000)
BRIDGE_KEY_SIA = jax.random.PRNGKey(SEED + 200_000)
BRIDGE_KEY_FIXED = jax.random.PRNGKey(SEED + 300_000)

# The chunk size controls the memory/time tradeoff. A value of 2^14 is usually
# safe on a Kaggle P100. If the run is killed by an out-of-memory error, try 2^13.
FINE_STEPS_PER_CHUNK = min(int(os.environ.get("FIG2A_CHUNK_STEPS", str(2**14))), NUMBER_OF_FINE_STEPS)
NUMBER_OF_CHUNKS = NUMBER_OF_FINE_STEPS // FINE_STEPS_PER_CHUNK
assert NUMBER_OF_FINE_STEPS % FINE_STEPS_PER_CHUNK == 0


# =============================================================================
# 2. Lamperti coefficients
# =============================================================================

# The original CIR model is
#
#   dX_t = kappa (lambda - X_t) dt + sigma sqrt(X_t) dW_t.
#
# Applying Y = sqrt(X) transforms it into
#
#   dY = (alpha / Y + beta Y) dt + gamma dW.
#
# The coefficients are
#
#   alpha = (4 kappa lambda - sigma^2) / 8,
#   beta  = -kappa / 2,
#   gamma = sigma / 2.
#
ALPHA = (4.0 * KAPPA * LAMBDA - SIGMA**2) / 8.0
BETA = -KAPPA / 2.0
GAMMA = SIGMA / 2.0

# It will be useful to store hmax for all six levels in one array.
HMAX_BY_LEVEL = 2.0 ** (-LEVELS.astype(jnp.float64))


# =============================================================================
# 3. The numerical schemes used in Fig. 2(a)
# =============================================================================

# We now list the numerical schemes before implementing them. This makes it
# easier to connect the code below to the mathematical formulas in the paper.
#
# Reference IF
# ------------
# The reference solution is computed in the Lamperti variable Y on the fine grid
# h_ref = 2^-25. It uses the drift-implicit update
#
#   Y_{n+1} = Y_n + h (alpha / Y_{n+1} + beta Y_{n+1}) + gamma Delta W_n.
#
# Solving this equation gives the positive-root formula implemented by
# implicit_lamperti_step(). The reference value used in the RMSE is X_ref=Y_ref^2.
#
# EA: Explicit Adaptive
# ---------------------
# EA is an explicit Euler method in the Lamperti variable Y. The adaptive step is
#
#   h_{n+1} = max(h_min, h_max min(1, |Y_n|)),      h_min = h_max / rho.
#
# Given a Brownian increment over this adaptive interval, EA proposes
#
#   Y_{n+1} = Y_n + h (alpha / Y_n + beta Y_n) + gamma Delta W_n.
#
# If this proposal is non-positive, or if the minimum step has been selected, the
# same Brownian increment is retaken with the implicit backstop.
#
# SIA: Semi-Implicit Adaptive
# ---------------------------
# SIA uses the same adaptive times as EA, but treats the beta Y term implicitly:
#
#   Y_{n+1} = (Y_n + h alpha / Y_n + gamma Delta W_n) / (1 - beta h).
#
# It uses the same positivity backstop rule as EA.
#
# EF_y: Explicit Fixed in the Lamperti Variable
# ---------------------------------------------
# EF_y is the fixed-step explicit Euler method in Y, run with h=h_mean:
#
#   Y_{n+1} = Y_n + h (alpha / Y_n + beta Y_n) + gamma Delta W_n,
#   X_{n+1} = Y_{n+1}^2.
#
# This is the curve that matches the published Fig. 2(a)-style EF behaviour.
#
# EF_x: Explicit Fixed in the Original CIR Variable
# -------------------------------------------------
# EF_x is the direct-CIR Higham-Mao style method:
#
#   X_{n+1} = X_n + h kappa (lambda - X_n)
#             + sigma sqrt(|X_n|) Delta W_n.
#
# We include EF_x as a diagnostic because it usually tracks FT closely for these
# parameters. This is why the printed direct-CIR formula does not reproduce the
# black EF curve in Fig. 2(a).
#
# FT: Fully Truncated Euler
# -------------------------
# FT evolves a raw variable X_tilde and reports its positive part:
#
#   X_tilde_{n+1} = X_tilde_n + h kappa (lambda - X_tilde_n^+)
#                  + sigma sqrt(X_tilde_n^+) Delta W_n,
#   X_{n+1} = X_tilde_{n+1}^+.
#
# Fixed IF
# --------
# The fixed IF curve uses the same implicit_lamperti_step() formula as the
# reference, but at h=h_mean rather than h_ref.
#
# Brownian bridge coupling
# ------------------------
# EA, SIA and all fixed schemes usually land at times that are not fine grid
# points. When a target time tau lies in [t_i, t_{i+1}], we sample W_tau from the
# Brownian bridge conditional on W_i and W_{i+1}. This keeps the approximation
# on the same Brownian trajectory as the fine reference.


# =============================================================================
# 4. Small mathematical building blocks
# =============================================================================

def implicit_lamperti_step(y_old, brownian_increment, step_size):
    """One drift-implicit Lamperti step.

    It solves

        y_new = y_old + h * (alpha / y_new + beta * y_new) + gamma * dW

    We use the positive quadratic root, so the update remains positive.
    """
    # Rearranging the implicit equation gives
    #
    #   (1 - beta h) y_new^2 - (y_old + gamma dW) y_new - alpha h = 0.
    #
    # If we write u = y_old + gamma dW and d = 1 - beta h, then the positive
    # root is
    #
    #   y_new = u/(2d) + sqrt(u^2/(4d^2) + alpha h/d).
    u = y_old + GAMMA * brownian_increment
    denominator = 1.0 - BETA * step_size
    return (
        u / (2.0 * denominator)
        + jnp.sqrt(u * u / (4.0 * denominator * denominator) + ALPHA * step_size / denominator)
    )


def choose_adaptive_step_size(y_values, current_times):
    """Return the next EA/SIA step size and whether hmin was used.

    The paper's adaptive strategy is

        h = max(hmin, hmax * min(1, |Y|)),
        hmin = hmax / rho.

    The final step is shortened if necessary so that the scheme lands exactly
    at T.
    """
    # The unclipped adaptive rule is
    #
    #   h_prop = hmax min(1, |Y_n|),        h_min = hmax / rho.
    #
    # Then the actual step is
    #
    #   h = max(h_min, h_prop).
    proposed = HMAX_BY_LEVEL[:, None] * jnp.minimum(1.0, jnp.abs(y_values))
    minimum = (HMAX_BY_LEVEL / RHO)[:, None]
    used_minimum_step = proposed <= minimum
    chosen = jnp.where(used_minimum_step, minimum, proposed)

    # The final clipping implements
    #
    #   h <- min(h, T - t_n),
    #
    # so every path lands exactly on the terminal time T=1.
    remaining_time = FINAL_TIME - current_times
    return jnp.minimum(chosen, remaining_time), used_minimum_step


def sample_brownian_bridge_value(w_left, w_right, t_left, t_right, target_time, key):
    """Sample W(target_time) inside a fine reference interval.

    We know W(t_left) and W(t_right). For target_time in [t_left, t_right],
    the Brownian bridge conditional distribution is Normal with

        mean = W_left + theta * (W_right - W_left),
        var  = (tau - t_left) * (t_right - tau) / h_ref.

    Notice that target_time has shape (levels, paths), while w_left and w_right
    have shape (paths,). NumPy/JAX broadcasting then gives one bridge value per
    level and path.
    """
    # The clip only protects against tiny floating point roundoff at boundaries.
    tau = jnp.clip(target_time, t_left, t_right)

    # Brownian bridge formula on [t_left, t_right]:
    #
    #   theta = (tau - t_left) / h_ref,
    #   E[W_tau | W_left, W_right]
    #       = W_left + theta (W_right - W_left),
    #   Var[W_tau | W_left, W_right]
    #       = (tau - t_left)(t_right - tau) / h_ref.
    theta = (tau - t_left) / REFERENCE_STEP
    conditional_mean = w_left[None, :] + theta * (w_right - w_left)[None, :]
    conditional_variance = (tau - t_left) * (t_right - tau) / REFERENCE_STEP
    standard_normals = jax.random.normal(key, target_time.shape, dtype=jnp.float64)
    return conditional_mean + jnp.sqrt(jnp.maximum(conditional_variance, 0.0)) * standard_normals


def make_brownian_increment_chunk(chunk_number):
    """Generate one deterministic chunk of Brownian increments.

    Calling this function with the same chunk_number in pass 1 and pass 2 gives
    the same increments. This is how we replay the Brownian trajectory without
    storing it.
    """
    chunk_key = jax.random.fold_in(PATH_KEY, chunk_number)
    shape = (FINE_STEPS_PER_CHUNK, NUMBER_OF_PATHS)

    # Brownian increments satisfy
    #
    #   Delta W_i = sqrt(h_ref) Z_i,        Z_i ~ Normal(0, 1).
    return jax.random.normal(chunk_key, shape, dtype=jnp.float64) * jnp.sqrt(REFERENCE_STEP)


# =============================================================================
# 5. Pass 1: fine IF reference plus adaptive EA and SIA
# =============================================================================

INITIAL_ADAPTIVE_STEP, INITIAL_ADAPTIVE_USED_MINIMUM = choose_adaptive_step_size(
    jnp.full((NUMBER_OF_LEVELS, NUMBER_OF_PATHS), INITIAL_Y, dtype=jnp.float64),
    0.0,
)


def create_pass1_state():
    """Create all variables needed during pass 1.

    We keep two different kinds of arrays:

        reference variables:          (paths,)
        adaptive scheme variables:    (levels, paths)
    """
    level_path_shape = (NUMBER_OF_LEVELS, NUMBER_OF_PATHS)
    return {
        # Fine reference IF and the fine Brownian path value W(t).
        "reference_y": jnp.full((NUMBER_OF_PATHS,), INITIAL_Y, dtype=jnp.float64),
        "brownian_w": jnp.zeros((NUMBER_OF_PATHS,), dtype=jnp.float64),

        # Adaptive explicit Euler in Y. Each entry corresponds to one level and
        # one sample path.
        "ea_y": jnp.full(level_path_shape, INITIAL_Y, dtype=jnp.float64),
        "ea_last_w": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "ea_last_t": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "ea_target_t": INITIAL_ADAPTIVE_STEP,
        "ea_used_minimum_step": INITIAL_ADAPTIVE_USED_MINIMUM,
        "ea_number_of_steps": jnp.zeros(level_path_shape, dtype=jnp.int32),
        "ea_sum_of_steps": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "ea_bridge_key": BRIDGE_KEY_EA,

        # Semi-implicit adaptive method, stored with the same layout as EA.
        "sia_y": jnp.full(level_path_shape, INITIAL_Y, dtype=jnp.float64),
        "sia_last_w": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "sia_last_t": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "sia_target_t": INITIAL_ADAPTIVE_STEP,
        "sia_used_minimum_step": INITIAL_ADAPTIVE_USED_MINIMUM,
        "sia_number_of_steps": jnp.zeros(level_path_shape, dtype=jnp.int32),
        "sia_sum_of_steps": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "sia_bridge_key": BRIDGE_KEY_SIA,
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
    use_sia_formula,
):
    """Update EA or SIA if its next target time is inside this fine interval.

    Most fine reference intervals do not contain an adaptive landing time. In
    that case we only carry the state forward. When a landing time is present,
    we obtain the required Brownian value by bridge sampling and then apply the
    relevant numerical scheme.
    """
    target_is_inside_this_fine_interval = target_t <= t_right

    def land_at_target(_):
        next_key, normal_key = jax.random.split(bridge_key)

        # The adaptive method wants a Brownian increment from its previous
        # scheme time last_t to the new scheme time target_t. Since target_t is
        # usually not a fine reference grid point, we first sample W(target_t)
        # from the Brownian bridge inside the current fine interval.
        #
        # The scheme increment is then
        #
        #   Delta W_scheme = W(target_t) - W(last_t).
        target_w_raw = sample_brownian_bridge_value(
            w_left, w_right, t_left, t_right, target_t, normal_key
        )

        # Only schemes that are landing should use target_w_raw. The others keep
        # their previous Brownian position and state.
        target_w = jnp.where(target_is_inside_this_fine_interval, target_w_raw, last_w)
        step_size = target_t - last_t
        scheme_dW = target_w - last_w

        if use_sia_formula:
            # SIA proposal:
            #   Y_new = (Y + h alpha/Y + gamma dW) / (1 - beta h).
            #
            # Only the linear beta*Y part is placed on the left-hand side.
            # The singular alpha/Y part is still evaluated explicitly at the
            # old value Y_n.
            explicit_candidate = (
                y_old + step_size * ALPHA / y_old + GAMMA * scheme_dW
            ) / (1.0 - BETA * step_size)
        else:
            # EA proposal:
            #   Y_new = Y + h(alpha/Y + beta Y) + gamma dW.
            #
            # This is the ordinary explicit Euler-Maruyama step applied after
            # the Lamperti transform. The adaptive time step is what prevents
            # the singular alpha/Y drift from becoming too large near zero.
            explicit_candidate = (
                y_old
                + step_size * (ALPHA / y_old + BETA * y_old)
                + GAMMA * scheme_dW
            )

        # The adaptive schemes are hybrid methods. If the explicit proposal is
        # non-positive, or if the minimum step hmin has been reached, we keep
        # the same Brownian increment but retake the step using the positive
        # implicit method. This is the backstop.
        #
        # In symbols:
        #
        #   Y_{n+1} = explicit_candidate,
        #
        # unless explicit_candidate <= 0 or h_n = h_min, in which case
        #
        #   Y_{n+1} = implicit_lamperti_step(Y_n, Delta W_n, h_n).
        backstop_value = implicit_lamperti_step(y_old, scheme_dW, step_size)
        should_use_backstop = used_minimum_step | (explicit_candidate <= 0.0)
        y_after_step = jnp.where(should_use_backstop, backstop_value, explicit_candidate)

        y_new = jnp.where(target_is_inside_this_fine_interval, y_after_step, y_old)
        last_w_new = jnp.where(target_is_inside_this_fine_interval, target_w, last_w)
        last_t_new = jnp.where(target_is_inside_this_fine_interval, target_t, last_t)

        number_of_steps_new = number_of_steps + target_is_inside_this_fine_interval.astype(jnp.int32)
        sum_of_steps_new = sum_of_steps + jnp.where(
            target_is_inside_this_fine_interval, step_size, 0.0
        )

        # Once a path has landed, its next target time is determined from the
        # newly computed value. Paths that did not land keep their old target.
        #
        # Mathematically this is just
        #
        #   t_{n+1} = t_n + h(Y_{n+1}),
        #
        # with the terminal clipping already handled by choose_adaptive_step_size().
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


def pass1_one_fine_step(state, scan_input):
    """Advance pass 1 by one fine reference interval."""
    fine_brownian_increment, fine_index_after_step = scan_input
    t_right = fine_index_after_step.astype(jnp.float64) * REFERENCE_STEP
    t_left = t_right - REFERENCE_STEP

    w_left = state["brownian_w"]
    w_right = w_left + fine_brownian_increment

    reference_y_new = implicit_lamperti_step(
        state["reference_y"], fine_brownian_increment, REFERENCE_STEP
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
        y_old=state["ea_y"],
        last_w=state["ea_last_w"],
        last_t=state["ea_last_t"],
        target_t=state["ea_target_t"],
        used_minimum_step=state["ea_used_minimum_step"],
        number_of_steps=state["ea_number_of_steps"],
        sum_of_steps=state["ea_sum_of_steps"],
        w_left=w_left,
        w_right=w_right,
        t_left=t_left,
        t_right=t_right,
        bridge_key=state["ea_bridge_key"],
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
        y_old=state["sia_y"],
        last_w=state["sia_last_w"],
        last_t=state["sia_last_t"],
        target_t=state["sia_target_t"],
        used_minimum_step=state["sia_used_minimum_step"],
        number_of_steps=state["sia_number_of_steps"],
        sum_of_steps=state["sia_sum_of_steps"],
        w_left=w_left,
        w_right=w_right,
        t_left=t_left,
        t_right=t_right,
        bridge_key=state["sia_bridge_key"],
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


@jax.jit
def run_pass1_chunk(state, brownian_increment_chunk, first_fine_index_in_chunk):
    fine_indices_after_step = first_fine_index_in_chunk + jnp.arange(
        1, brownian_increment_chunk.shape[0] + 1, dtype=jnp.int32
    )
    new_state, _ = lax.scan(
        pass1_one_fine_step,
        state,
        (brownian_increment_chunk, fine_indices_after_step),
    )
    return new_state


def run_pass1():
    """Run the reference, EA, and SIA through all fine chunks."""
    state = create_pass1_state()
    start = time.perf_counter()

    for chunk_number in range(NUMBER_OF_CHUNKS):
        chunk = make_brownian_increment_chunk(chunk_number)
        first_index = chunk_number * FINE_STEPS_PER_CHUNK
        state = run_pass1_chunk(state, chunk, first_index)

        if (chunk_number + 1) % 128 == 0 or (chunk_number + 1) == NUMBER_OF_CHUNKS:
            state["reference_y"].block_until_ready()
            elapsed = time.perf_counter() - start
            print(f"pass 1: {chunk_number + 1}/{NUMBER_OF_CHUNKS} chunks, {elapsed:.1f}s")

    return state


# =============================================================================
# 6. Pass 2: fixed schemes at h_mean
# =============================================================================

def create_pass2_state(h_mean_by_level):
    """Create state for EF_y, EF_x, fixed IF, and FT."""
    level_path_shape = (NUMBER_OF_LEVELS, NUMBER_OF_PATHS)
    h_fixed_by_level = h_mean_by_level.astype(jnp.float64)
    first_target_time = jnp.broadcast_to(
        jnp.minimum(h_fixed_by_level[:, None], FINAL_TIME),
        level_path_shape,
    )

    return {
        "brownian_w": jnp.zeros((NUMBER_OF_PATHS,), dtype=jnp.float64),

        # EF_y: explicit Euler in Lamperti Y, then square at the end.
        "ef_y_value": jnp.full(level_path_shape, INITIAL_Y, dtype=jnp.float64),

        # EF_x: direct-CIR Higham-Mao style explicit method.
        "ef_x_value": jnp.full(level_path_shape, INITIAL_X, dtype=jnp.float64),

        # Full truncation and fixed implicit Lamperti.
        "ft_raw_x": jnp.full(level_path_shape, INITIAL_X, dtype=jnp.float64),
        "fixed_if_y": jnp.full(level_path_shape, INITIAL_Y, dtype=jnp.float64),

        # Last scheme time and Brownian value used by the fixed schemes.
        "last_w": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "last_t": jnp.zeros(level_path_shape, dtype=jnp.float64),
        "target_t": first_target_time,
        "fixed_step": h_fixed_by_level,
        "bridge_key": BRIDGE_KEY_FIXED,
    }


def update_fixed_schemes_when_they_land(
    state,
    w_left,
    w_right,
    t_left,
    t_right,
):
    """Update EF_y, EF_x, fixed IF, and FT if their target time lands here.

    The fixed schemes all use the same target time for a given level, but the
    target is generally not a fine reference grid point. We therefore bridge to
    the target time before applying the fixed-step update.
    """
    target_is_inside_this_fine_interval = state["target_t"] <= t_right

    def land_at_target(_):
        next_key, normal_key = jax.random.split(state["bridge_key"])

        # As in the adaptive pass, the fixed schemes need W at their target
        # time, which will usually be between two fine reference grid points.
        # The Brownian bridge gives the correct conditional distribution.
        #
        # The fixed-scheme Brownian increment is
        #
        #   Delta W_n = W(t_n + h_mean) - W(t_n).
        target_w_raw = sample_brownian_bridge_value(
            w_left, w_right, t_left, t_right, state["target_t"], normal_key
        )
        target_w = jnp.where(target_is_inside_this_fine_interval, target_w_raw, state["last_w"])

        step_size = state["target_t"] - state["last_t"]
        scheme_dW = target_w - state["last_w"]

        # EF_y: explicit Euler in the Lamperti variable:
        #   Y_new = Y + h(alpha/Y + beta Y) + gamma dW.
        #
        # This method evolves Y and is converted to X only at the end by
        # squaring. This is the explicit fixed method whose convergence curve
        # aligns with EA/IF/SIA in Fig. 2(a).
        ef_y_candidate = (
            state["ef_y_value"]
            + step_size * (ALPHA / state["ef_y_value"] + BETA * state["ef_y_value"])
            + GAMMA * scheme_dW
        )
        ef_y_new = jnp.where(
            target_is_inside_this_fine_interval,
            ef_y_candidate,
            state["ef_y_value"],
        )

        # EF_x: direct-CIR Higham-Mao style explicit Euler:
        #   X_new = X + h kappa(lambda-X) + sigma sqrt(|X|) dW.
        #
        # This method evolves X directly. The absolute value under the square
        # root keeps the diffusion real even if the explicit state becomes
        # negative. For the Fig. 2(a) parameter set it tends to sit almost on
        # top of FT, which is why we plot it separately as EF_x.
        ef_x_candidate = (
            state["ef_x_value"]
            + step_size * KAPPA * (LAMBDA - state["ef_x_value"])
            + SIGMA * jnp.sqrt(jnp.abs(state["ef_x_value"])) * scheme_dW
        )
        ef_x_new = jnp.where(
            target_is_inside_this_fine_interval,
            ef_x_candidate,
            state["ef_x_value"],
        )

        # FT: full truncation in X:
        #   X_tilde_new = X_tilde + h kappa(lambda-X_tilde^+)
        #                 + sigma sqrt(X_tilde^+) dW.
        #
        # The state used in the drift and diffusion is truncated to its
        # positive part. The raw value is still carried forward, and the final
        # reported approximation is max(raw, 0).
        #
        # Here x^+ means max(x, 0).
        positive_part = jnp.maximum(state["ft_raw_x"], 0.0)
        ft_candidate = (
            state["ft_raw_x"]
            + step_size * KAPPA * (LAMBDA - positive_part)
            + SIGMA * jnp.sqrt(positive_part) * scheme_dW
        )
        ft_new = jnp.where(
            target_is_inside_this_fine_interval,
            ft_candidate,
            state["ft_raw_x"],
        )

        # Fixed IF: implicit Lamperti step at h_mean.
        #
        # This has the same formula as the fine reference, but it is evaluated
        # only at the coarser fixed step h_mean. It is included as one of the
        # comparison schemes in Fig. 2(a).
        #
        # In symbols:
        #
        #   Y_{n+1} = implicit_lamperti_step(Y_n, Delta W_n, h_mean).
        fixed_if_candidate = implicit_lamperti_step(
            state["fixed_if_y"], scheme_dW, step_size
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

        # The fixed schemes keep using the same step size for a given level.
        # The last step is shortened if needed so that the path lands exactly
        # at T.
        #
        # Thus
        #
        #   t_{n+1} = min(t_n + h_mean, T).
        next_target = jnp.minimum(
            last_t_new + state["fixed_step"][:, None],
            FINAL_TIME,
        )
        target_t_new = jnp.where(
            target_is_inside_this_fine_interval,
            next_target,
            state["target_t"],
        )

        return {
            "brownian_w": w_right,
            "ef_y_value": ef_y_new,
            "ef_x_value": ef_x_new,
            "ft_raw_x": ft_new,
            "fixed_if_y": fixed_if_new,
            "last_w": last_w_new,
            "last_t": last_t_new,
            "target_t": target_t_new,
            "fixed_step": state["fixed_step"],
            "bridge_key": next_key,
        }

    def do_not_land(_):
        return {
            "brownian_w": w_right,
            "ef_y_value": state["ef_y_value"],
            "ef_x_value": state["ef_x_value"],
            "ft_raw_x": state["ft_raw_x"],
            "fixed_if_y": state["fixed_if_y"],
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


def pass2_one_fine_step(state, scan_input):
    """Advance pass 2 by one fine reference interval."""
    fine_brownian_increment, fine_index_after_step = scan_input
    t_right = fine_index_after_step.astype(jnp.float64) * REFERENCE_STEP
    t_left = t_right - REFERENCE_STEP

    w_left = state["brownian_w"]
    w_right = w_left + fine_brownian_increment

    new_state = update_fixed_schemes_when_they_land(
        state=state,
        w_left=w_left,
        w_right=w_right,
        t_left=t_left,
        t_right=t_right,
    )
    return new_state, None


@jax.jit
def run_pass2_chunk(state, brownian_increment_chunk, first_fine_index_in_chunk):
    fine_indices_after_step = first_fine_index_in_chunk + jnp.arange(
        1, brownian_increment_chunk.shape[0] + 1, dtype=jnp.int32
    )
    new_state, _ = lax.scan(
        pass2_one_fine_step,
        state,
        (brownian_increment_chunk, fine_indices_after_step),
    )
    return new_state


def run_pass2(h_mean_by_level):
    """Run all fixed schemes at the EA-derived h_mean values."""
    state = create_pass2_state(h_mean_by_level)
    start = time.perf_counter()

    for chunk_number in range(NUMBER_OF_CHUNKS):
        chunk = make_brownian_increment_chunk(chunk_number)
        first_index = chunk_number * FINE_STEPS_PER_CHUNK
        state = run_pass2_chunk(state, chunk, first_index)

        if (chunk_number + 1) % 128 == 0 or (chunk_number + 1) == NUMBER_OF_CHUNKS:
            state["ef_y_value"].block_until_ready()
            elapsed = time.perf_counter() - start
            print(f"pass 2: {chunk_number + 1}/{NUMBER_OF_CHUNKS} chunks, {elapsed:.1f}s")

    return state


# =============================================================================
# 7. Error calculation and plotting
# =============================================================================

def root_mean_square_error(scheme_x_values, reference_x_values):
    """Compute the RMSE over paths, returning one value for each level."""
    # For each level i we estimate
    #
    #   RMSE_i = sqrt( (1/M) sum_{m=1}^M |X_i^{(m)}(T) - X_ref^{(m)}(T)|^2 ).
    path_errors = scheme_x_values - reference_x_values[None, :]
    return jnp.sqrt(jnp.mean(path_errors * path_errors, axis=1))


def fit_loglog_order(x_values, y_values):
    """Compute the slope of log(error) versus log(step size)."""
    return float(jnp.polyfit(jnp.log(x_values), jnp.log(y_values), 1)[0])


def plot_results(h_mean, errors):
    style = {
        "EA": dict(color="blue", marker="o", ls="-", mfc="none"),
        "EF_y": dict(color="black", marker="+", ls="-"),
        "EF_x": dict(color="0.45", marker="s", ls="--", mfc="none"),
        "IF": dict(color="magenta", marker="o", ls="-.", mfc="none"),
        "SIA": dict(color="cyan", marker="x", ls="--"),
        "FT": dict(color="red", marker="x", ls=":"),
    }

    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    for scheme_name in ("EA", "EF_y", "EF_x", "IF", "SIA", "FT"):
        order = fit_loglog_order(h_mean, errors[scheme_name])
        label = f"{scheme_name} ({order:.2f})"
        ax.loglog(h_mean, errors[scheme_name], label=label, **style[scheme_name])

    # The following two lines are visual guides. On a log-log plot, a line
    #
    #   error = C h^p
    #
    # has slope p. We draw p=1 and p=1/2 to compare with the observed orders.
    ax.loglog(
        h_mean,
        0.55 * errors["EA"][-1] * (h_mean / h_mean[-1]) ** 1.0,
        "--",
        color="tab:orange",
        lw=1,
    )
    ax.loglog(
        h_mean,
        2.7 * errors["EA"][-1] * (h_mean / h_mean[-1]) ** 0.5,
        "-",
        color="steelblue",
        lw=1,
    )

    ax.set_xlabel(r"$h_{\mathrm{mean}}$")
    ax.set_ylabel("RMSE")
    ax.set_title(r"KLM Fig. 2(a), bridge streaming JAX, $h=2^{-25}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    import pathlib
    if os.path.isdir("/kaggle/working"):
        fig_dir = csv_dir = pathlib.Path("/kaggle/working")
    else:
        if "__file__" in globals():
            repo_root = pathlib.Path(__file__).resolve().parents[1]
        else:
            repo_root = pathlib.Path.cwd()
        fig_dir = repo_root / "figures"
        csv_dir = repo_root / "results"
        fig_dir.mkdir(exist_ok=True)
        csv_dir.mkdir(exist_ok=True)
    stem = f"fig2a_bridge_streaming_jax_h{REFERENCE_POWER}"
    fig.savefig(fig_dir / f"{stem}.png", dpi=220)
    fig.savefig(fig_dir / f"{stem}.pdf")
    print("saved", fig_dir / f"{stem}.png")
    import csv as _csv
    with open(csv_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["level", "h_mean", "EA", "EF_y", "EF_x", "IF", "SIA", "FT"])
        for i, level in enumerate(LEVELS_AS_PYTHON_LIST):
            w.writerow([level, float(h_mean[i])] + [float(errors[k][i]) for k in ("EA", "EF_y", "EF_x", "IF", "SIA", "FT")])
    print("saved", csv_dir / f"{stem}.csv")
    plt.show()


# =============================================================================
# 8. Main program
# =============================================================================

def main():
    print("devices:", jax.devices())
    print(
        f"M={NUMBER_OF_PATHS}, "
        f"h_ref=2^-{REFERENCE_POWER}, "
        f"N_REF={NUMBER_OF_FINE_STEPS:,}, "
        f"chunks={NUMBER_OF_CHUNKS}, "
        f"chunk_steps={FINE_STEPS_PER_CHUNK}"
    )
    print("Brownian-bridge landing version. First chunk includes JIT compilation.")

    pass1_state = run_pass1()

    # The Lamperti schemes evolve Y, while Fig. 2 reports error in X.
    # Since Y = sqrt(X), every Lamperti terminal value is converted by
    #
    #   X(T) = Y(T)^2.
    reference_x = pass1_state["reference_y"] ** 2
    ea_x = pass1_state["ea_y"] ** 2
    sia_x = pass1_state["sia_y"] ** 2

    # The paper defines the fixed step size by averaging the EA time steps:
    #
    #   h_mean = (1/M) sum_{m=1}^M (1/N_m) sum_{n=1}^{N_m} h_n^{(m)}.
    #
    # Here ea_sum_of_steps / ea_number_of_steps is the pathwise average, and
    # the mean over axis=1 averages over the M paths for each hmax level.
    h_mean = jnp.mean(
        pass1_state["ea_sum_of_steps"] / pass1_state["ea_number_of_steps"],
        axis=1,
    )
    h_mean.block_until_ready()
    print("h_mean:", h_mean)

    pass2_state = run_pass2(h_mean)

    # Convert the fixed Lamperti schemes back to X and apply the FT output map
    #
    #   X_EF_y = Y_EF_y^2,       X_IF = Y_IF^2,       X_FT = max(X_tilde, 0).
    #
    # EF_x already evolves X directly.
    ef_y_x = pass2_state["ef_y_value"] ** 2
    ef_x_x = pass2_state["ef_x_value"]
    fixed_if_x = pass2_state["fixed_if_y"] ** 2
    ft_x = jnp.maximum(pass2_state["ft_raw_x"], 0.0)

    errors = {
        "EA": root_mean_square_error(ea_x, reference_x),
        "EF_y": root_mean_square_error(ef_y_x, reference_x),
        "EF_x": root_mean_square_error(ef_x_x, reference_x),
        "IF": root_mean_square_error(fixed_if_x, reference_x),
        "SIA": root_mean_square_error(sia_x, reference_x),
        "FT": root_mean_square_error(ft_x, reference_x),
    }
    for value in errors.values():
        value.block_until_ready()

    print("\nlevel h_mean EA EF_y EF_x IF SIA FT")
    for level_index, level in enumerate(LEVELS_AS_PYTHON_LIST):
        print(
            level,
            float(h_mean[level_index]),
            float(errors["EA"][level_index]),
            float(errors["EF_y"][level_index]),
            float(errors["EF_x"][level_index]),
            float(errors["IF"][level_index]),
            float(errors["SIA"][level_index]),
            float(errors["FT"][level_index]),
        )

    plot_results(h_mean, errors)


if __name__ == "__main__":
    main()
