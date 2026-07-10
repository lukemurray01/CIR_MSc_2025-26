# CPU (NumPy) vs JAX timing for the thesis schemes.
#
# Times full Monte Carlo terminal sampling -- noise generation INCLUDED, on
# the respective device -- for FTE, ProjEuler, KL, KLM and BLT, across a
# ladder of path counts.  JAX calls are warmed up once (compilation excluded)
# and synchronised with block_until_ready; every row records the JAX backend
# so CPU-only runs of this script are labelled as JAX-CPU, not GPU.  For the
# thesis GPU numbers run this same script on the Kaggle P100 and quote the
# backend column.
#
# The regime defaults to B (alpha > 0), where all five schemes are valid
# (uniform KL requires alpha >= 0; the KLM implicit backstop requires
# alpha > 0).  Requesting KL in an alpha < 0 regime fails fast.
#
# Usage:
#   uv run python experiments/run_cpu_gpu_timing.py
#   uv run python experiments/run_cpu_gpu_timing.py --n-paths 1000 10000
#   uv run python experiments/run_cpu_gpu_timing.py --jax-platform cuda
#
# Getting a GPU backend for the JAX rows:
#   - Kaggle (recommended): create a notebook with the P100 accelerator,
#     upload/clone this repo, and run this script unchanged; Kaggle's JAX
#     build detects the GPU and every CSV row records backend='gpu'.  The
#     P100 has 1:2 FP64 throughput, which matters because all kernels run
#     in float64 (jax_enable_x64).  Prefer --n-paths ladders up to 1e6 on
#     the GPU; the crossover against CPU sits near 1e4 paths.
#   - Native Windows has no CUDA JAX wheels; use WSL2 with
#     `pip install -U "jax[cuda12]"` if a local NVIDIA GPU is available.
#   - --jax-platform sets JAX_PLATFORMS before JAX is imported: use
#     `--jax-platform cuda` to fail loudly if the GPU is not picked up
#     (rather than silently timing the CPU backend), or `--jax-platform cpu`
#     to force a CPU-backend JAX run for comparison.
#
# Outputs:
#   results/cpu_gpu_timing.csv
#   figures/cpu_gpu_timing.pdf (+ .png preview)

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.samplers.blt_splitting import blt_terminal_from_noise
from src.samplers.full_truncation_euler import fte_terminal_from_dW
from src.samplers.kelly_lord import kl_uniform_terminal_from_dW
from src.samplers.klm_backstop import klm_backstop_terminal_from_fine_dW
from src.samplers.projected_euler import projected_euler_terminal_from_dW
from src.utils.io import config_path, figure_path, results_path
from src.utils.rng import make_brownian_increments, make_brownian_increments_with_infima, make_rng

ALL_SCHEMES = ["FTE", "ProjEuler", "KL", "KLM", "BLT"]

SCHEME_COLOURS = {
    "FTE": "#4477AA",
    "ProjEuler": "#AA3377",
    "KL": "#EE6677",
    "KLM": "#7B1FA2",
    "BLT": "#CC6677",
}


def load_config(filename):
    with open(config_path(filename), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="CPU/JAX scheme timing.")
    parser.add_argument("--schemes", nargs="+", default=ALL_SCHEMES,
                        choices=ALL_SCHEMES)
    parser.add_argument("--regime", default="B")
    parser.add_argument("--n-paths", nargs="+", type=int,
                        default=[1_000, 10_000, 100_000])
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Fixed-step count; also the KLM fine grid.")
    parser.add_argument("--klm-h-max-denominator", type=int, default=32)
    parser.add_argument(
        "--klm-rho", type=float, default=16.0,
        help="KLM floor parameter; default keeps h_min = h_max/rho >= dt_fine "
             "on the default 512-step grid so the fine-grid guard stays quiet.",
    )
    parser.add_argument("--reps", type=int, default=3,
                        help="Repetitions; the minimum wall time is reported.")
    parser.add_argument(
        "--jax-platform", choices=["cpu", "gpu", "cuda"], default=None,
        help="Force the JAX backend via JAX_PLATFORMS (set before JAX is "
             "imported). 'cuda' fails loudly when no GPU is available "
             "instead of silently falling back to CPU.",
    )
    return parser.parse_args()


def numpy_runners(params, T, n_steps, h_max, seed, klm_rho):
    """Callables n_paths -> terminal array; noise generation inside."""
    dt = T / n_steps
    common = dict(
        X0=params["x0"], kappa=params["kappa"],
        theta=params["theta"], sigma=params["sigma"],
    )

    def fte(n_paths):
        rng = make_rng(seed)
        dW = make_brownian_increments(rng, n_paths, n_steps, dt)
        return fte_terminal_from_dW(dt=dt, dW=dW, **common)

    def proj(n_paths):
        rng = make_rng(seed)
        dW = make_brownian_increments(rng, n_paths, n_steps, dt)
        return projected_euler_terminal_from_dW(dt=dt, dW=dW, **common)

    def kl(n_paths):
        rng = make_rng(seed)
        dW = make_brownian_increments(rng, n_paths, n_steps, dt)
        return kl_uniform_terminal_from_dW(dt=dt, dW=dW, **common)

    def klm(n_paths):
        rng = make_rng(seed)
        dW = make_brownian_increments(rng, n_paths, n_steps, dt)
        terminal, _ = klm_backstop_terminal_from_fine_dW(
            T=T, h_max=h_max, dW_fine=dW, rho=klm_rho, **common
        )
        return terminal

    def blt(n_paths):
        rng = make_rng(seed)
        dW, m = make_brownian_increments_with_infima(rng, n_paths, n_steps, dt)
        return blt_terminal_from_noise(dt=dt, dW=dW, m=m, **common)

    return {"FTE": fte, "ProjEuler": proj, "KL": kl, "KLM": klm, "BLT": blt}


def jax_runners(params, T, n_steps, h_max, seed, klm_rho):
    import jax

    from src.jax_schemes import (
        blt_terminal_from_noise_jax,
        brownian_increments_jax,
        brownian_increments_with_infima_jax,
        fte_terminal_from_dW_jax,
        kl_uniform_terminal_from_dW_jax,
        klm_backstop_terminal_from_fine_dW_jax,
        projected_euler_terminal_from_dW_jax,
    )

    dt = T / n_steps
    common = dict(
        X0=params["x0"], kappa=params["kappa"],
        theta=params["theta"], sigma=params["sigma"],
    )
    key = jax.random.PRNGKey(seed)

    def with_increments(kernel):
        def run(n_paths):
            dW = brownian_increments_jax(key, n_paths, n_steps, dt)
            return kernel(dt=dt, dW=dW, **common)
        return run

    def klm(n_paths):
        dW = brownian_increments_jax(key, n_paths, n_steps, dt)
        terminal, _ = klm_backstop_terminal_from_fine_dW_jax(
            T=T, h_max=h_max, dW_fine=dW, rho=klm_rho, **common
        )
        return terminal

    def blt(n_paths):
        dW, m = brownian_increments_with_infima_jax(key, n_paths, n_steps, dt)
        return blt_terminal_from_noise_jax(dt=dt, dW=dW, m=m, **common)

    return {
        "FTE": with_increments(fte_terminal_from_dW_jax),
        "ProjEuler": with_increments(projected_euler_terminal_from_dW_jax),
        "KL": with_increments(kl_uniform_terminal_from_dW_jax),
        "KLM": klm,
        "BLT": blt,
    }


def time_call(fn, n_paths, reps, sync=None):
    best = np.inf
    for _ in range(reps):
        start = time.perf_counter()
        out = fn(n_paths)
        if sync is not None:
            sync(out)
        best = min(best, time.perf_counter() - start)
    return best


def main():
    args = get_args()

    # Must happen before the first JAX import anywhere in the process.
    if args.jax_platform is not None:
        os.environ["JAX_PLATFORMS"] = args.jax_platform

    regimes_cfg = load_config("regimes.yaml")
    experiments_cfg = load_config("experiments.yaml")
    shared = regimes_cfg["shared"]
    master_seed = experiments_cfg["shared"]["master_seed"]
    T = experiments_cfg["shared"]["T"]

    sigma = regimes_cfg["regimes"][args.regime]["sigma"]
    alpha = (4.0 * shared["kappa"] * shared["theta"] - sigma**2) / 8.0
    if alpha < 0.0 and "KL" in args.schemes:
        raise SystemExit(
            f"uniform KL is undefined in regime {args.regime} (alpha < 0); "
            "time KL in regimes A-C or drop it via --schemes"
        )
    params = {
        "kappa": shared["kappa"], "theta": shared["theta"],
        "x0": shared["x0"], "sigma": sigma,
    }
    h_max = 1.0 / args.klm_h_max_denominator

    try:
        import jax

        jax_backend = jax.default_backend()
        jax_device = str(jax.devices()[0])
        have_jax = True
    except ImportError:
        jax_backend, jax_device, have_jax = "unavailable", "unavailable", False
    except (RuntimeError, AssertionError) as exc:
        # JAX raises RuntimeError or (some versions) a bare AssertionError
        # when JAX_PLATFORMS names a platform it cannot initialise.
        raise SystemExit(
            f"JAX could not initialise the requested platform "
            f"({args.jax_platform!r}): {exc!r}\n"
            "Install a CUDA-enabled JAX (WSL2/Kaggle) or drop --jax-platform."
        ) from exc

    if args.jax_platform in ("gpu", "cuda") and have_jax and jax_backend == "cpu":
        raise SystemExit(
            "requested a GPU JAX backend but jax.default_backend() is 'cpu'; "
            "install a CUDA-enabled JAX (WSL2/Kaggle) or drop --jax-platform"
        )

    print(f"Regime {args.regime}, n_steps={args.n_steps}, "
          f"KLM h_max=1/{args.klm_h_max_denominator}; JAX backend: {jax_backend}")

    np_run = numpy_runners(params, T, args.n_steps, h_max, master_seed, args.klm_rho)
    if have_jax:
        jx_run = jax_runners(params, T, args.n_steps, h_max, master_seed, args.klm_rho)

        def sync(out):
            out.block_until_ready()

    rows = []
    for scheme in args.schemes:
        for n_paths in args.n_paths:
            wall_np = time_call(np_run[scheme], n_paths, args.reps)
            rows.append(
                {
                    "scheme": scheme, "regime": args.regime,
                    "n_paths": n_paths, "n_steps": args.n_steps,
                    "impl": "numpy", "backend": "cpu",
                    "device": "host", "reps": args.reps,
                    "wall_s": wall_np,
                    "paths_steps_per_s": n_paths * args.n_steps / wall_np,
                    "includes_rng": True,
                }
            )

            if have_jax:
                # Warm-up call compiles the kernel; excluded from timing.
                sync(jx_run[scheme](n_paths))
                wall_jx = time_call(jx_run[scheme], n_paths, args.reps, sync=sync)
                rows.append(
                    {
                        "scheme": scheme, "regime": args.regime,
                        "n_paths": n_paths, "n_steps": args.n_steps,
                        "impl": "jax", "backend": jax_backend,
                        "device": jax_device, "reps": args.reps,
                        "wall_s": wall_jx,
                        "paths_steps_per_s": n_paths * args.n_steps / wall_jx,
                        "includes_rng": True,
                    }
                )
                speedup = wall_np / wall_jx
            else:
                speedup = np.nan

            print(f"  {scheme:<10} n_paths={n_paths:>7}: numpy {wall_np:.3f}s"
                  + (f", jax[{jax_backend}] {wall_jx:.3f}s, speedup {speedup:.2f}x"
                     if have_jax else ""))

    csv_path = results_path("cpu_gpu_timing.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    fig_path = figure_path("cpu_gpu_timing.pdf")
    plot_timing(rows, args, jax_backend, fig_path)
    print(f"wrote {fig_path}")


CPU_BAR_COLOUR = "#4477AA"
JAX_BAR_COLOUR = "#EE7733"


def plot_timing(rows, args, jax_backend, out_path):
    """Grouped bar chart: one panel per path count, per scheme a CPU bar and
    a JAX bar side by side, with the speed-up annotated above each pair."""
    path_counts = sorted({r["n_paths"] for r in rows})
    fig, axes = plt.subplots(
        1, len(path_counts), figsize=(4.2 * len(path_counts), 4.6), squeeze=False
    )
    axes = axes[0]

    bar_width = 0.38
    x = np.arange(len(args.schemes))

    for ax, n_paths in zip(axes, path_counts):
        wall = {
            impl: [
                next(
                    (r["wall_s"] for r in rows
                     if r["scheme"] == s and r["impl"] == impl
                     and r["n_paths"] == n_paths),
                    np.nan,
                )
                for s in args.schemes
            ]
            for impl in ("numpy", "jax")
        }

        cpu_bars = ax.bar(
            x - bar_width / 2, wall["numpy"], bar_width,
            color=CPU_BAR_COLOUR, label="NumPy (CPU)",
        )
        jax_bars = ax.bar(
            x + bar_width / 2, wall["jax"], bar_width,
            color=JAX_BAR_COLOUR, hatch="//", label=f"JAX ({jax_backend})",
        )

        for bars in (cpu_bars, jax_bars):
            ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)

        # Speed-up annotation above each scheme pair.
        for i, (t_np, t_jx) in enumerate(zip(wall["numpy"], wall["jax"])):
            if np.isfinite(t_np) and np.isfinite(t_jx) and t_jx > 0:
                top = max(t_np, t_jx)
                ax.annotate(
                    f"{t_np / t_jx:.1f}x",
                    xy=(i, top),
                    xytext=(0, 11),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="0.25",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(args.schemes, fontsize=8)
        ax.set_ylabel("wall time (s), best of reps")
        ax.set_title(f"{n_paths:,} paths")
        ax.margins(y=0.18)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"CPU vs JAX wall time per scheme -- regime {args.regime}, "
        f"{args.n_steps} steps, FP64, noise generation included; "
        f"speed-up numpy/jax above each pair (JAX backend: {jax_backend})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    fig.savefig(Path(out_path).with_suffix(".png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
