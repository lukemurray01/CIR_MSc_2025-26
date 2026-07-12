import ast
import json
from pathlib import Path


NOTEBOOK_DIR = Path("notebooks/kaggle")

STANDALONE_NOTEBOOKS = [
    "kaggle_cir_benchmark_suite.ipynb",
    "kaggle_cir_benchmark_suite_JAX.ipynb",
    "kaggle_reference_sensitivity.ipynb",
    "kaggle_reference_ladder_JAX.ipynb",
    "kaggle_four_scheme_regime_rates_JAX.ipynb",
    "kaggle_klm_fig2a.ipynb",
    "kaggle_klm_fig2a_JAX.ipynb",
    "kaggle_klm_fig3_full.ipynb",
    "kaggle_klm_fig3_full_JAX.ipynb",
    "kaggle_kl_adaptive_splitting_paper.ipynb",
    "kaggle_kl_adaptive_splitting_paper_JAX.ipynb",
]


def notebook_source(name: str) -> str:
    notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
    return "".join(notebook["cells"][0]["source"])


def test_kaggle_notebooks_are_single_code_cell_and_parse():
    for name in STANDALONE_NOTEBOOKS:
        notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))

        assert len(notebook["cells"]) == 1, name
        assert notebook["cells"][0]["cell_type"] == "code", name
        ast.parse("".join(notebook["cells"][0]["source"]))


def test_standalone_kaggle_notebooks_do_not_clone_remote_branch():
    forbidden_remote_checkout_markers = [
        "!git clone",
        "git clone https://",
        "git clone git@",
        "subprocess.run([\"git\"",
        "feature/thesis-code-suite",
        "FILES_JSON",
        "write_repo_snapshot",
        "from src",
        "import src",
        "from klm_jax",
        "import klm_jax",
    ]

    for name in STANDALONE_NOTEBOOKS:
        source = notebook_source(name)

        for marker in forbidden_remote_checkout_markers:
            assert marker not in source


def test_cir_benchmark_notebooks_are_monolithic_not_repo_snapshot_runners():
    cpu_source = notebook_source("kaggle_cir_benchmark_suite.ipynb")
    jax_source = notebook_source("kaggle_cir_benchmark_suite_JAX.ipynb")

    for source in [cpu_source, jax_source]:
        assert "CIR_SUITE_RUN_MODE" in source
        assert "run_strong_error_suite" in source
        assert "adaptive-soft-zero" in source
        assert "scheme_variant" in source
        assert "kl_alpha" in source
        assert 'REGIME_LIST = ["A", "B", "C", "D", "E"]' in source
        assert "CIR_SUITE_REFERENCE_N_STEPS" in source
        assert "klm" in source.lower()
        assert "hh" in source.lower()

    assert "16384" in cpu_source

    assert "def hh_milstein_step" in cpu_source
    assert "def kl_adaptive_terminal_from_fine_dW" in cpu_source
    assert "def klm_backstop_terminal_from_fine_dW" in cpu_source
    assert "def run_distributional" in cpu_source
    assert "def run_cev_experiment" in cpu_source
    assert "import jax" not in cpu_source

    assert "import jax" in jax_source
    assert "jax.config.update(\"jax_enable_x64\", True)" in jax_source
    assert "def hh_terminal_from_dW_jax" in jax_source
    assert "def kl_adaptive_terminal_from_fine_dW_jax" in jax_source
    assert "def klm_terminal_from_fine_dW_jax" in jax_source
    assert "use_kl_adaptive = alpha < 0.0" in jax_source
    assert "CIR_SUITE_REFERENCE_POWER" in jax_source
    assert 'default_power=22' in jax_source
    # Production default matches the thesis methodology chapter: 20000
    # Brownian-coupled paths (batched; checkpoint/resume covers the runtime).
    assert '"20000"' in jax_source
    assert "CIR_SUITE_PATH_BATCH_SIZE" in jax_source
    assert "stream_strong_chunk" in jax_source
    assert "run_config.json" in jax_source
    assert "jax_strong_error_batch_contributions.csv" in jax_source
    assert "CIR_SUITE_RESUME" in jax_source
    assert "RUN_CONFIG_HASH" in jax_source
    assert "latest_partial.zip" in jax_source
    assert "CIR_SUITE_FIG3_REFERENCE_POWER" in jax_source
    assert "fig3_config(1000, 16" in jax_source
    assert "def run_fig3_order_sweep" in jax_source


def test_reference_sensitivity_notebook_is_standalone_gate():
    source = notebook_source("kaggle_reference_sensitivity.ipynb")

    assert "REFERENCE_SENSITIVITY_RUN_MODE" in source
    assert "REFSENS_STRONG_REFERENCE_N_STEPS" in source
    assert "4096,16384,32768" in source
    assert "REFSENS_FIG3_REFERENCE_POWERS" in source
    assert "14,16,18" in source
    assert "def run_strong_reference_sensitivity" in source
    assert "def run_fig3_reference_sensitivity" in source
    assert "strong_reference_sensitivity.csv" in source
    assert "fig3_reference_sensitivity.csv" in source
    assert "adaptive-soft-zero" in source


def test_reference_ladder_notebook_is_streamed_deep_gate():
    source = notebook_source("kaggle_reference_ladder_JAX.ipynb")

    # Deep-reference ladder: streamed HH rungs on ONE shared Brownian path.
    assert "CIR_SUITE_REFERENCE_GRID" in source
    assert "4096,16384,32768" in source
    assert "CIR_SUITE_ADAPTIVE_GRID_STEPS" in source
    assert "CIR_SUITE_CHUNK_STEPS" in source
    assert "CIR_SUITE_MEM_BUDGET_GB" in source
    assert "def streaming_parity_check" in source
    assert "def run_strong_error_suite" in source
    assert "adaptive-soft-zero" in source
    # Gate outputs consumed by experiments/fig_order_summary.py.
    assert "strong_reference_sensitivity_orders.csv" in source
    assert "jax_fig_order_vs_delta_summary" in source
    # Thesis-grade sampling and the post-2026-07-05 ProjEuler floor.
    assert 'N_PATHS = 20000' in source
    assert "0.5 * sigma * np.sqrt(dt)" in source


def test_klm_fig2a_notebooks_are_numpy_cpu_and_jax_gpu_split():
    cpu_source = notebook_source("kaggle_klm_fig2a.ipynb")
    jax_source = notebook_source("kaggle_klm_fig2a_JAX.ipynb")

    assert "import jax" not in cpu_source
    assert "Standalone NumPy KLM Fig. 2(a)" in cpu_source
    assert 'FIG2A_RUN_MODE", "full"' in cpu_source
    assert 'FIG2A_REFERENCE_POWER", "25"' in cpu_source
    assert "def run_pass1" in cpu_source
    assert "def run_pass2" in cpu_source
    assert "def make_brownian_increment_chunk" in cpu_source

    assert "import jax" in jax_source
    assert "jax.config.update(\"jax_enable_x64\", True)" in jax_source
    assert 'FIG2A_RUN_MODE", "full"' in jax_source
    assert 'FIG2A_REFERENCE_POWER", "25"' in jax_source
    assert "FIG2A_CHUNK_STEPS" in jax_source
    assert "def run_pass1" in jax_source
    assert "def run_pass2" in jax_source


def test_klm_fig3_full_notebooks_embed_jax_scripts_directly():
    for name in [
        "kaggle_klm_fig3_full.ipynb",
        "kaggle_klm_fig3_full_JAX.ipynb",
    ]:
        source = notebook_source(name)

        assert "jax.config.update(\"jax_enable_x64\", True)" in source
        assert "matplotlib.use(\"Agg\")" in source
        assert "RUN_MODE" in source
        assert "/kaggle/working" in source
        assert 'FIG3_FULL_DIAG_RUN_MODE", "full"' in source
        assert 'FIG3_FULL_DIAG_REFERENCE_POWER", "25"' in source
        assert "FIG3_FULL_DIAG_CHUNK_STEPS" in source

    assert "FIG3_FULL_DIAG_REFERENCE_POWER" in notebook_source(
        "kaggle_klm_fig3_full.ipynb"
    )



def test_four_scheme_regime_rate_notebook_is_standalone_jax():
    source = notebook_source("kaggle_four_scheme_regime_rates_JAX.ipynb")

    assert 'PREFIX = "FOUR_SCHEME_REGIME_RATE"' in source
    assert 'f"{PREFIX}_RUN_MODE"' in source
    assert "default_power=22" in source
    assert "h_ref = 2^-22" in source
    assert "FTE" in source
    assert "ProjEuler" in source
    assert "KL" in source
    assert "KLM" in source
    assert "2.0,0.2" in source
    assert "REGIME_A_VALUES" in source
    assert "four_scheme_regime_rates.csv" in source
    assert "four_scheme_regime_rates_jax.pdf" in source
    assert "run_config.json" in source
    assert "four_scheme_regime_rate_batch_contributions.csv" in source
    assert "FOUR_SCHEME_REGIME_RATE_RESUME" in source
    assert "RUN_CONFIG_HASH" in source
    assert "latest_partial.zip" in source
    assert "stream_strong_chunk" in source
    assert "import jax" in source
